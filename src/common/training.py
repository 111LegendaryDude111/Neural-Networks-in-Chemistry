from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "results" / ".matplotlib"),
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split

from src.common.config import (
    HOLDOUT_TEST_SIZE,
    INNER_FOLDS,
    OUTER_FOLDS,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_DIR,
    TARGET_COLUMNS,
    TaskConfig,
)
from src.common.cv import describe_evaluation_strategy, make_splitter
from src.common.data import prepare_task_data
from src.common.io import ensure_dir, ensure_project_dirs, write_dataframe, write_json, write_text
from src.common.logging_utils import setup_logger
from src.common.metrics import (
    classification_scoring,
    compute_classification_metrics,
    compute_regression_metrics,
    leaderboard_ascending,
    metric_columns,
    regression_scoring,
    summarize_metric_frame,
)
from src.common.models import ModelSpec, resolve_model_specs
from src.common.preprocessing import (
    build_model_pipeline,
    clip_regression_predictions,
    extract_final_estimator,
    prefix_param_grid,
)

matplotlib.use("Agg")


@dataclass
class EvaluatedModel:
    model_name: str
    mode: str
    summary: dict
    fold_metrics: pd.DataFrame
    final_best_params: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray | None
    importance_df: pd.DataFrame | None
    component_importances: dict[str, pd.DataFrame] | None = None


def build_task_parser(task_config: TaskConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=task_config.title)
    parser.add_argument(
        "--evaluation-strategy",
        choices=("nested", "holdout"),
        default="nested",
        help="Preferred protocol is nested 5x3; holdout uses 20%% test + 3-fold inner CV.",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Subset of models to run.")
    parser.add_argument(
        "--skip-catboost",
        action="store_true",
        help="Skip CatBoost even if the dependency is available.",
    )
    parser.add_argument("--outer-folds", type=int, default=OUTER_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    parser.add_argument("--test-size", type=float, default=HOLDOUT_TEST_SIZE)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--top-k-importance", type=int, default=20)
    return parser


def scoring_config(task_config: TaskConfig):
    if task_config.problem_type == "regression":
        return regression_scoring(), "mae"
    include_pr_auc = task_config.primary_metric == "pr_auc"
    return classification_scoring(include_pr_auc=include_pr_auc), task_config.primary_metric


def prediction_scores(fitted_model, x_frame: pd.DataFrame):
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(x_frame)[:, 1]
    if hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(x_frame)
    return None


def clean_param_names(parameters: dict) -> dict:
    cleaned = {}
    for key, value in parameters.items():
        normalized = key
        for prefix in ("regressor__", "estimator__"):
            while normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        cleaned[normalized] = value
    return cleaned


def fit_search(
    model_spec: ModelSpec,
    task_config: TaskConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    inner_folds: int,
    seed: int,
):
    estimator = build_model_pipeline(
        estimator=model_spec.estimator_factory(),
        requires_scaling=model_spec.requires_scaling,
        use_log_target=task_config.problem_type == "regression" and task_config.use_log_target,
    )
    param_grid = prefix_param_grid(
        model_spec.param_grid,
        use_log_target=task_config.problem_type == "regression" and task_config.use_log_target,
    )
    scoring, refit_metric = scoring_config(task_config)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=make_splitter(task_config.problem_type, inner_folds, seed),
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    return search


def compute_task_metrics(task_config: TaskConfig, y_true, y_pred, y_score=None) -> dict[str, float]:
    if task_config.problem_type == "regression":
        return compute_regression_metrics(y_true, y_pred)
    return compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        include_pr_auc=task_config.primary_metric == "pr_auc",
    )


def extract_feature_importance(fitted_model, feature_columns: list[str]) -> pd.DataFrame | None:
    estimator = extract_final_estimator(fitted_model)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        values = coefficients[0] if coefficients.ndim > 1 else coefficients
    else:
        return None

    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": values,
            "abs_importance": np.abs(values),
        }
    )
    return importance_frame.sort_values("abs_importance", ascending=False).reset_index(drop=True)


def summarize_model_run(
    task_config: TaskConfig,
    model_name: str,
    mode: str,
    fold_metrics: pd.DataFrame,
    best_params: dict,
    evaluation_strategy: str,
    fit_seconds: float,
) -> dict:
    summary = {
        "task_name": task_config.name,
        "problem_type": task_config.problem_type,
        "target_column": task_config.target_column,
        "primary_metric": task_config.primary_metric,
        "evaluation_strategy": evaluation_strategy,
        "model_name": model_name,
        "mode": mode,
        "fit_seconds": fit_seconds,
        "best_params_json": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
    }
    summary.update(
        summarize_metric_frame(
            fold_metrics,
            metric_columns(
                task_config.problem_type,
                include_pr_auc=task_config.primary_metric == "pr_auc",
            ),
        )
    )
    return summary


def evaluate_direct_model(
    x_frame: pd.DataFrame,
    y_target: pd.Series,
    feature_columns: list[str],
    task_config: TaskConfig,
    model_spec: ModelSpec,
    evaluation_strategy: str,
    inner_folds: int,
    outer_folds: int,
    test_size: float,
    seed: int,
) -> EvaluatedModel:
    start_time = time.perf_counter()
    fold_rows: list[dict] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_score_all: list[np.ndarray] = []

    if evaluation_strategy == "nested":
        outer_splitter = make_splitter(task_config.problem_type, outer_folds, seed)
        split_iterator = outer_splitter.split(x_frame, y_target)

        for fold_index, (train_idx, test_idx) in enumerate(split_iterator, start=1):
            x_train = x_frame.iloc[train_idx]
            x_test = x_frame.iloc[test_idx]
            y_train = y_target.iloc[train_idx]
            y_test = y_target.iloc[test_idx]

            search = fit_search(
                model_spec=model_spec,
                task_config=task_config,
                x_train=x_train,
                y_train=y_train,
                inner_folds=inner_folds,
                seed=seed + fold_index,
            )
            y_pred = search.best_estimator_.predict(x_test)
            y_score = None
            if task_config.problem_type == "regression":
                y_pred = clip_regression_predictions(y_pred)
            else:
                y_score = prediction_scores(search.best_estimator_, x_test)

            metrics = compute_task_metrics(task_config, y_test, y_pred, y_score=y_score)
            metrics.update(
                {
                    "fold": fold_index,
                    "best_params_json": json.dumps(
                        clean_param_names(search.best_params_),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
            fold_rows.append(metrics)
            y_true_all.append(np.asarray(y_test))
            y_pred_all.append(np.asarray(y_pred))
            if y_score is not None:
                y_score_all.append(np.asarray(y_score))
    else:
        stratify = y_target if task_config.problem_type == "classification" else None
        x_train, x_test, y_train, y_test = train_test_split(
            x_frame,
            y_target,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
        search = fit_search(
            model_spec=model_spec,
            task_config=task_config,
            x_train=x_train,
            y_train=y_train,
            inner_folds=inner_folds,
            seed=seed,
        )
        y_pred = search.best_estimator_.predict(x_test)
        y_score = None
        if task_config.problem_type == "regression":
            y_pred = clip_regression_predictions(y_pred)
        else:
            y_score = prediction_scores(search.best_estimator_, x_test)

        metrics = compute_task_metrics(task_config, y_test, y_pred, y_score=y_score)
        metrics.update(
            {
                "fold": "holdout",
                "best_params_json": json.dumps(
                    clean_param_names(search.best_params_),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        fold_rows.append(metrics)
        y_true_all.append(np.asarray(y_test))
        y_pred_all.append(np.asarray(y_pred))
        if y_score is not None:
            y_score_all.append(np.asarray(y_score))

    final_search = fit_search(
        model_spec=model_spec,
        task_config=task_config,
        x_train=x_frame,
        y_train=y_target,
        inner_folds=inner_folds,
        seed=seed + 999,
    )
    final_best_params = clean_param_names(final_search.best_params_)
    importance_df = extract_feature_importance(final_search.best_estimator_, feature_columns)
    fold_metrics = pd.DataFrame(fold_rows)

    return EvaluatedModel(
        model_name=model_spec.name,
        mode="direct",
        summary=summarize_model_run(
            task_config=task_config,
            model_name=model_spec.name,
            mode="direct",
            fold_metrics=fold_metrics,
            best_params=final_best_params,
            evaluation_strategy=evaluation_strategy,
            fit_seconds=time.perf_counter() - start_time,
        ),
        fold_metrics=fold_metrics,
        final_best_params=final_best_params,
        y_true=np.concatenate(y_true_all),
        y_pred=np.concatenate(y_pred_all),
        y_score=np.concatenate(y_score_all) if y_score_all else None,
        importance_df=importance_df,
    )


def evaluate_indirect_si_model(
    x_frame: pd.DataFrame,
    y_ic50: pd.Series,
    y_cc50: pd.Series,
    y_si: pd.Series,
    feature_columns: list[str],
    model_spec: ModelSpec,
    evaluation_strategy: str,
    inner_folds: int,
    outer_folds: int,
    test_size: float,
    seed: int,
) -> EvaluatedModel:
    start_time = time.perf_counter()
    fold_rows: list[dict] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    ratio_task = TaskConfig(
        name="regression_si",
        title="Regression: SI indirect",
        problem_type="regression",
        target_column=TARGET_COLUMNS["SI"],
        primary_metric="mae",
        use_log_target=True,
    )

    if evaluation_strategy == "nested":
        outer_splitter = make_splitter("regression", outer_folds, seed)
        split_iterator = outer_splitter.split(x_frame, y_si)

        for fold_index, (train_idx, test_idx) in enumerate(split_iterator, start=1):
            x_train = x_frame.iloc[train_idx]
            x_test = x_frame.iloc[test_idx]

            ic50_train = y_ic50.iloc[train_idx]
            ic50_test = y_ic50.iloc[test_idx]
            cc50_train = y_cc50.iloc[train_idx]
            cc50_test = y_cc50.iloc[test_idx]
            si_test = y_si.iloc[test_idx]

            ic50_search = fit_search(
                model_spec=model_spec,
                task_config=TaskConfig(
                    name="regression_ic50",
                    title="Regression: IC50",
                    problem_type="regression",
                    target_column=TARGET_COLUMNS["IC50"],
                    primary_metric="mae",
                    use_log_target=True,
                ),
                x_train=x_train,
                y_train=ic50_train,
                inner_folds=inner_folds,
                seed=seed + fold_index,
            )
            cc50_search = fit_search(
                model_spec=model_spec,
                task_config=TaskConfig(
                    name="regression_cc50",
                    title="Regression: CC50",
                    problem_type="regression",
                    target_column=TARGET_COLUMNS["CC50"],
                    primary_metric="mae",
                    use_log_target=True,
                ),
                x_train=x_train,
                y_train=cc50_train,
                inner_folds=inner_folds,
                seed=seed + 100 + fold_index,
            )

            ic50_pred = clip_regression_predictions(ic50_search.best_estimator_.predict(x_test))
            cc50_pred = clip_regression_predictions(cc50_search.best_estimator_.predict(x_test))
            si_pred = cc50_pred / np.clip(ic50_pred, 1e-8, None)
            metrics = compute_regression_metrics(si_test, si_pred)
            metrics.update(
                {
                    "fold": fold_index,
                    "best_params_json": json.dumps(
                        {
                            "ic50_component": clean_param_names(ic50_search.best_params_),
                            "cc50_component": clean_param_names(cc50_search.best_params_),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
            fold_rows.append(metrics)
            y_true_all.append(np.asarray(si_test))
            y_pred_all.append(np.asarray(si_pred))
    else:
        x_train, x_test, ic50_train, ic50_test, cc50_train, cc50_test, si_train, si_test = train_test_split(
            x_frame,
            y_ic50,
            y_cc50,
            y_si,
            test_size=test_size,
            random_state=seed,
        )
        ic50_search = fit_search(
            model_spec=model_spec,
            task_config=TaskConfig(
                name="regression_ic50",
                title="Regression: IC50",
                problem_type="regression",
                target_column=TARGET_COLUMNS["IC50"],
                primary_metric="mae",
                use_log_target=True,
            ),
            x_train=x_train,
            y_train=ic50_train,
            inner_folds=inner_folds,
            seed=seed,
        )
        cc50_search = fit_search(
            model_spec=model_spec,
            task_config=TaskConfig(
                name="regression_cc50",
                title="Regression: CC50",
                problem_type="regression",
                target_column=TARGET_COLUMNS["CC50"],
                primary_metric="mae",
                use_log_target=True,
            ),
            x_train=x_train,
            y_train=cc50_train,
            inner_folds=inner_folds,
            seed=seed + 100,
        )
        ic50_pred = clip_regression_predictions(ic50_search.best_estimator_.predict(x_test))
        cc50_pred = clip_regression_predictions(cc50_search.best_estimator_.predict(x_test))
        si_pred = cc50_pred / np.clip(ic50_pred, 1e-8, None)
        metrics = compute_regression_metrics(si_test, si_pred)
        metrics.update(
            {
                "fold": "holdout",
                "best_params_json": json.dumps(
                    {
                        "ic50_component": clean_param_names(ic50_search.best_params_),
                        "cc50_component": clean_param_names(cc50_search.best_params_),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        fold_rows.append(metrics)
        y_true_all.append(np.asarray(si_test))
        y_pred_all.append(np.asarray(si_pred))

    ic50_final = fit_search(
        model_spec=model_spec,
        task_config=TaskConfig(
            name="regression_ic50",
            title="Regression: IC50",
            problem_type="regression",
            target_column=TARGET_COLUMNS["IC50"],
            primary_metric="mae",
            use_log_target=True,
        ),
        x_train=x_frame,
        y_train=y_ic50,
        inner_folds=inner_folds,
        seed=seed + 999,
    )
    cc50_final = fit_search(
        model_spec=model_spec,
        task_config=TaskConfig(
            name="regression_cc50",
            title="Regression: CC50",
            problem_type="regression",
            target_column=TARGET_COLUMNS["CC50"],
            primary_metric="mae",
            use_log_target=True,
        ),
        x_train=x_frame,
        y_train=y_cc50,
        inner_folds=inner_folds,
        seed=seed + 1_099,
    )
    fold_metrics = pd.DataFrame(fold_rows)
    final_best_params = {
        "ic50_component": clean_param_names(ic50_final.best_params_),
        "cc50_component": clean_param_names(cc50_final.best_params_),
    }
    component_importances = {
        "ic50_component": extract_feature_importance(ic50_final.best_estimator_, feature_columns),
        "cc50_component": extract_feature_importance(cc50_final.best_estimator_, feature_columns),
    }

    return EvaluatedModel(
        model_name=model_spec.name,
        mode="indirect",
        summary=summarize_model_run(
            task_config=ratio_task,
            model_name=model_spec.name,
            mode="indirect",
            fold_metrics=fold_metrics,
            best_params=final_best_params,
            evaluation_strategy=evaluation_strategy,
            fit_seconds=time.perf_counter() - start_time,
        ),
        fold_metrics=fold_metrics,
        final_best_params=final_best_params,
        y_true=np.concatenate(y_true_all),
        y_pred=np.concatenate(y_pred_all),
        y_score=None,
        importance_df=None,
        component_importances=component_importances,
    )


def save_confusion_matrix_artifact(task_dir: Path, y_true, y_pred) -> Path:
    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Winner Confusion Matrix")
    figure.tight_layout()
    output_path = task_dir / "winner_confusion_matrix.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_evaluated_model(task_dir: Path, run: EvaluatedModel, top_k_importance: int) -> dict:
    write_dataframe(task_dir / f"{run.model_name}_{run.mode}_fold_metrics.csv", run.fold_metrics)
    write_json(task_dir / f"{run.model_name}_{run.mode}_summary.json", run.summary)

    saved_artifacts = {}
    if run.importance_df is not None:
        importance_path = task_dir / f"{run.model_name}_{run.mode}_feature_importance.csv"
        write_dataframe(importance_path, run.importance_df.head(top_k_importance))
        saved_artifacts["feature_importance_path"] = str(importance_path)

    if run.component_importances:
        component_paths = {}
        for component_name, component_frame in run.component_importances.items():
            if component_frame is None:
                continue
            component_path = task_dir / f"{run.model_name}_{run.mode}_{component_name}_feature_importance.csv"
            write_dataframe(component_path, component_frame.head(top_k_importance))
            component_paths[component_name] = str(component_path)
        if component_paths:
            saved_artifacts["component_feature_importance_paths"] = component_paths

    return saved_artifacts


def render_task_report(
    task_config: TaskConfig,
    data_contract: dict,
    leakage_report: dict,
    leaderboard: pd.DataFrame,
    winner_row: pd.Series,
    strategy_text: str,
    label_distribution: dict | None = None,
) -> str:
    lines = [
        f"# {task_config.title}",
        "",
        "## Methodology",
        f"- Evaluation strategy: `{strategy_text}`",
        f"- Dataset path: `{data_contract['dataset_path']}`",
        f"- Dataset checksum (SHA256): `{data_contract['checksum_sha256']}`",
        f"- Feature count after leakage guard: `{data_contract['feature_count']}`",
        f"- Leakage validation status: `{leakage_report['status']}`",
        f"- Primary metric: `{task_config.primary_metric}`",
    ]

    if task_config.problem_type == "regression":
        lines.append(f"- Target log1p transform: `{task_config.use_log_target}`")
    if label_distribution is not None:
        lines.append(f"- Label balance: `{label_distribution}`")

    lines.extend(
        [
            "",
            "## Winner",
            f"- Model: `{winner_row['model_name']}`",
            f"- Mode: `{winner_row['mode']}`",
            f"- {task_config.primary_metric}: `{winner_row[task_config.primary_metric]:.6f}`",
        ]
    )

    for metric in metric_columns(
        task_config.problem_type,
        include_pr_auc=task_config.primary_metric == "pr_auc",
    ):
        if metric != task_config.primary_metric and metric in winner_row:
            lines.append(f"- {metric}: `{winner_row[metric]:.6f}`")

    lines.extend(
        [
            "",
            "## Leaderboard",
            leaderboard.to_markdown(index=False),
            "",
            "## Leakage Checklist",
        ]
    )
    for item in leakage_report["checklist"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def run_supervised_task(task_config: TaskConfig, args) -> dict:
    ensure_project_dirs()
    task_dir = ensure_dir(RESULTS_DIR / task_config.name)
    logger = setup_logger(task_config.name, task_dir / "run.log")
    prepared = prepare_task_data(task_config)
    write_json(RESULTS_DIR / "data_contract.json", prepared["data_contract"])
    write_json(task_dir / "leakage_check.json", prepared["leakage_report"])

    label_distribution = None
    if task_config.problem_type == "classification":
        label_distribution = prepared["y"].value_counts().sort_index().to_dict()

    model_specs = resolve_model_specs(
        task_config.problem_type,
        selected_models=args.models,
        include_catboost=not args.skip_catboost,
    )

    logger.info("Running %s with models: %s", task_config.name, [spec.name for spec in model_specs])

    model_runs: list[EvaluatedModel] = []
    for model_spec in model_specs:
        logger.info("Evaluating model %s", model_spec.name)
        run = evaluate_direct_model(
            x_frame=prepared["X"],
            y_target=prepared["y"],
            feature_columns=prepared["feature_columns"],
            task_config=task_config,
            model_spec=model_spec,
            evaluation_strategy=args.evaluation_strategy,
            inner_folds=args.inner_folds,
            outer_folds=args.outer_folds,
            test_size=args.test_size,
            seed=args.random_seed,
        )
        save_evaluated_model(task_dir, run, args.top_k_importance)
        model_runs.append(run)

    leaderboard = pd.DataFrame([run.summary for run in model_runs])
    leaderboard = leaderboard.sort_values(
        by=task_config.primary_metric,
        ascending=leaderboard_ascending(task_config.primary_metric),
    ).reset_index(drop=True)
    write_dataframe(task_dir / "leaderboard.csv", leaderboard)

    winner_row = leaderboard.iloc[0]
    winner_run = next(
        run for run in model_runs if run.model_name == winner_row["model_name"] and run.mode == winner_row["mode"]
    )

    summary = {
        "task_name": task_config.name,
        "title": task_config.title,
        "problem_type": task_config.problem_type,
        "target_column": task_config.target_column,
        "threshold": task_config.threshold,
        "primary_metric": task_config.primary_metric,
        "evaluation_strategy": args.evaluation_strategy,
        "random_seed": args.random_seed,
        "data_contract_path": str(RESULTS_DIR / "data_contract.json"),
        "leaderboard_path": str(task_dir / "leaderboard.csv"),
        "winner": winner_row.to_dict(),
        "label_distribution": label_distribution,
    }

    if task_config.problem_type == "classification":
        confusion_path = save_confusion_matrix_artifact(task_dir, winner_run.y_true, winner_run.y_pred)
        summary["winner_confusion_matrix_path"] = str(confusion_path)

    if winner_run.importance_df is not None:
        importance_path = task_dir / "winner_feature_importance.csv"
        write_dataframe(importance_path, winner_run.importance_df.head(args.top_k_importance))
        summary["winner_feature_importance_path"] = str(importance_path)

    write_json(task_dir / "summary.json", summary)
    report_text = render_task_report(
        task_config=task_config,
        data_contract=prepared["data_contract"],
        leakage_report=prepared["leakage_report"],
        leaderboard=leaderboard,
        winner_row=winner_row,
        strategy_text=describe_evaluation_strategy(
            args.evaluation_strategy,
            args.outer_folds,
            args.inner_folds,
            args.test_size,
        ),
        label_distribution=label_distribution,
    )
    write_text(REPORTS_DIR / f"{task_config.name}.md", report_text)
    logger.info("Finished %s. Winner: %s", task_config.name, winner_row["model_name"])
    return summary


def run_regression_si_task(task_config: TaskConfig, args) -> dict:
    ensure_project_dirs()
    task_dir = ensure_dir(RESULTS_DIR / task_config.name)
    direct_dir = ensure_dir(task_dir / "direct")
    indirect_dir = ensure_dir(task_dir / "indirect")
    logger = setup_logger(task_config.name, task_dir / "run.log")

    prepared = prepare_task_data(task_config)
    write_json(RESULTS_DIR / "data_contract.json", prepared["data_contract"])
    write_json(task_dir / "leakage_check.json", prepared["leakage_report"])

    model_specs = resolve_model_specs(
        "regression",
        selected_models=args.models,
        include_catboost=not args.skip_catboost,
    )
    logger.info("Running SI direct vs indirect with models: %s", [spec.name for spec in model_specs])

    direct_runs: list[EvaluatedModel] = []
    indirect_runs: list[EvaluatedModel] = []

    for model_spec in model_specs:
        logger.info("Evaluating direct SI model %s", model_spec.name)
        direct_run = evaluate_direct_model(
            x_frame=prepared["X"],
            y_target=prepared["y"],
            feature_columns=prepared["feature_columns"],
            task_config=task_config,
            model_spec=model_spec,
            evaluation_strategy=args.evaluation_strategy,
            inner_folds=args.inner_folds,
            outer_folds=args.outer_folds,
            test_size=args.test_size,
            seed=args.random_seed,
        )
        save_evaluated_model(direct_dir, direct_run, args.top_k_importance)
        direct_runs.append(direct_run)

        logger.info("Evaluating indirect SI model %s", model_spec.name)
        indirect_run = evaluate_indirect_si_model(
            x_frame=prepared["X"],
            y_ic50=prepared["dataframe"][TARGET_COLUMNS["IC50"]].astype(float),
            y_cc50=prepared["dataframe"][TARGET_COLUMNS["CC50"]].astype(float),
            y_si=prepared["dataframe"][TARGET_COLUMNS["SI"]].astype(float),
            feature_columns=prepared["feature_columns"],
            model_spec=model_spec,
            evaluation_strategy=args.evaluation_strategy,
            inner_folds=args.inner_folds,
            outer_folds=args.outer_folds,
            test_size=args.test_size,
            seed=args.random_seed,
        )
        save_evaluated_model(indirect_dir, indirect_run, args.top_k_importance)
        indirect_runs.append(indirect_run)

    leaderboard = pd.DataFrame([run.summary for run in direct_runs + indirect_runs])
    leaderboard = leaderboard.sort_values(by="mae", ascending=True).reset_index(drop=True)
    write_dataframe(task_dir / "leaderboard.csv", leaderboard)

    best_direct = leaderboard[leaderboard["mode"] == "direct"].iloc[0]
    best_indirect = leaderboard[leaderboard["mode"] == "indirect"].iloc[0]
    overall_winner = leaderboard.iloc[0]
    summary = {
        "task_name": task_config.name,
        "title": task_config.title,
        "problem_type": task_config.problem_type,
        "target_column": task_config.target_column,
        "primary_metric": task_config.primary_metric,
        "evaluation_strategy": args.evaluation_strategy,
        "random_seed": args.random_seed,
        "leaderboard_path": str(task_dir / "leaderboard.csv"),
        "data_contract_path": str(RESULTS_DIR / "data_contract.json"),
        "best_direct": best_direct.to_dict(),
        "best_indirect": best_indirect.to_dict(),
        "winner": overall_winner.to_dict(),
    }

    direct_winner_run = next(
        run for run in direct_runs if run.model_name == best_direct["model_name"] and run.mode == "direct"
    )
    indirect_winner_run = next(
        run for run in indirect_runs if run.model_name == best_indirect["model_name"] and run.mode == "indirect"
    )

    if direct_winner_run.importance_df is not None:
        direct_importance_path = task_dir / "winner_feature_importance_direct.csv"
        write_dataframe(direct_importance_path, direct_winner_run.importance_df.head(args.top_k_importance))
        summary["winner_feature_importance_direct_path"] = str(direct_importance_path)

    if indirect_winner_run.component_importances:
        for component_name, component_frame in indirect_winner_run.component_importances.items():
            if component_frame is None:
                continue
            component_path = task_dir / f"winner_feature_importance_indirect_{component_name}.csv"
            write_dataframe(component_path, component_frame.head(args.top_k_importance))
            summary.setdefault("winner_feature_importance_indirect_paths", {})[component_name] = str(component_path)

    write_json(task_dir / "summary.json", summary)

    report_lines = [
        f"# {task_config.title}",
        "",
        "## Methodology",
        f"- Evaluation strategy: `{describe_evaluation_strategy(args.evaluation_strategy, args.outer_folds, args.inner_folds, args.test_size)}`",
        f"- Dataset path: `{prepared['data_contract']['dataset_path']}`",
        f"- Leakage validation status: `{prepared['leakage_report']['status']}`",
        "- Direct model: `SI ~ descriptors`",
        "- Indirect model: `SI_hat = CC50_hat / IC50_hat`",
        "- Target log1p transform on component regressions: `True`",
        "",
        "## Direct vs Indirect",
        f"- Best direct model: `{best_direct['model_name']}` with `MAE={best_direct['mae']:.6f}`",
        f"- Best indirect model: `{best_indirect['model_name']}` with `MAE={best_indirect['mae']:.6f}`",
        f"- Overall winner: `{overall_winner['mode']}::{overall_winner['model_name']}`",
        "",
        "## Combined Leaderboard",
        leaderboard.to_markdown(index=False),
        "",
        "## Leakage Checklist",
    ]
    for item in prepared["leakage_report"]["checklist"]:
        report_lines.append(f"- {item}")
    write_text(REPORTS_DIR / f"{task_config.name}.md", "\n".join(report_lines))
    logger.info(
        "Finished %s. Best direct=%s, best indirect=%s, overall=%s::%s",
        task_config.name,
        best_direct["model_name"],
        best_indirect["model_name"],
        overall_winner["mode"],
        overall_winner["model_name"],
    )
    return summary
