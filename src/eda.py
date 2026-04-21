from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "results" / ".matplotlib"),
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common.config import CLASSIFICATION_TASK_NAMES, REPORTS_DIR, RESULTS_DIR, TARGET_COLUMNS, TASKS
from src.common.data import build_data_contract, find_dataset_path, load_dataset
from src.common.io import ensure_dir, write_dataframe, write_json, write_text
from src.common.preprocessing import get_feature_columns, get_feature_frame

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDA для датасета химической активности.")
    parser.add_argument("--top-k-correlations", type=int, default=20)
    return parser


def build_target_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    target_columns = list(TARGET_COLUMNS.values())
    summary = dataframe[target_columns].describe().T
    summary["skew"] = dataframe[target_columns].skew()
    summary["iqr"] = summary["75%"] - summary["25%"]
    return summary.reset_index().rename(columns={"index": "target"})


def build_missing_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_counts = dataframe.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if missing_counts.empty:
        return pd.DataFrame(columns=["feature", "missing_count", "missing_share"])

    return pd.DataFrame(
        {
            "feature": missing_counts.index,
            "missing_count": missing_counts.values.astype(int),
            "missing_share": (missing_counts.values / len(dataframe)).astype(float),
        }
    )


def build_class_balance(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_name in CLASSIFICATION_TASK_NAMES:
        task = TASKS[task_name]
        value_counts = dataframe[task.target_column].gt(task.threshold).value_counts().sort_index()
        for label, count in value_counts.items():
            rows.append(
                {
                    "task_name": task_name,
                    "target_column": task.target_column,
                    "threshold": task.threshold,
                    "label": int(label),
                    "count": int(count),
                    "share": float(count / len(dataframe)),
                }
            )
    return pd.DataFrame(rows)


def build_outlier_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_column in TARGET_COLUMNS.values():
        series = dataframe[target_column].astype(float)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        rows.append(
            {
                "target": target_column,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_count": int(outlier_mask.sum()),
                "outlier_share": float(outlier_mask.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_low_variance_summary(feature_frame: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "feature": feature_frame.columns,
            "variance": feature_frame.var(numeric_only=True).reindex(feature_frame.columns).values,
            "unique_values": feature_frame.nunique(dropna=False).reindex(feature_frame.columns).values,
            "missing_count": feature_frame.isna().sum().reindex(feature_frame.columns).values,
        }
    )
    return summary.sort_values(["variance", "unique_values", "missing_count"]).reset_index(drop=True)


def build_high_correlation_pairs(feature_frame: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation_matrix = feature_frame.corr().abs()
    pairs = []
    for row_index, left_feature in enumerate(correlation_matrix.columns):
        for right_feature in correlation_matrix.columns[row_index + 1 :]:
            correlation = correlation_matrix.loc[left_feature, right_feature]
            if correlation >= threshold:
                pairs.append(
                    {
                        "feature_left": left_feature,
                        "feature_right": right_feature,
                        "abs_correlation": float(correlation),
                    }
                )

    pairs_frame = pd.DataFrame(pairs)
    if pairs_frame.empty:
        return pd.DataFrame(columns=["feature_left", "feature_right", "abs_correlation"])
    return pairs_frame.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def build_top_target_correlations(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    top_k_correlations: int,
) -> pd.DataFrame:
    rows = []
    for target_column in TARGET_COLUMNS.values():
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = dataframe[feature_columns].corrwith(dataframe[target_column]).dropna()
        strongest = correlations.abs().sort_values(ascending=False).head(top_k_correlations)
        for feature_name in strongest.index:
            rows.append(
                {
                    "target_column": target_column,
                    "feature": feature_name,
                    "correlation": float(correlations.loc[feature_name]),
                    "abs_correlation": float(abs(correlations.loc[feature_name])),
                }
            )
    return pd.DataFrame(rows)


def save_target_plots(eda_dir: Path, dataframe: pd.DataFrame) -> None:
    target_columns = list(TARGET_COLUMNS.values())

    figure, axes = plt.subplots(3, 1, figsize=(10, 12))
    for axis, target_column in zip(axes, target_columns):
        sns.histplot(dataframe[target_column], kde=True, ax=axis, bins=30, color="#2c7fb8")
        axis.set_title(f"Распределение: {target_column}")
        axis.set_xlabel(target_column)
    figure.tight_layout()
    figure.savefig(eda_dir / "target_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, target_column in zip(axes, target_columns):
        sns.boxplot(y=dataframe[target_column], ax=axis, color="#7fcdbb")
        axis.set_title(f"Выбросы по {target_column}")
    figure.tight_layout()
    figure.savefig(eda_dir / "target_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(dataframe[target_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axis)
    axis.set_title("Связь между целевыми переменными")
    figure.tight_layout()
    figure.savefig(eda_dir / "target_correlations.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_missing_plot(eda_dir: Path, missing_summary: pd.DataFrame) -> None:
    if missing_summary.empty:
        return

    figure, axis = plt.subplots(figsize=(10, 4))
    sns.barplot(data=missing_summary, x="feature", y="missing_count", ax=axis, color="#fdae6b")
    axis.set_title("Где есть пропуски")
    axis.set_xlabel("Дескриптор")
    axis.set_ylabel("Число пропусков")
    axis.tick_params(axis="x", rotation=90)
    figure.tight_layout()
    figure.savefig(eda_dir / "missing_values.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_class_balance_plot(eda_dir: Path, class_balance: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(12, 4))
    sns.barplot(data=class_balance, x="task_name", y="count", hue="label", ax=axis)
    axis.set_title("Баланс классов по задачам")
    axis.set_xlabel("Задача")
    axis.set_ylabel("Число объектов")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(eda_dir / "class_balance.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_target_correlation_plot(eda_dir: Path, correlation_frame: pd.DataFrame) -> None:
    top_rows = (
        correlation_frame.sort_values(["target_column", "abs_correlation"], ascending=[True, False])
        .groupby("target_column", as_index=False)
        .head(10)
    )
    figure, axes = plt.subplots(3, 1, figsize=(10, 12))
    for axis, target_column in zip(axes, TARGET_COLUMNS.values()):
        subset = top_rows[top_rows["target_column"] == target_column]
        sns.barplot(data=subset, x="abs_correlation", y="feature", ax=axis, color="#74a9cf")
        axis.set_title(f"Самые сильные линейные связи для {target_column}")
        axis.set_xlabel("|corr|")
        axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(eda_dir / "top_target_correlations.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_eda_report(
    data_contract: dict,
    target_summary: pd.DataFrame,
    missing_summary: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    low_variance_summary: pd.DataFrame,
    high_correlation_pairs: pd.DataFrame,
    class_balance: pd.DataFrame,
    target_correlations: pd.DataFrame,
) -> str:
    skew_map = target_summary.set_index("target")["skew"]
    outlier_map = outlier_summary.set_index("target")["outlier_count"]
    quasi_constant_count = int((low_variance_summary["unique_values"] <= 2).sum())
    strong_pairs_count = int(len(high_correlation_pairs))
    top_missing = ", ".join(missing_summary["feature"].head(5).tolist()) if not missing_summary.empty else "нет"

    top_corr_lines = []
    for target_column in TARGET_COLUMNS.values():
        subset = (
            target_correlations[target_correlations["target_column"] == target_column]
            .sort_values("abs_correlation", ascending=False)
            .head(5)
        )
        summary = ", ".join(f"{row.feature} ({row.correlation:.3f})" for row in subset.itertuples())
        top_corr_lines.append(f"- {target_column}: {summary}")

    balance_lines = []
    for task_name in CLASSIFICATION_TASK_NAMES:
        subset = class_balance[class_balance["task_name"] == task_name].sort_values("label")
        negative, positive = subset["count"].tolist()
        balance_lines.append(f"- {task_name}: 0 -> {negative}, 1 -> {positive}")

    lines = [
        "# Сводка EDA",
        "",
        "## Что видно сразу",
        f"- В таблице `{data_contract['rows']}` строк и `{data_contract['columns']}` колонок. После отсечения таргетов и явных утечек остаётся `{data_contract['feature_count']}` дескрипторов.",
        f"- Пропусков мало: всего `{data_contract['missing_values']}` значений. Они сидят в `{len(missing_summary)}` колонках, в основном это `{top_missing}`.",
        f"- Все три регрессионные цели с тяжёлым правым хвостом: skew по IC50=`{skew_map[TARGET_COLUMNS['IC50']]:.3f}`, CC50=`{skew_map[TARGET_COLUMNS['CC50']]:.3f}`, SI=`{skew_map[TARGET_COLUMNS['SI']]:.3f}`. Для моделей это хороший аргумент в пользу `log1p`.",
        f"- По IQR-критерию выбросов у IC50=`{int(outlier_map[TARGET_COLUMNS['IC50']])}`, у CC50=`{int(outlier_map[TARGET_COLUMNS['CC50']])}`, у SI=`{int(outlier_map[TARGET_COLUMNS['SI']])}`. Самый тяжёлый хвост у `SI`, и это ожидаемо: показатель строится как отношение.",
        f"- Есть заметный хвост слабополезных и дублирующих друг друга дескрипторов: `{quasi_constant_count}` признаков имеют не более двух уникальных значений, а пар с |corr| >= 0.95 нашлось `{strong_pairs_count}`.",
        "",
        "## Баланс классов",
        *balance_lines,
        "",
        "## Самые заметные линейные связи",
        *top_corr_lines,
        "",
        "## Практический вывод",
        "- Для `IC50` и `CC50` есть несколько дескрипторов с умеренной линейной связью, поэтому имеет смысл сравнивать и линейные, и нелинейные модели.",
        "- Для `SI` линейные корреляции заметно слабее, поэтому здесь особенно важно не ограничиваться одной моделью и отдельно смотреть прямой и косвенный подходы.",
        "- Предварительно удалять пропуски или вручную выбрасывать хвосты не требуется: пропусков очень мало, а выбросы сами по себе несут химический смысл и лучше обрабатываются устойчивыми моделями и лог-преобразованием таргета.",
        "- Если сокращать пространство признаков, начинать стоит с почти постоянных дескрипторов и плотных коррелированных групп, а не с ручной чистки отдельных наблюдений.",
    ]
    return "\n".join(lines)


def run_eda(top_k_correlations: int = 20) -> dict:
    dataset_path = find_dataset_path()
    dataframe = load_dataset(dataset_path)
    feature_columns = get_feature_columns(dataframe)
    feature_frame = get_feature_frame(dataframe)

    eda_dir = ensure_dir(RESULTS_DIR / "eda")
    data_contract = build_data_contract(dataframe, dataset_path)
    write_json(RESULTS_DIR / "data_contract.json", data_contract)

    target_summary = build_target_summary(dataframe)
    missing_summary = build_missing_summary(dataframe)
    class_balance = build_class_balance(dataframe)
    outlier_summary = build_outlier_summary(dataframe)
    low_variance_summary = build_low_variance_summary(feature_frame)
    high_correlation_pairs = build_high_correlation_pairs(feature_frame)
    target_correlations = build_top_target_correlations(dataframe, feature_columns, top_k_correlations)

    write_dataframe(eda_dir / "target_summary.csv", target_summary)
    write_dataframe(eda_dir / "missing_values.csv", missing_summary)
    write_dataframe(eda_dir / "class_balance.csv", class_balance)
    write_dataframe(eda_dir / "outlier_summary.csv", outlier_summary)
    write_dataframe(eda_dir / "low_variance_features.csv", low_variance_summary.head(30))
    write_dataframe(eda_dir / "high_correlation_pairs.csv", high_correlation_pairs)
    write_dataframe(eda_dir / "top_descriptor_target_correlations.csv", target_correlations)

    save_target_plots(eda_dir, dataframe)
    save_missing_plot(eda_dir, missing_summary)
    save_class_balance_plot(eda_dir, class_balance)
    save_target_correlation_plot(eda_dir, target_correlations)

    report_text = build_eda_report(
        data_contract=data_contract,
        target_summary=target_summary,
        missing_summary=missing_summary,
        outlier_summary=outlier_summary,
        low_variance_summary=low_variance_summary,
        high_correlation_pairs=high_correlation_pairs,
        class_balance=class_balance,
        target_correlations=target_correlations,
    )
    write_text(REPORTS_DIR / "eda_summary.md", report_text)

    return {
        "dataset_path": dataset_path,
        "dataframe": dataframe,
        "feature_columns": feature_columns,
        "feature_frame": feature_frame,
        "data_contract": data_contract,
        "target_summary": target_summary,
        "missing_summary": missing_summary,
        "class_balance": class_balance,
        "outlier_summary": outlier_summary,
        "low_variance_summary": low_variance_summary,
        "high_correlation_pairs": high_correlation_pairs,
        "target_correlations": target_correlations,
        "report_text": report_text,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_eda(top_k_correlations=args.top_k_correlations)


if __name__ == "__main__":
    main()
