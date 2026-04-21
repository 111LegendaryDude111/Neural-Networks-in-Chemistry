from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import REPORTS_DIR, RESULTS_DIR, TASK_ORDER, TASKS
from src.common.io import ensure_dir, write_dataframe, write_text
from src.common.metrics import leaderboard_ascending

REPORT_COLUMN_TRANSLATIONS = {
    "task_name": "Код задачи",
    "problem_type": "Тип задачи",
    "target_column": "Целевая переменная",
    "primary_metric": "Основная метрика",
    "evaluation_strategy": "Стратегия оценки",
    "model_name": "Модель",
    "mode": "Режим",
    "fit_seconds": "Время обучения, с",
    "best_params_json": "Лучшие параметры (JSON)",
    "mae": "MAE",
    "mae_std": "MAE std",
    "rmse": "RMSE",
    "rmse_std": "RMSE std",
    "r2": "R2",
    "r2_std": "R2 std",
    "title": "Название",
    "primary_metric_name": "Код основной метрики",
    "roc_auc": "ROC-AUC",
    "roc_auc_std": "ROC-AUC std",
    "f1": "F1",
    "f1_std": "F1 std",
    "balanced_accuracy": "Balanced Accuracy",
    "balanced_accuracy_std": "Balanced Accuracy std",
    "pr_auc": "PR-AUC",
    "pr_auc_std": "PR-AUC std",
}

REPORT_VALUE_TRANSLATIONS = {
    "problem_type": {"regression": "регрессия", "classification": "классификация"},
    "evaluation_strategy": {"holdout": "отложенная выборка", "nested": "вложенная CV"},
    "mode": {"direct": "прямой", "indirect": "косвенный"},
}


def localize_report_value(column: str, value):
    return REPORT_VALUE_TRANSLATIONS.get(column, {}).get(value, value)


def localize_mode_case(mode: str) -> str:
    return {"direct": "прямом", "indirect": "косвенном"}.get(mode, mode)


def leaderboard_to_markdown(frame: pd.DataFrame) -> str:
    localized = frame.copy()
    for column, mapping in REPORT_VALUE_TRANSLATIONS.items():
        if column in localized.columns:
            localized[column] = localized[column].map(lambda value: mapping.get(value, value))
    return localized.rename(columns=REPORT_COLUMN_TRANSLATIONS).to_markdown(index=False)


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Собрать общую сводку по уже рассчитанным задачам.")


def load_task_leaderboard(task_name: str) -> pd.DataFrame | None:
    leaderboard_path = RESULTS_DIR / task_name / "leaderboard.csv"
    if not leaderboard_path.exists():
        return None
    frame = pd.read_csv(leaderboard_path)
    frame["task_name"] = task_name
    return frame


def main() -> None:
    build_parser().parse_args()
    comparison_dir = ensure_dir(RESULTS_DIR / "comparison")
    all_frames = []
    missing_tasks = []

    for task_name in TASK_ORDER:
        task_config = TASKS[task_name]
        leaderboard = load_task_leaderboard(task_name)
        if leaderboard is None:
            missing_tasks.append(task_name)
            continue
        leaderboard["title"] = task_config.title
        leaderboard["primary_metric_name"] = task_config.primary_metric
        all_frames.append(leaderboard)

    if not all_frames:
        raise FileNotFoundError("В results/ не найдено таблиц лидеров. Сначала запустите задачи.")

    all_results = pd.concat(all_frames, ignore_index=True)
    write_dataframe(comparison_dir / "all_model_results.csv", all_results)

    winner_rows = []
    baseline_rows = []
    report_lines = [
        "# Итоговая сводка",
        "",
        "## Как читать результаты",
        "- В регрессии основной ориентир `MAE`, дополнительно сохранены `RMSE` и `R2`.",
        "- В классификации сравнение идёт по `ROC-AUC`, `F1` и `Balanced Accuracy`, а для `SI > 8` главным остаётся `PR-AUC`.",
        "- Для `SI` сохранены оба сценария: прямая модель по дескрипторам и косвенная через отдельные прогнозы `CC50 / IC50`.",
        "",
        "## Лучшие модели по задачам",
    ]

    for task_name in TASK_ORDER:
        task_config = TASKS[task_name]
        task_rows = all_results[all_results["task_name"] == task_name].copy()
        if task_rows.empty:
            continue

        ascending = leaderboard_ascending(task_config.primary_metric)
        task_rows = task_rows.sort_values(task_config.primary_metric, ascending=ascending).reset_index(drop=True)
        winner = task_rows.iloc[0]
        winner_rows.append(winner)

        baseline = task_rows[task_rows["model_name"] == "dummy"]
        if not baseline.empty:
            baseline_rows.append(baseline.iloc[0])

        report_lines.append(
            f"- `{task_name}`: лучше всего сработала модель `{winner['model_name']}` "
            f"в `{localize_mode_case(winner['mode'])}` режиме "
            f"с `{task_config.primary_metric}={winner[task_config.primary_metric]:.6f}`."
        )

    winners_frame = pd.DataFrame(winner_rows)
    baselines_frame = pd.DataFrame(baseline_rows)
    write_dataframe(comparison_dir / "winners.csv", winners_frame)
    if not baselines_frame.empty:
        write_dataframe(comparison_dir / "baselines.csv", baselines_frame)

    report_lines.extend(["", "## Таблица победителей", leaderboard_to_markdown(winners_frame), ""])
    if not baselines_frame.empty:
        report_lines.extend(["## Бейзлайны", leaderboard_to_markdown(baselines_frame), ""])

    report_lines.extend(["## Что брать как рабочую отправную точку"])
    for winner in winners_frame.itertuples():
        report_lines.append(
            f"- Для `{winner.task_name}` стоит начинать с `{localize_report_value('mode', winner.mode)}::{winner.model_name}`: "
            f"эта конфигурация сейчас даёт лучший `{winner.primary_metric_name}`."
        )

    report_lines.append("")
    report_lines.append("## Артефакты важности признаков")
    for winner in winners_frame.itertuples():
        task_dir = RESULTS_DIR / winner.task_name
        importance_paths = sorted(task_dir.glob("winner_feature_importance*.csv"))
        if importance_paths:
            for importance_path in importance_paths:
                report_lines.append(f"- `{importance_path}`")
        else:
            report_lines.append(f"- `{winner.task_name}`: для победившей модели отдельный файл важности не сохранён.")

    if missing_tasks:
        report_lines.extend(["", "## Что ещё не посчитано"])
        for task_name in missing_tasks:
            report_lines.append(f"- Нет `leaderboard.csv` для `{task_name}`.")

    write_text(REPORTS_DIR / "project_summary.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
