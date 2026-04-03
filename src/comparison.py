from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import ALL_TASKS, REPORTS_DIR, RESULTS_DIR
from src.common.io import ensure_dir, write_dataframe, write_text
from src.common.metrics import leaderboard_ascending


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Aggregate all task results and build a project summary.")


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

    for task_name, task_config in ALL_TASKS.items():
        leaderboard = load_task_leaderboard(task_name)
        if leaderboard is None:
            missing_tasks.append(task_name)
            continue
        leaderboard["title"] = task_config.title
        leaderboard["primary_metric_name"] = task_config.primary_metric
        all_frames.append(leaderboard)

    if not all_frames:
        raise FileNotFoundError("No task leaderboards found in results/. Run task scripts first.")

    all_results = pd.concat(all_frames, ignore_index=True)
    write_dataframe(comparison_dir / "all_model_results.csv", all_results)

    winner_rows = []
    baseline_rows = []
    report_lines = [
        "# Project Summary",
        "",
        "## Methodology",
        "- Features are limited to descriptor columns only, enforced by centralized anti-leakage guards.",
        "- Regression uses MAE as the primary metric and applies transparent log1p target transformation.",
        "- Classification uses ROC-AUC, F1 and Balanced Accuracy; `SI > 8` also uses PR-AUC as the key metric.",
        "- Nested CV `5x3` is the preferred protocol; holdout + inner CV is the explicit compute fallback.",
        "- `SI` is evaluated in both direct and indirect formulations.",
        "",
        "## Winners by Task",
    ]

    for task_name, task_config in ALL_TASKS.items():
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
            f"- `{task_name}`: winner=`{winner['mode']}::{winner['model_name']}` "
            f"with `{task_config.primary_metric}={winner[task_config.primary_metric]:.6f}` "
            f"on `{winner['evaluation_strategy']}`."
        )

    winners_frame = pd.DataFrame(winner_rows)
    baselines_frame = pd.DataFrame(baseline_rows)
    write_dataframe(comparison_dir / "winners.csv", winners_frame)
    if not baselines_frame.empty:
        write_dataframe(comparison_dir / "baselines.csv", baselines_frame)

    report_lines.extend(["", "## Winner Table", winners_frame.to_markdown(index=False), ""])
    if not baselines_frame.empty:
        report_lines.extend(["## Baselines", baselines_frame.to_markdown(index=False), ""])

    report_lines.append("## Feature Importance Artifacts")
    for winner in winners_frame.itertuples():
        task_dir = RESULTS_DIR / winner.task_name
        importance_paths = sorted(task_dir.glob("winner_feature_importance*.csv"))
        if importance_paths:
            for importance_path in importance_paths:
                report_lines.append(f"- `{importance_path}`")
        else:
            report_lines.append(f"- `{winner.task_name}`: no feature-importance artifact for winner model.")

    report_lines.extend(["", "## Recommendations"])
    for winner in winners_frame.itertuples():
        report_lines.append(
            f"- For `{winner.task_name}`, start from `{winner.mode}::{winner.model_name}` "
            f"because it achieved the best `{winner.primary_metric_name}` in the current benchmark."
        )

    if missing_tasks:
        report_lines.extend(["", "## Missing Tasks"])
        for task_name in missing_tasks:
            report_lines.append(f"- `{task_name}` has no generated leaderboard yet.")

    write_text(REPORTS_DIR / "project_summary.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
