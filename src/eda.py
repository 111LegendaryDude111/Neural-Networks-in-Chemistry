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
import pandas as pd
import seaborn as sns

from src.common.config import FIXED_THRESHOLDS, REPORTS_DIR, RESULTS_DIR, TARGET_COLUMNS
from src.common.data import build_data_contract, find_dataset_path, load_dataset
from src.common.io import ensure_dir, write_dataframe, write_json, write_text
from src.common.leakage import get_descriptor_columns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDA for the chemical activity dataset.")
    parser.add_argument("--top-k-correlations", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = find_dataset_path()
    dataframe = load_dataset(dataset_path)
    feature_columns = get_descriptor_columns(dataframe)

    eda_dir = ensure_dir(RESULTS_DIR / "eda")
    data_contract = build_data_contract(dataframe, dataset_path)
    write_json(RESULTS_DIR / "data_contract.json", data_contract)

    target_columns = list(TARGET_COLUMNS.values())
    target_summary = dataframe[target_columns].describe().T
    target_summary["skew"] = dataframe[target_columns].skew()
    write_dataframe(eda_dir / "target_summary.csv", target_summary.reset_index().rename(columns={"index": "target"}))

    missing_summary = dataframe.isna().sum().sort_values(ascending=False)
    missing_summary = missing_summary[missing_summary > 0]
    missing_frame = missing_summary.reset_index()
    missing_frame.columns = ["feature", "missing_count"]
    write_dataframe(eda_dir / "missing_values.csv", missing_frame)

    class_balance_rows = []
    for task_name, threshold in FIXED_THRESHOLDS.items():
        if task_name.startswith("ic50"):
            column = TARGET_COLUMNS["IC50"]
        elif task_name.startswith("cc50"):
            column = TARGET_COLUMNS["CC50"]
        else:
            column = TARGET_COLUMNS["SI"]
        value_counts = dataframe[column].gt(threshold).value_counts().sort_index()
        for label, count in value_counts.items():
            class_balance_rows.append(
                {
                    "task_name": task_name,
                    "target_column": column,
                    "threshold": threshold,
                    "label": int(label),
                    "count": int(count),
                }
            )
    class_balance_frame = pd.DataFrame(class_balance_rows)
    write_dataframe(eda_dir / "class_balance.csv", class_balance_frame)

    correlation_rows = []
    for target_column in target_columns:
        correlations = dataframe[feature_columns].corrwith(dataframe[target_column]).dropna()
        top_correlations = (
            correlations.abs()
            .sort_values(ascending=False)
            .head(args.top_k_correlations)
            .index.tolist()
        )
        for feature_name in top_correlations:
            correlation_rows.append(
                {
                    "target_column": target_column,
                    "feature": feature_name,
                    "correlation": float(correlations.loc[feature_name]),
                    "abs_correlation": float(abs(correlations.loc[feature_name])),
                }
            )
    top_correlations_frame = pd.DataFrame(correlation_rows)
    write_dataframe(eda_dir / "top_descriptor_target_correlations.csv", top_correlations_frame)

    figure, axes = plt.subplots(3, 1, figsize=(10, 12))
    for axis, target_column in zip(axes, target_columns):
        sns.histplot(dataframe[target_column], kde=True, ax=axis, bins=30, color="#2c7fb8")
        axis.set_title(f"Distribution: {target_column}")
        axis.set_xlabel(target_column)
    figure.tight_layout()
    figure.savefig(eda_dir / "target_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, target_column in zip(axes, target_columns):
        sns.boxplot(y=dataframe[target_column], ax=axis, color="#7fcdbb")
        axis.set_title(f"Boxplot: {target_column}")
    figure.tight_layout()
    figure.savefig(eda_dir / "target_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(dataframe[target_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axis)
    axis.set_title("Target Correlation Heatmap")
    figure.tight_layout()
    figure.savefig(eda_dir / "target_correlations.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    if not missing_frame.empty:
        figure, axis = plt.subplots(figsize=(10, 4))
        sns.barplot(data=missing_frame, x="feature", y="missing_count", ax=axis, color="#fdae6b")
        axis.set_title("Missing Values by Feature")
        axis.set_xlabel("Feature")
        axis.set_ylabel("Missing Count")
        axis.tick_params(axis="x", rotation=90)
        figure.tight_layout()
        figure.savefig(eda_dir / "missing_values.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4))
    sns.barplot(data=class_balance_frame, x="task_name", y="count", hue="label", ax=axis)
    axis.set_title("Classification Label Balance")
    axis.set_xlabel("Task")
    axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(eda_dir / "class_balance.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    top_corr_text = []
    for target_column in target_columns:
        subset = (
            top_correlations_frame[top_correlations_frame["target_column"] == target_column]
            .sort_values("abs_correlation", ascending=False)
            .head(5)
        )
        summary = ", ".join(
            f"{row.feature} ({row.correlation:.3f})" for row in subset.itertuples()
        )
        top_corr_text.append(f"- {target_column}: {summary}")

    report_lines = [
        "# EDA Summary",
        "",
        "## Data Contract",
        f"- Dataset path: `{data_contract['dataset_path']}`",
        f"- SHA256: `{data_contract['checksum_sha256']}`",
        f"- Rows: `{data_contract['rows']}`",
        f"- Columns: `{data_contract['columns']}`",
        f"- Descriptor features after leakage guard: `{data_contract['feature_count']}`",
        f"- Missing values: `{data_contract['missing_values']}`",
        f"- Duplicates: `{data_contract['duplicates']}`",
        "",
        "## Main Findings",
        f"- Все три regression targets положительные и сильно skewed: IC50=`{target_summary.loc[TARGET_COLUMNS['IC50'], 'skew']:.3f}`, CC50=`{target_summary.loc[TARGET_COLUMNS['CC50'], 'skew']:.3f}`, SI=`{target_summary.loc[TARGET_COLUMNS['SI'], 'skew']:.3f}`.",
        f"- Пропуски сосредоточены в `{len(missing_frame)}` дескрипторах и составляют `{int(missing_frame['missing_count'].sum())}` значений суммарно.",
        "- Median-based classification задачи почти идеально сбалансированы; `SI > 8` заметно более дисбалансна.",
        "- Корреляция `IC50` и `CC50` умеренная положительная, а `SI` с ними линейно связан слабо из-за ratio-природы и тяжёлого хвоста.",
        "",
        "## Top Descriptor Correlations",
        *top_corr_text,
        "",
        "## Saved Artifacts",
        f"- `{eda_dir / 'target_distributions.png'}`",
        f"- `{eda_dir / 'target_boxplots.png'}`",
        f"- `{eda_dir / 'target_correlations.png'}`",
        f"- `{eda_dir / 'missing_values.csv'}`",
        f"- `{eda_dir / 'top_descriptor_target_correlations.csv'}`",
    ]
    write_text(REPORTS_DIR / "eda_summary.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
