from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.common.config import (
    FIXED_THRESHOLDS,
    PREFERRED_DATASET_PATH,
    REPO_ROOT,
    TARGET_COLUMNS,
    TaskConfig,
)
from src.common.leakage import build_leakage_report, get_descriptor_columns


def find_dataset_path() -> Path:
    if PREFERRED_DATASET_PATH.exists():
        return PREFERRED_DATASET_PATH

    xlsx_files = sorted(PREFERRED_DATASET_PATH.parent.glob("*.xlsx"))
    if len(xlsx_files) == 1:
        return xlsx_files[0]

    raise FileNotFoundError(
        "Dataset was not found at the preferred path and data/ does not contain a single XLSX fallback."
    )


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    dataset_path = dataset_path or find_dataset_path()
    return pd.read_excel(dataset_path)


def build_data_contract(dataframe: pd.DataFrame, dataset_path: Path | None = None) -> dict:
    dataset_path = dataset_path or find_dataset_path()
    feature_columns = get_descriptor_columns(dataframe)
    return {
        "dataset_path": str(dataset_path.relative_to(REPO_ROOT)),
        "checksum_sha256": compute_sha256(dataset_path),
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "feature_count": len(feature_columns),
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicates": int(dataframe.duplicated().sum()),
        "target_columns": list(TARGET_COLUMNS.values()),
        "thresholds": FIXED_THRESHOLDS,
        "leakage_rules": build_leakage_report(dataframe, feature_columns)["checklist"],
    }


def prepare_task_data(task_config: TaskConfig):
    dataset_path = find_dataset_path()
    dataframe = load_dataset(dataset_path)
    feature_columns = get_descriptor_columns(dataframe)
    leakage_report = build_leakage_report(dataframe, feature_columns)
    x_frame = dataframe[feature_columns].copy()

    if task_config.problem_type == "regression":
        target = dataframe[task_config.target_column].astype(float).copy()
    else:
        if task_config.threshold is None:
            raise ValueError(f"Threshold is required for classification task {task_config.name}")
        target = dataframe[task_config.target_column].gt(task_config.threshold).astype(int)

    return {
        "dataset_path": dataset_path,
        "dataframe": dataframe,
        "X": x_frame,
        "y": target,
        "feature_columns": feature_columns,
        "data_contract": build_data_contract(dataframe, dataset_path),
        "leakage_report": leakage_report,
    }

