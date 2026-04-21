from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.common.config import FIXED_THRESHOLDS, PREFERRED_DATASET_PATH, REPO_ROOT, TARGET_COLUMNS
from src.common.preprocessing import build_leakage_report, get_feature_columns


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
    feature_columns = get_feature_columns(dataframe)
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
