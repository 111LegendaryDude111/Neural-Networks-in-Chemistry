from __future__ import annotations

import re
from typing import Iterable

from src.common.config import LEAKAGE_TOKEN_PATTERNS, MANDATORY_EXCLUDED_COLUMNS

TOKEN_PATTERN = re.compile(
    rf"(^|_)({'|'.join(re.escape(token) for token in LEAKAGE_TOKEN_PATTERNS)})(_|$)"
)


def normalize_column_name(column_name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(column_name).strip().lower())
    return normalized.strip("_")


def is_forbidden_feature(column_name: str) -> bool:
    if column_name in MANDATORY_EXCLUDED_COLUMNS:
        return True
    return bool(TOKEN_PATTERN.search(normalize_column_name(column_name)))


def leakage_checklist() -> list[str]:
    return [
        "Drop mandatory columns: Unnamed: 0, IC50, CC50, SI and actual target columns with units.",
        "Reject any feature whose normalized name contains standalone tokens ic50, cc50, si, or selectivity_index.",
        "Validate that feature matrix contains only numeric descriptor columns.",
        "Validate before every task and persist a leakage_check.json artifact.",
    ]


def get_descriptor_columns(dataframe) -> list[str]:
    feature_columns = [column for column in dataframe.columns if not is_forbidden_feature(column)]
    validate_feature_columns(dataframe, feature_columns)
    return feature_columns


def validate_feature_columns(dataframe, feature_columns: Iterable[str]) -> None:
    feature_columns = list(feature_columns)
    forbidden_columns = [column for column in feature_columns if is_forbidden_feature(column)]
    if forbidden_columns:
        raise ValueError(f"Leakage detected in feature columns: {forbidden_columns}")

    non_numeric = (
        dataframe[feature_columns]
        .select_dtypes(exclude=["number", "bool"])
        .columns.tolist()
    )
    if non_numeric:
        raise ValueError(f"Non-numeric columns found in feature matrix: {non_numeric}")


def build_leakage_report(dataframe, feature_columns: Iterable[str]) -> dict:
    feature_columns = list(feature_columns)
    forbidden_in_dataframe = [
        column for column in dataframe.columns if is_forbidden_feature(column)
    ]
    forbidden_in_features = [column for column in feature_columns if is_forbidden_feature(column)]
    non_numeric_features = (
        dataframe[feature_columns]
        .select_dtypes(exclude=["number", "bool"])
        .columns.tolist()
    )

    status = "passed" if not forbidden_in_features and not non_numeric_features else "failed"
    report = {
        "status": status,
        "total_columns": len(dataframe.columns),
        "feature_count": len(feature_columns),
        "forbidden_columns_present_in_dataset": forbidden_in_dataframe,
        "forbidden_columns_detected_in_features": forbidden_in_features,
        "non_numeric_features": non_numeric_features,
        "checklist": leakage_checklist(),
    }

    if status != "passed":
        raise ValueError(f"Leakage validation failed: {report}")
    return report

