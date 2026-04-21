from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import (
    EPSILON,
    LEAKAGE_TOKEN_PATTERNS,
    MANDATORY_EXCLUDED_COLUMNS,
    TaskConfig,
)

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
        "В признаки не попадают колонки с таргетами и их прямые производные.",
        "Остаются только числовые дескрипторы.",
        "Пропуски не затираются заранее, а закрываются внутри sklearn-пайплайна медианной импутацией.",
        "Масштабирование включается только для моделей, которым оно реально нужно.",
    ]


def validate_feature_columns(
    dataframe: pd.DataFrame, feature_columns: Iterable[str]
) -> None:
    feature_columns = list(feature_columns)
    forbidden_columns = [
        column for column in feature_columns if is_forbidden_feature(column)
    ]
    if forbidden_columns:
        raise ValueError(f"Leakage detected in feature columns: {forbidden_columns}")

    non_numeric = (
        dataframe[feature_columns]
        .select_dtypes(exclude=["number", "bool"])
        .columns.tolist()
    )
    if non_numeric:
        raise ValueError(f"Non-numeric columns found in feature matrix: {non_numeric}")


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    feature_columns = [
        column for column in dataframe.columns if not is_forbidden_feature(column)
    ]
    validate_feature_columns(dataframe, feature_columns)
    return feature_columns


def get_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe[get_feature_columns(dataframe)].copy()


def build_leakage_report(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> dict:
    feature_columns = list(feature_columns or get_feature_columns(dataframe))
    validate_feature_columns(dataframe, feature_columns)

    forbidden_in_dataset = [
        column for column in dataframe.columns if is_forbidden_feature(column)
    ]
    non_numeric_features = (
        dataframe[feature_columns]
        .select_dtypes(exclude=["number", "bool"])
        .columns.tolist()
    )

    return {
        "status": "passed",
        "total_columns": len(dataframe.columns),
        "feature_count": len(feature_columns),
        "forbidden_columns_present_in_dataset": forbidden_in_dataset,
        "forbidden_columns_detected_in_features": [],
        "non_numeric_features": non_numeric_features,
        "checklist": leakage_checklist(),
    }


def build_target(dataframe: pd.DataFrame, task_config: TaskConfig) -> pd.Series:
    if task_config.problem_type == "regression":
        return dataframe[task_config.target_column].astype(float).copy()

    if task_config.threshold is None:
        raise ValueError(
            f"Threshold is required for classification task {task_config.name}"
        )

    return dataframe[task_config.target_column].gt(task_config.threshold).astype(int)


def prepare_task_data(dataframe: pd.DataFrame, task_config: TaskConfig) -> dict:
    feature_columns = get_feature_columns(dataframe)
    return {
        "X": dataframe[feature_columns].copy(),
        "y": build_target(dataframe, task_config),
        "feature_columns": feature_columns,
        "leakage_report": build_leakage_report(dataframe, feature_columns),
    }


def build_model_pipeline(
    estimator, requires_scaling: bool, use_log_target: bool = False
):
    pipeline_steps = [("imputer", SimpleImputer(strategy="median"))]
    if requires_scaling:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(("estimator", estimator))

    pipeline = Pipeline(pipeline_steps)
    if use_log_target:
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return pipeline


def prefix_param_grid(param_grid: dict[str, list], use_log_target: bool = False):
    if not param_grid:
        return [{}]

    prefix = "estimator__"
    if use_log_target:
        prefix = f"regressor__{prefix}"
    return {f"{prefix}{key}": values for key, values in param_grid.items()}


def extract_final_estimator(fitted_model):
    if isinstance(fitted_model, TransformedTargetRegressor):
        pipeline = fitted_model.regressor_
    else:
        pipeline = fitted_model
    return pipeline.named_steps["estimator"]


def clip_regression_predictions(predictions):
    return np.clip(np.asarray(predictions, dtype=float), EPSILON, None)
