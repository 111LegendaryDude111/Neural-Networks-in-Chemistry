from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import EPSILON


def build_model_pipeline(estimator, requires_scaling: bool, use_log_target: bool = False):
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

