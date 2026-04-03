from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_scoring() -> dict[str, str]:
    return {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }


def classification_scoring(include_pr_auc: bool = False) -> dict[str, str]:
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
    }
    if include_pr_auc:
        scoring["pr_auc"] = "average_precision"
    return scoring


def metric_columns(problem_type: str, include_pr_auc: bool = False) -> list[str]:
    if problem_type == "regression":
        return ["mae", "rmse", "r2"]
    columns = ["roc_auc", "f1", "balanced_accuracy"]
    if include_pr_auc:
        columns.append("pr_auc")
    return columns


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_classification_metrics(
    y_true,
    y_pred,
    y_score,
    include_pr_auc: bool = False,
) -> dict[str, float]:
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if include_pr_auc:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    return metrics


def leaderboard_ascending(primary_metric: str) -> bool:
    return primary_metric in {"mae", "rmse"}


def summarize_metric_frame(metric_frame, metric_names: Iterable[str]) -> dict[str, float]:
    summary = {}
    for metric in metric_names:
        summary[metric] = float(metric_frame[metric].mean())
        summary[f"{metric}_std"] = float(metric_frame[metric].std(ddof=0)) if len(metric_frame) > 1 else 0.0
    return summary
