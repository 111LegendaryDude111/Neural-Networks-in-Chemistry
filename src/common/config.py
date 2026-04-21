from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ProblemType = Literal["regression", "classification"]


@dataclass(frozen=True)
class TaskConfig:
    name: str
    title: str
    problem_type: ProblemType
    target_column: str
    primary_metric: str
    threshold: float | None = None
    use_log_target: bool = False


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"

PREFERRED_DATASET_NAME = "data.xlsx"
PREFERRED_DATASET_PATH = DATA_DIR / PREFERRED_DATASET_NAME

TARGET_COLUMNS = {
    "IC50": "IC50, mM",
    "CC50": "CC50, mM",
    "SI": "SI",
}

MANDATORY_EXCLUDED_COLUMNS = (
    "Unnamed: 0",
    "IC50",
    "CC50",
    "SI",
    "IC50, mM",
    "CC50, mM",
)

LEAKAGE_TOKEN_PATTERNS = (
    "ic50",
    "cc50",
    "si",
    "selectivity_index",
)

FIXED_THRESHOLDS = {
    "classification_ic50_gt_median": 46.585183,
    "classification_cc50_gt_median": 411.039342,
    "classification_si_gt_median": 3.846154,
    "classification_si_gt_8": 8.0,
}

RANDOM_SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
HOLDOUT_TEST_SIZE = 0.2
TOP_K_IMPORTANCE = 20
EPSILON = 1e-8

TASK_ORDER = [
    "regression_ic50",
    "regression_cc50",
    "regression_si",
    "classification_ic50_gt_median",
    "classification_cc50_gt_median",
    "classification_si_gt_median",
    "classification_si_gt_8",
]

TASKS = {
    "regression_ic50": TaskConfig(
        name="regression_ic50",
        title="Регрессия: IC50",
        problem_type="regression",
        target_column=TARGET_COLUMNS["IC50"],
        primary_metric="mae",
        use_log_target=True,
    ),
    "regression_cc50": TaskConfig(
        name="regression_cc50",
        title="Регрессия: CC50",
        problem_type="regression",
        target_column=TARGET_COLUMNS["CC50"],
        primary_metric="mae",
        use_log_target=True,
    ),
    "regression_si": TaskConfig(
        name="regression_si",
        title="Регрессия: SI",
        problem_type="regression",
        target_column=TARGET_COLUMNS["SI"],
        primary_metric="mae",
        use_log_target=True,
    ),
    "classification_ic50_gt_median": TaskConfig(
        name="classification_ic50_gt_median",
        title="Классификация: IC50 > 46.585183",
        problem_type="classification",
        target_column=TARGET_COLUMNS["IC50"],
        threshold=FIXED_THRESHOLDS["classification_ic50_gt_median"],
        primary_metric="roc_auc",
    ),
    "classification_cc50_gt_median": TaskConfig(
        name="classification_cc50_gt_median",
        title="Классификация: CC50 > 411.039342",
        problem_type="classification",
        target_column=TARGET_COLUMNS["CC50"],
        threshold=FIXED_THRESHOLDS["classification_cc50_gt_median"],
        primary_metric="roc_auc",
    ),
    "classification_si_gt_median": TaskConfig(
        name="classification_si_gt_median",
        title="Классификация: SI > 3.846154",
        problem_type="classification",
        target_column=TARGET_COLUMNS["SI"],
        threshold=FIXED_THRESHOLDS["classification_si_gt_median"],
        primary_metric="roc_auc",
    ),
    "classification_si_gt_8": TaskConfig(
        name="classification_si_gt_8",
        title="Классификация: SI > 8",
        problem_type="classification",
        target_column=TARGET_COLUMNS["SI"],
        threshold=FIXED_THRESHOLDS["classification_si_gt_8"],
        primary_metric="pr_auc",
    ),
}

REGRESSION_TASK_NAMES = [
    task_name
    for task_name in TASK_ORDER
    if TASKS[task_name].problem_type == "regression"
]
CLASSIFICATION_TASK_NAMES = [
    task_name
    for task_name in TASK_ORDER
    if TASKS[task_name].problem_type == "classification"
]
