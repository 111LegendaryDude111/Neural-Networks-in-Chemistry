from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from src.common.config import RANDOM_SEED


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_factory: callable
    param_grid: dict[str, list]
    requires_scaling: bool = False
    optional_dependency: str | None = None


def _catboost_available() -> bool:
    return find_spec("catboost") is not None


def regression_model_specs(include_catboost: bool = True) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="dummy",
            estimator_factory=lambda: DummyRegressor(strategy="median"),
            param_grid={"strategy": ["mean", "median"]},
        ),
        ModelSpec(
            name="ridge",
            estimator_factory=lambda: Ridge(random_state=RANDOM_SEED),
            param_grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
            requires_scaling=True,
        ),
        ModelSpec(
            name="lasso",
            estimator_factory=lambda: Lasso(max_iter=10_000, random_state=RANDOM_SEED),
            param_grid={"alpha": [0.001, 0.01, 0.1, 1.0]},
            requires_scaling=True,
        ),
        ModelSpec(
            name="knn",
            estimator_factory=lambda: KNeighborsRegressor(),
            param_grid={
                "n_neighbors": [3, 5, 11, 21],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            requires_scaling=True,
        ),
        ModelSpec(
            name="svr",
            estimator_factory=lambda: SVR(),
            param_grid={
                "C": [0.1, 1.0, 10.0],
                "epsilon": [0.01, 0.1, 0.5],
                "gamma": ["scale", "auto"],
            },
            requires_scaling=True,
        ),
        ModelSpec(
            name="random_forest",
            estimator_factory=lambda: RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            param_grid={
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", 0.5],
            },
        ),
        ModelSpec(
            name="extra_trees",
            estimator_factory=lambda: ExtraTreesRegressor(
                n_estimators=200,
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            param_grid={
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", 0.5],
            },
        ),
        ModelSpec(
            name="gradient_boosting",
            estimator_factory=lambda: GradientBoostingRegressor(random_state=RANDOM_SEED),
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3],
                "subsample": [0.8, 1.0],
            },
        ),
    ]

    if include_catboost and _catboost_available():
        from catboost import CatBoostRegressor

        specs.append(
            ModelSpec(
                name="catboost",
                estimator_factory=lambda: CatBoostRegressor(
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=RANDOM_SEED,
                    thread_count=1,
                ),
                param_grid={
                    "iterations": [200, 400],
                    "depth": [4, 6],
                    "learning_rate": [0.03, 0.1],
                    "l2_leaf_reg": [1, 3, 5],
                },
                optional_dependency="catboost",
            )
        )

    return specs


def classification_model_specs(include_catboost: bool = True) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="dummy",
            estimator_factory=lambda: DummyClassifier(strategy="prior"),
            param_grid={"strategy": ["prior", "stratified", "most_frequent"]},
        ),
        ModelSpec(
            name="logistic_regression",
            estimator_factory=lambda: LogisticRegression(
                solver="liblinear",
                max_iter=5_000,
                random_state=RANDOM_SEED,
            ),
            param_grid={
                "C": [0.01, 0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            },
            requires_scaling=True,
        ),
        ModelSpec(
            name="knn",
            estimator_factory=lambda: KNeighborsClassifier(),
            param_grid={
                "n_neighbors": [3, 5, 11, 21],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            requires_scaling=True,
        ),
        ModelSpec(
            name="svc",
            estimator_factory=lambda: SVC(probability=True, random_state=RANDOM_SEED),
            param_grid={
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", "auto"],
                "class_weight": [None, "balanced"],
            },
            requires_scaling=True,
        ),
        ModelSpec(
            name="random_forest",
            estimator_factory=lambda: RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            param_grid={
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 2, 5],
                "class_weight": [None, "balanced"],
                "max_features": ["sqrt", 0.5],
            },
        ),
        ModelSpec(
            name="extra_trees",
            estimator_factory=lambda: ExtraTreesClassifier(
                n_estimators=200,
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            param_grid={
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 2, 5],
                "class_weight": [None, "balanced"],
                "max_features": ["sqrt", 0.5],
            },
        ),
        ModelSpec(
            name="gradient_boosting",
            estimator_factory=lambda: GradientBoostingClassifier(random_state=RANDOM_SEED),
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3],
                "subsample": [0.8, 1.0],
            },
        ),
    ]

    if include_catboost and _catboost_available():
        from catboost import CatBoostClassifier

        specs.append(
            ModelSpec(
                name="catboost",
                estimator_factory=lambda: CatBoostClassifier(
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=RANDOM_SEED,
                    thread_count=1,
                ),
                param_grid={
                    "iterations": [200, 400],
                    "depth": [4, 6],
                    "learning_rate": [0.03, 0.1],
                    "l2_leaf_reg": [1, 3, 5],
                },
                optional_dependency="catboost",
            )
        )

    return specs


def resolve_model_specs(
    problem_type: str,
    selected_models: list[str] | None = None,
    include_catboost: bool = True,
) -> list[ModelSpec]:
    available_specs = (
        regression_model_specs(include_catboost=include_catboost)
        if problem_type == "regression"
        else classification_model_specs(include_catboost=include_catboost)
    )

    if not selected_models:
        return available_specs

    selected = set(selected_models)
    resolved = [spec for spec in available_specs if spec.name in selected]
    missing = sorted(selected - {spec.name for spec in available_specs})
    if missing:
        raise ValueError(f"Unknown or unavailable models requested: {missing}")
    return resolved
