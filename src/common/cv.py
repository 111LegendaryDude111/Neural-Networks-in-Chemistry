from __future__ import annotations

from sklearn.model_selection import KFold, StratifiedKFold


def make_splitter(problem_type: str, n_splits: int, seed: int):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def describe_evaluation_strategy(strategy: str, outer_folds: int, inner_folds: int, test_size: float) -> str:
    if strategy == "nested":
        return f"nested CV ({outer_folds} outer folds x {inner_folds} inner folds)"
    return f"holdout ({test_size:.0%} test) + inner CV ({inner_folds} folds)"

