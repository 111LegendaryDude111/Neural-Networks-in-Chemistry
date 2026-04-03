from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common.config import REPORTS_DIR, RESULTS_DIR


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs() -> None:
    ensure_dir(RESULTS_DIR)
    ensure_dir(REPORTS_DIR)


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(to_serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_dataframe(path: Path, dataframe, index: bool = False) -> None:
    ensure_dir(path.parent)
    dataframe.to_csv(path, index=index)

