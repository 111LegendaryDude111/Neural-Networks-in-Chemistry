from __future__ import annotations

from src.common.config import REGRESSION_TASKS
from src.common.training import build_task_parser, run_regression_si_task


def main() -> None:
    task_config = REGRESSION_TASKS["regression_si"]
    parser = build_task_parser(task_config)
    args = parser.parse_args()
    run_regression_si_task(task_config, args)


if __name__ == "__main__":
    main()

