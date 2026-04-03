from __future__ import annotations

from src.common.config import CLASSIFICATION_TASKS
from src.common.training import build_task_parser, run_supervised_task


def main() -> None:
    task_config = CLASSIFICATION_TASKS["classification_si_gt_8"]
    parser = build_task_parser(task_config)
    args = parser.parse_args()
    run_supervised_task(task_config, args)


if __name__ == "__main__":
    main()

