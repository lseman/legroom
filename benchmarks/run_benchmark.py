"""Run Legroom's versioned evaluation suite."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legroom.analysis.evaluation import (  # noqa: E402
    CompositeTaskEvaluator,
    DeclaredChecksEvaluator,
    EvaluationSuite,
    ExpectedTermsEvaluator,
    format_markdown,
)
from legroom.analysis.task_replay import SubprocessTaskRunner, TaskRunnerEvaluator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--suite", type=Path, default=PROJECT_ROOT / "benchmarks" / "suite-v1.json")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--strategies",
        help="Comma-separated strategy names; defaults to every strategy in the suite",
    )
    parser.add_argument(
        "--task-runner-command",
        help=(
            "Explicit JSON-over-stdio task runner command. The command is split "
            "without invoking a shell and produces task_verified evidence."
        ),
    )
    parser.add_argument("--task-runner-timeout", type=float, default=300.0)
    parser.add_argument("--task-runner-cwd", type=Path)
    args = parser.parse_args()
    strategy_names = (
        tuple(name.strip() for name in args.strategies.split(",") if name.strip())
        if args.strategies
        else None
    )
    evaluator = None
    if args.task_runner_command:
        command = shlex.split(args.task_runner_command)
        runner = SubprocessTaskRunner(
            command,
            timeout_seconds=args.task_runner_timeout,
            cwd=args.task_runner_cwd,
        )
        evaluator = CompositeTaskEvaluator(
            (
                ExpectedTermsEvaluator(),
                DeclaredChecksEvaluator(),
                TaskRunnerEvaluator(runner),
            )
        )
    report = EvaluationSuite.load(args.suite, evaluator=evaluator).run(
        model=args.model, repeat=args.repeat, strategy_names=strategy_names
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True) if args.as_json else format_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
