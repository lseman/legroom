"""Run Legroom's versioned evaluation suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legroom.evaluation import EvaluationSuite, format_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--suite", type=Path, default=PROJECT_ROOT / "benchmarks" / "suite-v1.json")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    report = EvaluationSuite.load(args.suite).run(model=args.model, repeat=args.repeat)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True) if args.as_json else format_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
