#!/usr/bin/env python
"""Benchmark harness — measures legroom's compression on realistic message traces.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --fixtures benchmarks/fixtures --model gpt-4o
    python benchmarks/run_benchmark.py --json > results.json

Fixtures are JSON files of the form {"description": str, "messages": [...]}.
Each fixture is compressed with legroom's default pipeline; token counts,
per-transform application, and latency are reported per-fixture and in
aggregate. If the `headroom` package is importable, its compression is run
on the same fixtures for a side-by-side comparison — otherwise that column
is omitted rather than faked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legroom import compress, CompressConfig  # noqa: E402


@dataclass
class FixtureResult:
    name: str
    description: str
    tokens_before: int
    tokens_after: int
    latency_ms: float
    transforms_applied: list[str]
    headroom_tokens_after: int | None = None
    headroom_latency_ms: float | None = None

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def ratio(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return self.tokens_saved / self.tokens_before


def _load_fixtures(fixtures_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    fixtures = []
    for path in sorted(fixtures_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        fixtures.append((path.stem, data))
    return fixtures


def _try_headroom_compress(messages: list[dict[str, Any]], model: str) -> tuple[int, float] | None:
    """Run headroom's compressor on the same messages, if installed.

    Returns (tokens_after, latency_ms) or None if headroom isn't available
    or its API doesn't match what we expect. We do not guess at an API
    that may not exist — absence just means the comparison is skipped.
    """
    try:
        import headroom  # type: ignore
    except ImportError:
        return None

    for attr in ("compress", "optimize"):
        fn = getattr(headroom, attr, None)
        if callable(fn):
            start = time.perf_counter()
            try:
                result = fn(messages, model=model)
            except TypeError:
                result = fn(messages)
            latency_ms = (time.perf_counter() - start) * 1000

            for tok_attr in ("tokens_after", "compressed_token_count", "token_count"):
                tokens_after = getattr(result, tok_attr, None)
                if tokens_after is not None:
                    return int(tokens_after), latency_ms
    return None


def run_benchmark(fixtures_dir: Path, model: str, config: CompressConfig) -> list[FixtureResult]:
    results = []
    for name, fixture in _load_fixtures(fixtures_dir):
        messages = fixture["messages"]
        description = fixture.get("description", "")

        start = time.perf_counter()
        result = compress([dict(m) for m in messages], model=model, config=config)
        latency_ms = (time.perf_counter() - start) * 1000

        headroom_result = _try_headroom_compress([dict(m) for m in messages], model)

        results.append(
            FixtureResult(
                name=name,
                description=description,
                tokens_before=result.tokens_before,
                tokens_after=result.tokens_after,
                latency_ms=latency_ms,
                transforms_applied=result.transforms_applied,
                headroom_tokens_after=headroom_result[0] if headroom_result else None,
                headroom_latency_ms=headroom_result[1] if headroom_result else None,
            )
        )
    return results


def _fmt_pct(ratio: float) -> str:
    return f"{ratio * 100:.1f}%"


def print_report(results: list[FixtureResult]) -> None:
    headroom_available = any(r.headroom_tokens_after is not None for r in results)

    name_w = max(len(r.name) for r in results) + 2
    header = f"{'fixture':<{name_w}}{'before':>8}{'after':>8}{'saved':>8}{'ratio':>8}{'ms':>8}"
    if headroom_available:
        header += f"{'hr_after':>10}{'hr_ratio':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        row = (
            f"{r.name:<{name_w}}{r.tokens_before:>8}{r.tokens_after:>8}"
            f"{r.tokens_saved:>8}{_fmt_pct(r.ratio):>8}{r.latency_ms:>8.2f}"
        )
        if headroom_available:
            if r.headroom_tokens_after is not None:
                hr_ratio = (
                    (r.tokens_before - r.headroom_tokens_after) / r.tokens_before
                    if r.tokens_before
                    else 0.0
                )
                row += f"{r.headroom_tokens_after:>10}{_fmt_pct(hr_ratio):>10}"
            else:
                row += f"{'—':>10}{'—':>10}"
        print(row)

    total_before = sum(r.tokens_before for r in results)
    total_after = sum(r.tokens_after for r in results)
    total_saved = total_before - total_after
    total_ratio = total_saved / total_before if total_before else 0.0
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0.0

    print("-" * len(header))
    total_row = (
        f"{'TOTAL':<{name_w}}{total_before:>8}{total_after:>8}"
        f"{total_saved:>8}{_fmt_pct(total_ratio):>8}{avg_latency:>8.2f}"
    )
    print(total_row)

    if not headroom_available:
        print(
            "\n(headroom not installed — comparison columns omitted; "
            "install it and re-run to see a side-by-side)"
        )

    print("\nTransforms applied per fixture:")
    for r in results:
        print(f"  {r.name}: {', '.join(r.transforms_applied) or '(none)'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path(__file__).parent / "fixtures",
        help="Directory of fixture JSON files (default: benchmarks/fixtures)",
    )
    parser.add_argument("--model", default="gpt-4o", help="Model name for token counting")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table")
    args = parser.parse_args()

    if not args.fixtures.is_dir():
        parser.error(f"fixtures directory not found: {args.fixtures}")

    config = CompressConfig()
    results = run_benchmark(args.fixtures, args.model, config)

    if not results:
        parser.error(f"no *.json fixtures found in {args.fixtures}")

    if args.json:
        payload = [
            {
                "name": r.name,
                "description": r.description,
                "tokens_before": r.tokens_before,
                "tokens_after": r.tokens_after,
                "tokens_saved": r.tokens_saved,
                "ratio": r.ratio,
                "latency_ms": r.latency_ms,
                "transforms_applied": r.transforms_applied,
                "headroom_tokens_after": r.headroom_tokens_after,
                "headroom_latency_ms": r.headroom_latency_ms,
            }
            for r in results
        ]
        print(json.dumps(payload, indent=2))
    else:
        print_report(results)


if __name__ == "__main__":
    main()
