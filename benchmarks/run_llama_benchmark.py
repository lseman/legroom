#!/usr/bin/env python3
"""Run Legroom's llama.cpp KV cache alignment benchmark suite.

This benchmark suite specifically tests Legroom's llama.cpp optimizations:
- JSON canonicalization correctness (round-trip equivalence)
- Sequential number normalization stability
- Stable prefix cache behavior
- Overall KV cache alignment quality

Usage::

    # Run full suite
    python benchmarks/run_llama_benchmark.py

    # Run with specific model
    python benchmarks/run_llama_benchmark.py --model llama-3.1-8b

    # Output JSON for CI
    python benchmarks/run_llama_benchmark.py --json

    # Run specific strategies
    python benchmarks/run_llama_benchmark.py --strategies llama_cpp_default,llama_cpp_no_canonicalization

    # Repeat multiple times
    python benchmarks/run_llama_benchmark.py --repeat 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legroom.analysis.evaluation import (  # noqa: E402
    Fixture,
    _load_declared_checks,
    count_tokens_messages,
)
from legroom.analysis.llama_benchmarks import (  # noqa: E402
    KvCacheAlignmentEvaluator,
    LlamaBenchmarkRunner,
    LlamaResult,
)
from legroom.runtime.compress import compress  # noqa: E402
from legroom.runtime.config import CompressConfig  # noqa: E402


def _load_fixture(path: Path) -> Fixture:
    """Load a single fixture from JSON."""
    document = json.loads(path.read_text(encoding="utf-8"))
    checks = _load_declared_checks(document)
    return Fixture(
        name=path.stem,
        description=document.get("description", ""),
        messages=document["messages"],
        expected_terms=tuple(document.get("expected_terms", [])),
        declared_checks=checks,
    )


def _load_suite(suite_path: Path) -> dict[str, Any]:
    """Load the suite manifest."""
    return json.loads(suite_path.read_text(encoding="utf-8"))


def _run_strategy(
    fixture: Fixture,
    strategy_config: dict[str, Any],
    model: str,
    repeat: int = 3,
) -> dict[str, Any]:
    """Run one strategy on one fixture."""
    config = CompressConfig(**{
        k: v for k, v in strategy_config.items()
        if k in CompressConfig.__dataclass_fields__
    })

    latencies: list[float] = []
    alignment_scores: list[float] = []
    alignment_passed_list: list[bool] = []
    details_list: list[dict[str, Any]] = []

    for _ in range(repeat):
        tracemalloc.start()
        started = time.perf_counter()

        compressed = compress(
            fixture.messages,
            model=model,
            config=config,
        )

        latencies.append((time.perf_counter() - started) * 1000)

        # Run llama.cpp evaluators
        evaluator = KvCacheAlignmentEvaluator()
        result: LlamaResult = evaluator.evaluate(
            fixture,
            compressed.messages,
            fixture.messages,
            backend=config.backend,
        )

        tracemalloc.stop()

        alignment_scores.append(result.score)
        alignment_passed_list.append(result.passed)
        details_list.append(result.details)

    return {
        "fixture": fixture.name,
        "strategy": strategy_config.get("name", "default"),
        "tokens_before": count_tokens_messages(fixture.messages, model),
        "tokens_after": compressed.tokens_after,
        "compression_ratio": 1.0 - compressed.tokens_after / count_tokens_messages(fixture.messages, model)
        if count_tokens_messages(fixture.messages, model) > 0 else 0.0,
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2],
        "alignment_score": sum(alignment_scores) / len(alignment_scores),
        "alignment_passed": all(alignment_passed_list),
        "details": details_list[-1],  # Last repeat's details
    }


def _format_markdown(results: dict[str, Any]) -> str:
    """Format benchmark results as Markdown table."""
    lines = [
        "# Legroom llama.cpp Benchmark Results",
        "",
        f"Model: {results['model']}",
        f"Suite: {results['suite']}",
        "",
        "## Strategy Comparison",
        "",
        "| Strategy | Fixture | Saved | Alignment | Passed | Latency p50 |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _strategy_name, strategy_results in results.get("strategy_results", {}).items():
        for result in strategy_results:
            lines.append(
                f"| {result['strategy']} | {result['fixture']} | "
                f"{result['compression_ratio']:.1%} | "
                f"{result['alignment_score']:.2f} | "
                f"{'✓' if result['alignment_passed'] else '✗'} | "
                f"{result['latency_ms_p50']:.2f} |"
            )

    # Summary section
    lines.extend([
        "",
        "## Alignment Sub-Scores",
        "",
        "| Strategy | Fixture | JSON Correctness | Sequential Stability | Prefix Stability |",
        "|---|---|---:|---:|---:|",
    ])

    for strategy_name, strategy_results in results.get("strategy_results", {}).items():
        for result in strategy_results:
            details = result.get("details", {})
            sub = details.get("sub_scores", {})
            lines.append(
                f"| {strategy_name} | {result['fixture']} | "
                f"{sub.get('json_correctness', 1.0):.2f} | "
                f"{sub.get('sequential_stability', 1.0):.2f} | "
                f"{sub.get('prefix_stability', 1.0):.2f} |"
            )

    # Overall summary
    lines.extend(["", "## Overall Summary", ""])
    for strategy_name, strategy_results in results.get("strategy_results", {}).items():
        avg_alignment = sum(r["alignment_score"] for r in strategy_results) / len(strategy_results)
        avg_compression = sum(r["compression_ratio"] for r in strategy_results) / len(strategy_results)
        passed = sum(1 for r in strategy_results if r["alignment_passed"])
        lines.append(
            f"- **{strategy_name}**: avg alignment={avg_alignment:.3f}, "
            f"avg saved={avg_compression:.1%}, "
            f"passed={passed}/{len(strategy_results)}"
        )

    return "\n".join(lines)


def _format_json(results: dict[str, Any]) -> str:
    """Format benchmark results as JSON."""
    return json.dumps(results, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o", help="Model name for token counting")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output JSON")
    parser.add_argument(
        "--suite",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "llama-suite-v1.json",
        help="Path to suite manifest",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of repetitions per strategy (default: 3)",
    )
    parser.add_argument(
        "--strategies",
        help="Comma-separated strategy names; defaults to every strategy in the suite",
    )
    parser.add_argument(
        "--fixtures",
        help="Comma-separated fixture names; defaults to all fixtures",
    )
    args = parser.parse_args()

    # Load suite
    suite = _load_suite(args.suite)

    # Load fixtures
    fixtures_dir = args.suite.parent / "fixtures"
    all_fixtures = [_load_fixture(p) for p in fixtures_dir.glob("llama_*.json")]

    fixture_names = (
        tuple(name.strip() for name in args.fixtures.split(",") if name.strip())
        if args.fixtures
        else None
    )
    if fixture_names:
        fixtures = [f for f in all_fixtures if f.name in fixture_names]
    else:
        fixtures = all_fixtures

    # Load strategies
    strategy_definitions = suite.get("strategies", [])
    strategy_names = (
        tuple(name.strip() for name in args.strategies.split(",") if name.strip())
        if args.strategies
        else None
    )

    if strategy_names:
        strategies = {
            sd["name"]: sd.get("config", {})
            for sd in strategy_definitions
            if sd["name"] in strategy_names
        }
    else:
        strategies = {
            sd["name"]: sd.get("config", {})
            for sd in strategy_definitions
        }

    # Run benchmarks
    LlamaBenchmarkRunner()
    strategy_results: dict[str, list[dict[str, Any]]] = {}

    for strategy_name, config in strategies.items():
        print(f"Running strategy: {strategy_name}...")
        results: list[dict[str, Any]] = []
        for fixture in fixtures:
            result = _run_strategy(
                fixture=fixture,
                strategy_config={**config, "name": strategy_name},
                model=args.model,
                repeat=args.repeat,
            )
            results.append(result)
            print(f"  {fixture.name}: alignment={result['alignment_score']:.3f}, "
                  f"saved={result['compression_ratio']:.1%}")
        strategy_results[strategy_name] = results

    # Compile report
    report = {
        "schema_version": 1,
        "suite": suite.get("name", "llama-benchmark"),
        "model": args.model,
        "repeat": args.repeat,
        "fixture_count": len(fixtures),
        "strategy_count": len(strategies),
        "strategy_results": strategy_results,
    }

    if args.as_json:
        print(_format_json(report))
    else:
        print(_format_markdown(report))

    # Exit with error if any strategy failed alignment
    all_passed = all(
        r["alignment_passed"]
        for results in strategy_results.values()
        for r in results
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
