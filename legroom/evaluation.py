"""Reproducible quality, cost, latency, and memory evaluation for Legroom."""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .compress import compress
from .config import CompressConfig
from .tokenizer import count_tokens_messages

Messages = list[dict[str, Any]]
Strategy = Callable[[Messages, str], Messages]


@dataclass(frozen=True)
class TaskEvaluation:
    score: float
    passed: bool
    details: dict[str, Any]


class TaskEvaluator(Protocol):
    """Seam for executable downstream task evaluation."""

    def evaluate(self, fixture: Fixture, output: Messages) -> TaskEvaluation: ...


class ExpectedTermsEvaluator:
    def evaluate(self, fixture: Fixture, output: Messages) -> TaskEvaluation:
        text = _serialized(output).lower()
        matched = [term for term in fixture.expected_terms if term.lower() in text]
        score = len(matched) / len(fixture.expected_terms) if fixture.expected_terms else 1.0
        return TaskEvaluation(
            score,
            score == 1.0,
            {"matched_terms": matched, "expected_terms": list(fixture.expected_terms)},
        )


class CallableTaskEvaluator:
    """Adapter for running repository tests, model calls, or agent tasks."""

    def __init__(self, callback: Callable[[Fixture, Messages], TaskEvaluation]) -> None:
        self._callback = callback

    def evaluate(self, fixture: Fixture, output: Messages) -> TaskEvaluation:
        result = self._callback(fixture, output)
        if not 0 <= result.score <= 1:
            raise ValueError("task evaluator score must be between 0 and 1")
        return result


@dataclass(frozen=True)
class Fixture:
    name: str
    description: str
    messages: Messages
    expected_terms: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationPoint:
    fixture: str
    strategy: str
    tokens_before: int
    tokens_after: int
    compression_ratio: float
    latency_ms_p50: float
    latency_ms_p95: float
    peak_memory_bytes: int
    invariant_score: float
    task_success_score: float
    task_passed: bool
    task_details: dict[str, Any]


@dataclass(frozen=True)
class StrategySummary:
    strategy: str
    mean_compression_ratio: float
    mean_quality_score: float
    mean_invariant_score: float
    latency_ms_p50: float
    peak_memory_bytes: int
    pareto_optimal: bool


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: int
    suite: str
    model: str
    points: tuple[EvaluationPoint, ...]

    @property
    def strategies(self) -> tuple[StrategySummary, ...]:
        grouped: dict[str, list[EvaluationPoint]] = {}
        for point in self.points:
            grouped.setdefault(point.strategy, []).append(point)
        raw = [
            StrategySummary(
                strategy=name,
                mean_compression_ratio=statistics.fmean(p.compression_ratio for p in points),
                mean_quality_score=statistics.fmean(p.task_success_score for p in points),
                mean_invariant_score=statistics.fmean(p.invariant_score for p in points),
                latency_ms_p50=statistics.median(p.latency_ms_p50 for p in points),
                peak_memory_bytes=max(p.peak_memory_bytes for p in points),
                pareto_optimal=False,
            )
            for name, points in grouped.items()
        ]
        summaries: list[StrategySummary] = []
        for candidate in raw:
            dominated = any(
                other.strategy != candidate.strategy
                and other.mean_compression_ratio >= candidate.mean_compression_ratio
                and other.mean_quality_score >= candidate.mean_quality_score
                and (
                    other.mean_compression_ratio > candidate.mean_compression_ratio
                    or other.mean_quality_score > candidate.mean_quality_score
                )
                for other in raw
            )
            summaries.append(
                StrategySummary(**{**asdict(candidate), "pareto_optimal": not dominated})
            )
        return tuple(summaries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "model": self.model,
            "points": [asdict(point) for point in self.points],
            "quality_token_frontier": [asdict(summary) for summary in self.strategies],
        }


def _identity(messages: Messages, model: str) -> Messages:
    return messages


def _recent_window(messages: Messages, model: str, keep: int = 3) -> Messages:
    if len(messages) <= keep:
        return messages
    omitted = len(messages) - keep
    return [{"role": "system", "content": f"[{omitted} earlier messages omitted]"}, *messages[-keep:]]


def _head_tail(messages: Messages, model: str, chars: int = 800) -> Messages:
    output: Messages = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and len(content) > chars:
            half = chars // 2
            output.append({**message, "content": f"{content[:half]}\n[…truncated…]\n{content[-half:]}"})
        else:
            output.append(message)
    return output


def _legroom(messages: Messages, model: str) -> Messages:
    return compress(
        messages,
        model=model,
        config=CompressConfig(protect_recent=2, ccr_enabled=False),
    ).messages


STRATEGIES: dict[str, Strategy] = {
    "identity": _identity,
    "recent_window": _recent_window,
    "head_tail": _head_tail,
    "legroom": _legroom,
}


def _serialized(messages: Messages) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def _invariant_score(original: Messages, output: Messages) -> float:
    """Score structural fields and the current turn, which must remain exact."""
    checks: list[bool] = []
    for message in original:
        for key, value in message.items():
            if key != "content":
                checks.append(any(candidate.get(key) == value for candidate in output))
    if original:
        checks.append(original[-1] in output)
    return sum(checks) / len(checks) if checks else 1.0


class EvaluationSuite:
    def __init__(
        self,
        name: str,
        fixtures: tuple[Fixture, ...],
        evaluator: TaskEvaluator | None = None,
    ) -> None:
        self.name = name
        self.fixtures = fixtures
        self.evaluator = evaluator or ExpectedTermsEvaluator()

    @classmethod
    def load(
        cls, manifest_path: Path, *, evaluator: TaskEvaluator | None = None
    ) -> EvaluationSuite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported evaluation suite schema")
        fixtures: list[Fixture] = []
        for definition in manifest["fixtures"]:
            path = manifest_path.parent / definition["path"]
            document = json.loads(path.read_text(encoding="utf-8"))
            fixtures.append(
                Fixture(
                    path.stem,
                    document["description"],
                    document["messages"],
                    tuple(definition.get("expected_terms", ())),
                )
            )
        return cls(manifest["name"], tuple(fixtures), evaluator)

    def run(self, *, model: str, repeat: int = 3) -> EvaluationReport:
        if repeat < 1:
            raise ValueError("repeat must be positive")
        points: list[EvaluationPoint] = []
        for fixture in self.fixtures:
            before = count_tokens_messages(fixture.messages, model)
            for strategy_name, strategy in STRATEGIES.items():
                latencies: list[float] = []
                peaks: list[int] = []
                output = fixture.messages
                for _ in range(repeat):
                    tracemalloc.start()
                    started = time.perf_counter()
                    output = strategy(fixture.messages, model)
                    latencies.append((time.perf_counter() - started) * 1000)
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peaks.append(peak)
                after = count_tokens_messages(output, model)
                task = self.evaluator.evaluate(fixture, output)
                ordered = sorted(latencies)
                p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
                points.append(
                    EvaluationPoint(
                        fixture.name,
                        strategy_name,
                        before,
                        after,
                        1.0 - after / before if before else 0.0,
                        statistics.median(latencies),
                        ordered[p95_index],
                        max(peaks),
                        _invariant_score(fixture.messages, output),
                        task.score,
                        task.passed,
                        task.details,
                    )
                )
        return EvaluationReport(1, self.name, model, tuple(points))


def format_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# {report.suite} ({report.model})",
        "",
        "| Fixture | Strategy | Saved | Quality | Invariants | p50 ms | Peak KiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for point in report.points:
        lines.append(
            f"| {point.fixture} | {point.strategy} | {point.compression_ratio:.1%} | "
            f"{point.task_success_score:.0%} | {point.invariant_score:.0%} | "
            f"{point.latency_ms_p50:.2f} | {point.peak_memory_bytes / 1024:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Quality–token tradeoff",
            "",
            "| Strategy | Mean saved | Mean quality | Mean invariants | Pareto |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for summary in report.strategies:
        lines.append(
            f"| {summary.strategy} | {summary.mean_compression_ratio:.1%} | "
            f"{summary.mean_quality_score:.0%} | {summary.mean_invariant_score:.0%} | "
            f"{'yes' if summary.pareto_optimal else 'no'} |"
        )
    return "\n".join(lines)
