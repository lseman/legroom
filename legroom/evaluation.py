"""Reproducible quality, cost, latency, and memory evaluation for Legroom."""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from .compress import compress
from .config import CompressConfig
from .tokenizer import count_tokens_messages

Messages = list[dict[str, Any]]
Strategy = Callable[[Messages, str], Messages]
EvidenceKind = Literal["unknown", "heuristic", "model_graded", "task_verified"]


@dataclass(frozen=True)
class QualityEvidence:
    """Typed provenance for a quality score.

    A numeric score without its provenance is easy to overinterpret.  Evidence
    explicitly distinguishes cheap retention heuristics from model grading and
    executable downstream task verification.
    """

    kind: EvidenceKind
    score: float | None
    passed: bool | None
    details: dict[str, Any]

    def __post_init__(self) -> None:
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError("quality evidence score must be between 0 and 1")
        if self.kind == "unknown" and (self.score is not None or self.passed is not None):
            raise ValueError("unknown quality evidence cannot claim a score or outcome")


@dataclass(frozen=True)
class TaskEvaluation:
    score: float
    passed: bool
    details: dict[str, Any]
    evidence_kind: EvidenceKind = "task_verified"

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("task evaluation score must be between 0 and 1")
        if self.evidence_kind == "unknown":
            raise ValueError("a scored task evaluation cannot have unknown evidence")

    @property
    def evidence(self) -> QualityEvidence:
        return QualityEvidence(
            self.evidence_kind, self.score, self.passed, dict(self.details)
        )


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
            "heuristic",
        )


@dataclass(frozen=True)
class DeclaredCheck:
    """One deterministic, manifest-declared context preservation check."""

    name: str
    path: tuple[str | int, ...]
    expected: Any


class DeclaredChecksEvaluator:
    """Evaluate exact JSON values declared by a fixture manifest.

    This remains a deterministic context grader rather than a downstream task
    replay, so its evidence is deliberately labelled ``heuristic``.
    """

    def evaluate(self, fixture: Fixture, output: Messages) -> TaskEvaluation:
        results: list[dict[str, Any]] = []
        for check in fixture.declared_checks:
            found, actual = _resolve_path(output, check.path)
            matched = found and actual == check.expected
            results.append(
                {
                    "name": check.name,
                    "path": list(check.path),
                    "matched": matched,
                    "actual": actual if found else None,
                    "expected": check.expected,
                }
            )
        if not results:
            return TaskEvaluation(1.0, True, {"declared_checks": []}, "heuristic")
        score = sum(bool(result["matched"]) for result in results) / len(results)
        return TaskEvaluation(score, score == 1.0, {"declared_checks": results}, "heuristic")


class CompositeTaskEvaluator:
    """Require all supplied evaluators and retain their individual evidence."""

    def __init__(self, evaluators: Sequence[TaskEvaluator]) -> None:
        if not evaluators:
            raise ValueError("composite evaluator requires at least one evaluator")
        self._evaluators = tuple(evaluators)

    def evaluate(self, fixture: Fixture, output: Messages) -> TaskEvaluation:
        results = [evaluator.evaluate(fixture, output) for evaluator in self._evaluators]
        kind: EvidenceKind = (
            "task_verified"
            if any(result.evidence_kind == "task_verified" for result in results)
            else "model_graded"
            if any(result.evidence_kind == "model_graded" for result in results)
            else "heuristic"
        )
        return TaskEvaluation(
            min(result.score for result in results),
            all(result.passed for result in results),
            {"evaluations": [asdict(result.evidence) for result in results]},
            kind,
        )


class CallableTaskEvaluator:
    """Adapter for running repository tests, model calls, or agent tasks."""

    def __init__(self, callback: Callable[[Fixture, Messages], TaskEvaluation]) -> None:
        self._callback = callback

    def evaluate(self, fixture: Fixture, output: Messages) -> TaskEvaluation:
        result = self._callback(fixture, output)
        return result


@dataclass(frozen=True)
class Fixture:
    name: str
    description: str
    messages: Messages
    expected_terms: tuple[str, ...]
    declared_checks: tuple[DeclaredCheck, ...] = ()
    task: dict[str, Any] = field(default_factory=dict)


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
    quality_evidence: EvidenceKind
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


def legroom_strategy(**config_overrides: Any) -> Strategy:
    """Build a named benchmark strategy from validated ``CompressConfig`` fields."""
    unknown = sorted(set(config_overrides) - set(CompressConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown CompressConfig fields: {', '.join(unknown)}")

    def run(messages: Messages, model: str) -> Messages:
        values: dict[str, Any] = {"protect_recent": 2, "ccr_enabled": False}
        values.update(config_overrides)
        config = CompressConfig(**values)
        return compress(messages, model=model, config=config).messages

    return run


STRATEGIES: dict[str, Strategy] = {
    "identity": _identity,
    "recent_window": _recent_window,
    "head_tail": _head_tail,
    "legroom": _legroom,
}


def _resolve_path(document: Any, path: tuple[str | int, ...]) -> tuple[bool, Any]:
    current = document
    for segment in path:
        if isinstance(segment, int) and isinstance(current, list):
            index = segment if segment >= 0 else len(current) + segment
            if index < 0 or index >= len(current):
                return False, None
            current = current[index]
        elif isinstance(segment, str) and isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return False, None
    return True, current


def _load_declared_checks(definition: Mapping[str, Any]) -> tuple[DeclaredCheck, ...]:
    checks: list[DeclaredCheck] = []
    for index, raw in enumerate(definition.get("checks", ())):
        if not isinstance(raw, dict):
            raise TypeError("fixture checks must be objects")
        path = raw.get("path")
        if not isinstance(path, list) or not all(isinstance(item, (str, int)) for item in path):
            raise ValueError("fixture check path must be a list of strings or integers")
        if "expected" not in raw:
            raise ValueError("fixture check requires an expected value")
        checks.append(
            DeclaredCheck(
                str(raw.get("name", f"check_{index}")), tuple(path), raw["expected"]
            )
        )
    return tuple(checks)


def _manifest_strategies(
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Strategy]:
    strategies = dict(STRATEGIES)
    for definition in definitions:
        name = definition.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("strategy definition requires a non-empty name")
        if name in strategies:
            raise ValueError(f"duplicate strategy name: {name}")
        base = definition.get("base", "legroom")
        if base != "legroom":
            raise ValueError(f"unsupported strategy base: {base}")
        config = definition.get("config", {})
        if not isinstance(config, dict):
            raise TypeError("strategy config must be an object")
        strategies[name] = legroom_strategy(**config)
    return strategies


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
        strategies: Mapping[str, Strategy] | None = None,
    ) -> None:
        self.name = name
        self.fixtures = fixtures
        self.evaluator = evaluator or CompositeTaskEvaluator(
            (ExpectedTermsEvaluator(), DeclaredChecksEvaluator())
        )
        self.strategies = dict(strategies or STRATEGIES)

    @classmethod
    def load(
        cls, manifest_path: Path, *, evaluator: TaskEvaluator | None = None
    ) -> EvaluationSuite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema_version = manifest.get("schema_version")
        if schema_version not in {1, 2}:
            raise ValueError("unsupported evaluation suite schema")
        fixtures: list[Fixture] = []
        for definition in manifest["fixtures"]:
            path = manifest_path.parent / definition["path"]
            document = json.loads(path.read_text(encoding="utf-8"))
            task = definition.get("task", {})
            if not isinstance(task, dict):
                raise TypeError("fixture task metadata must be an object")
            fixtures.append(
                Fixture(
                    path.stem,
                    document["description"],
                    document["messages"],
                    tuple(definition.get("expected_terms", ())),
                    _load_declared_checks(definition),
                    dict(task),
                )
            )
        strategy_definitions = manifest.get("strategies", []) if schema_version == 2 else []
        if not isinstance(strategy_definitions, list):
            raise TypeError("manifest strategies must be a list")
        strategies = _manifest_strategies(strategy_definitions)
        return cls(manifest["name"], tuple(fixtures), evaluator, strategies)

    def run(
        self,
        *,
        model: str,
        repeat: int = 3,
        strategy_names: Sequence[str] | None = None,
    ) -> EvaluationReport:
        if repeat < 1:
            raise ValueError("repeat must be positive")
        selected = tuple(strategy_names or self.strategies)
        unknown = sorted(set(selected) - set(self.strategies))
        if unknown:
            raise ValueError(f"unknown evaluation strategies: {', '.join(unknown)}")
        if not selected:
            raise ValueError("at least one evaluation strategy is required")
        points: list[EvaluationPoint] = []
        for fixture in self.fixtures:
            before = count_tokens_messages(fixture.messages, model)
            for strategy_name in selected:
                strategy = self.strategies[strategy_name]
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
                        task.evidence_kind,
                        task.details,
                    )
                )
        return EvaluationReport(2, self.name, model, tuple(points))


def format_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# {report.suite} ({report.model})",
        "",
        "| Fixture | Strategy | Saved | Quality | Evidence | Invariants | p50 ms | Peak KiB |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for point in report.points:
        lines.append(
            f"| {point.fixture} | {point.strategy} | {point.compression_ratio:.1%} | "
            f"{point.task_success_score:.0%} | {point.quality_evidence} | "
            f"{point.invariant_score:.0%} | "
            f"{point.latency_ms_p50:.2f} | {point.peak_memory_bytes / 1024:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Quality-token tradeoff",
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
