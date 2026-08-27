from __future__ import annotations

import json
from pathlib import Path

import pytest

from legroom.analysis.evaluation import (
    CallableTaskEvaluator,
    CompositeTaskEvaluator,
    EvaluationSuite,
    ExpectedTermsEvaluator,
    Fixture,
    QualityEvidence,
    TaskEvaluation,
    format_markdown,
)


def test_versioned_evaluation_suite_runs(tmp_path: Path):
    (tmp_path / "trace.json").write_text(
        json.dumps({"description": "small trace", "messages": [{"role": "user", "content": "important term"}]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "name": "test-v1", "fixtures": [{"path": "trace.json", "expected_terms": ["important"]}]}),
        encoding="utf-8",
    )
    report = EvaluationSuite.load(manifest).run(model="gpt-4o", repeat=1)
    assert report.schema_version == 2
    assert {point.strategy for point in report.points} == {"identity", "recent_window", "head_tail", "legroom"}
    assert all(0 <= point.task_success_score <= 1 for point in report.points)
    assert len(report.strategies) == 4
    assert any(summary.pareto_optimal for summary in report.strategies)
    assert "quality_token_frontier" in report.to_dict()
    assert "| Fixture | Strategy |" in format_markdown(report)
    assert all(point.quality_evidence == "heuristic" for point in report.points)


def test_evaluation_rejects_unknown_schema(tmp_path: Path):
    manifest = tmp_path / "suite.json"
    manifest.write_text('{"schema_version": 3}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        EvaluationSuite.load(manifest)


def test_evaluation_runs_executable_downstream_callback(tmp_path: Path):
    (tmp_path / "trace.json").write_text(
        json.dumps({"description": "task", "messages": [{"role": "user", "content": "x"}]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "name": "tasks", "fixtures": [{"path": "trace.json"}]}),
        encoding="utf-8",
    )
    calls: list[str] = []

    def execute(fixture, output):
        calls.append(fixture.name)
        return TaskEvaluation(0.75, False, {"tests_passed": 3, "tests_total": 4})

    report = EvaluationSuite.load(
        manifest, evaluator=CallableTaskEvaluator(execute)
    ).run(model="gpt-4o", repeat=1)
    assert calls == ["trace"] * 4
    assert all(point.task_success_score == 0.75 for point in report.points)
    assert all(point.task_details["tests_passed"] == 3 for point in report.points)
    assert all(point.quality_evidence == "task_verified" for point in report.points)


def test_schema_v2_declares_exact_checks_and_ablation_strategies(tmp_path: Path):
    messages = [
        {"role": "tool", "tool_call_id": "call_1", "content": {"status": "cancelled"}},
        {"role": "user", "content": "continue"},
    ]
    (tmp_path / "trace.json").write_text(
        json.dumps({"description": "structured trace", "messages": messages}),
        encoding="utf-8",
    )
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "name": "structured-v2",
                "fixtures": [
                    {
                        "path": "trace.json",
                        "checks": [
                            {
                                "name": "tool result remains exact",
                                "path": [0, "content", "status"],
                                "expected": "cancelled",
                            }
                        ],
                    }
                ],
                "strategies": [
                    {
                        "name": "legroom_no_dedup",
                        "base": "legroom",
                        "config": {"cross_turn_dedup_enabled": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = EvaluationSuite.load(manifest)
    report = suite.run(
        model="gpt-4o", repeat=1, strategy_names=("identity", "legroom_no_dedup")
    )
    assert {point.strategy for point in report.points} == {
        "identity",
        "legroom_no_dedup",
    }
    assert all(point.task_passed for point in report.points)
    assert all(point.quality_evidence == "heuristic" for point in report.points)


def test_evidence_rejects_unknown_numeric_claims():
    with pytest.raises(ValueError, match="unknown"):
        QualityEvidence("unknown", 1.0, True, {})


def test_composite_evaluator_keeps_strongest_evidence_kind():
    evaluator = CompositeTaskEvaluator(
        (
            ExpectedTermsEvaluator(),
            CallableTaskEvaluator(
                lambda _fixture, _output: TaskEvaluation(
                    0.8, False, {"exit_code": 1}, "task_verified"
                )
            ),
        )
    )
    result = evaluator.evaluate(
        Fixture("trace", "", [{"role": "user", "content": "needle"}], ("needle",)),
        [{"role": "user", "content": "needle"}],
    )
    assert result.score == 0.8
    assert not result.passed
    assert result.evidence_kind == "task_verified"
