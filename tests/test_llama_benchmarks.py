"""Tests for llama.cpp benchmark evaluators."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legroom.analysis.evaluation import Fixture  # noqa: E402
from legroom.analysis.llama_benchmarks import (  # noqa: E402
    JsonCanonicalizationCorrectnessEvaluator,
    KvCacheAlignmentEvaluator,
    LlamaBenchmarkRunner,
    LlamaKvCacheHitRateEstimator,
    PrefixStabilityEvaluator,
    SequentialNormalizationStabilityEvaluator,
    _check_tool_structure_preserved,
    _count_json_spans,
)

# ---------------------------------------------------------------------------
# Fixtures for tests
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_fixture():
    return Fixture(
        name="test",
        description="test fixture",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2=4"},
        ],
        expected_terms=["assistant", "helpful"],
    )


@pytest.fixture
def tool_call_fixture():
    return Fixture(
        name="tool_test",
        description="tool call fixture",
        messages=[
            {"role": "system", "content": "You have tools."},
            {"role": "user", "content": "Read a file."},
            {"role": "assistant", "content": "Reading...", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": '{"path": "test.py", "offset": 0, "limit": 100}',
                }},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents here"},
        ],
        expected_terms=["read_file"],
    )


@pytest.fixture
def grep_fixture():
    return Fixture(
        name="grep_test",
        description="grep result fixture",
        messages=[
            {"role": "user", "content": "Find CompressConfig usages."},
            {"role": "assistant", "content": "Searching.", "tool_calls": [
                {"id": "call_grep", "type": "function", "function": {
                    "name": "grep_files",
                    "arguments": '{"pattern": "CompressConfig", "limit": 50}',
                }},
            ]},
            {"role": "tool", "tool_call_id": "call_grep", "content": (
                "legroom/compress.py:14:    config: CompressConfig | None = None,\n"
                "legroom/compress.py:22:    cfg = config or CompressConfig()\n"
                "legroom/cli.py:41:        config = CompressConfig()\n"
                "legroom/proxy_server.py:88:    config = CompressConfig()\n"
            )},
        ],
        expected_terms=["CompressConfig"],
    )


# ---------------------------------------------------------------------------
# JsonCanonicalizationCorrectnessEvaluator tests
# ---------------------------------------------------------------------------

class TestJsonCanonicalizationCorrectnessEvaluator:
    def test_openai_backend_returns_perfect(self, simple_fixture):
        evaluator = JsonCanonicalizationCorrectnessEvaluator()
        result = evaluator.evaluate(simple_fixture, simple_fixture.messages, backend="openai")
        assert result.score == 1.0
        assert result.passed is True

    def test_valid_json_args(self, tool_call_fixture):
        evaluator = JsonCanonicalizationCorrectnessEvaluator()
        result = evaluator.evaluate(tool_call_fixture, tool_call_fixture.messages, backend="llama_cpp")
        assert result.passed is True
        assert result.details["valid_json_args"] is True

    def test_malformed_json_args(self, tool_call_fixture):
        # Inject malformed JSON
        bad_messages = [
            {**msg} for msg in tool_call_fixture.messages
        ]
        for msg in bad_messages:
            if "tool_calls" in msg:
                msg["tool_calls"][0]["function"]["arguments"] = '{"path": INVALID}'
        evaluator = JsonCanonicalizationCorrectnessEvaluator()
        result = evaluator.evaluate(tool_call_fixture, bad_messages, backend="llama_cpp")
        assert result.passed is False
        assert result.details["valid_json_args"] is False

    def test_tool_structure_preserved(self, tool_call_fixture):
        result = _check_tool_structure_preserved(tool_call_fixture.messages)
        assert result is True

    def test_tool_structure_broken(self):
        broken = [
            {"role": "assistant", "content": "hi", "tool_calls": [
                {"id": "call_1"},  # Missing "function"
            ]},
        ]
        assert _check_tool_structure_preserved(broken) is False

    def test_no_tool_calls(self, simple_fixture):
        result = _check_tool_structure_preserved(simple_fixture.messages)
        assert result is True


# ---------------------------------------------------------------------------
# SequentialNormalizationStabilityEvaluator tests
# ---------------------------------------------------------------------------

class TestSequentialNormalizationStabilityEvaluator:
    def test_openai_backend_returns_perfect(self, simple_fixture):
        evaluator = SequentialNormalizationStabilityEvaluator()
        result = evaluator.evaluate(simple_fixture, simple_fixture.messages, backend="openai")
        assert result.score == 1.0
        assert result.passed is True

    def test_grep_content_with_normalized_numbers(self, grep_fixture):
        evaluator = SequentialNormalizationStabilityEvaluator()
        # Simulate already-normalized output (replace line numbers with LN)
        normalized_output = [
            {**msg} for msg in grep_fixture.messages
        ]
        for msg in normalized_output:
            content = msg.get("content", "")
            if isinstance(content, str) and ":" in content:
                content = content.replace("legroom/compress.py:14:", "legroom/compress.py:LN    ")
                content = content.replace("legroom/compress.py:22:", "legroom/compress.py:LN    ")
                content = content.replace("legroom/cli.py:41:", "legroom/cli.py:LN    ")
                content = content.replace("legroom/proxy_server.py:88:", "legroom/proxy_server.py:LN    ")
                msg["content"] = content

        result = evaluator.evaluate(grep_fixture, normalized_output, backend="llama_cpp")
        assert result.passed is True
        stats = result.details["normalization_stats"]
        # Should have found normalized file:LN patterns
        assert stats["file_refs_normalized"] > 0, f"Expected normalized file refs, got {stats}"
        # Should have no unnormalized file:number patterns
        total = stats["total_patterns_found"]
        file_norm = stats["file_refs_normalized"]
        assert file_norm <= total, f"Normalized count {file_norm} exceeds total {total}"

    def test_grep_content_unnormalized(self, grep_fixture):
        evaluator = SequentialNormalizationStabilityEvaluator()
        result = evaluator.evaluate(grep_fixture, grep_fixture.messages, backend="llama_cpp")
        # Should detect unnormalized file:LINE patterns
        assert result.passed is False


# ---------------------------------------------------------------------------
# PrefixStabilityEvaluator tests
# ---------------------------------------------------------------------------

class TestPrefixStabilityEvaluator:
    def test_prefix_preserved(self):
        original = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        output = list(original)
        evaluator = PrefixStabilityEvaluator()
        result = evaluator.evaluate(output, original)
        assert result.passed is True
        assert result.score == 1.0

    def test_prefix_modified(self):
        original = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        output = [
            {"role": "system", "content": "Modified system prompt."},  # Changed
            {"role": "user", "content": "Hello"},
        ]
        evaluator = PrefixStabilityEvaluator()
        result = evaluator.evaluate(output, original)
        assert result.passed is False

    def test_tool_id_preserved(self):
        original = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "What happened?"},
        ]
        output = list(original)
        evaluator = PrefixStabilityEvaluator()
        result = evaluator.evaluate(output, original)
        assert result.passed is True

    def test_tool_definition_modified(self):
        # Tool definitions (with 'id' + 'function') ARE stable and should be preserved
        original = [
            {"role": "tool", "id": "call_1", "function": {"name": "read", "arguments": '{"path": "a.py"}'}},
            {"role": "user", "content": "What happened?"},
        ]
        output = [
            {"role": "tool", "id": "call_1", "function": {"name": "read", "arguments": '{"path": "b.py"}'}},  # Modified
            {"role": "user", "content": "What happened?"},
        ]
        evaluator = PrefixStabilityEvaluator()
        result = evaluator.evaluate(output, original)
        assert result.passed is False


# ---------------------------------------------------------------------------
# KvCacheAlignmentEvaluator tests
# ---------------------------------------------------------------------------

class TestKvCacheAlignmentEvaluator:
    def test_perfect_alignment(self, tool_call_fixture):
        evaluator = KvCacheAlignmentEvaluator()
        result = evaluator.evaluate(
            tool_call_fixture,
            tool_call_fixture.messages,
            tool_call_fixture.messages,
            backend="llama_cpp",
        )
        assert result.passed is True
        assert result.score > 0.9

    def test_json_canonicalization_correct(self, tool_call_fixture):
        evaluator = KvCacheAlignmentEvaluator()
        result = evaluator.evaluate(
            tool_call_fixture,
            tool_call_fixture.messages,
            tool_call_fixture.messages,
            backend="llama_cpp",
        )
        assert result.details["sub_scores"]["json_correctness"] == 1.0

    def test_openai_backend_skips_json_check(self, tool_call_fixture):
        evaluator = KvCacheAlignmentEvaluator()
        result = evaluator.evaluate(
            tool_call_fixture,
            tool_call_fixture.messages,
            tool_call_fixture.messages,
            backend="openai",
        )
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# LlamaKvCacheHitRateEstimator tests
# ---------------------------------------------------------------------------

class TestLlamaKvCacheHitRateEstimator:
    def test_zero_requests(self):
        estimator = LlamaKvCacheHitRateEstimator()
        assert estimator.estimate_hit_rate(0, 1, 1) == 0.0

    def test_single_request(self):
        estimator = LlamaKvCacheHitRateEstimator()
        # 1 request, 1 prefix, 1 tail = 0 hits
        assert estimator.estimate_hit_rate(1, 1, 1) == 0.0

    def test_repeated_same_prefix_tail(self):
        estimator = LlamaKvCacheHitRateEstimator()
        # 10 requests, 1 prefix, 1 tail = 9 hits
        rate = estimator.estimate_hit_rate(10, 1, 1)
        assert rate == 0.9

    def test_all_different_tails(self):
        estimator = LlamaKvCacheHitRateEstimator()
        # 10 requests, 1 prefix, 10 tails = 0 hits
        rate = estimator.estimate_hit_rate(10, 1, 10)
        assert rate == 0.0

    def test_mixed_scenarios(self):
        estimator = LlamaKvCacheHitRateEstimator()
        # 10 requests, 2 prefixes, 5 tails = 10 combinations, 0 hits
        rate = estimator.estimate_hit_rate(10, 2, 5)
        assert rate == 0.0

        # 100 requests, 2 prefixes, 5 tails = 10 combinations, 90 hits
        rate = estimator.estimate_hit_rate(100, 2, 5)
        assert rate == 0.9

    def test_record_canonicalization(self):
        estimator = LlamaKvCacheHitRateEstimator()
        estimator.record_canonicalization("json_canonicalization", 5, 50)
        estimator.record_canonicalization("sequential_normalization", 10, 30)
        metrics = estimator.get_metrics()
        assert metrics["canonicalization_counts"]["json_canonicalization"] == 5
        assert metrics["canonicalization_counts"]["sequential_normalization"] == 10


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_count_json_spans_dict(self):
        text = 'Here is some JSON: {"key": "value", "nested": {"a": 1}}'
        count = _count_json_spans(text)
        assert count >= 1

    def test_count_json_spans_array(self):
        text = 'Results: [1, 2, 3, {"a": 1}]'
        count = _count_json_spans(text)
        assert count >= 1

    def test_count_json_spans_none(self):
        text = "No JSON here, just plain text."
        count = _count_json_spans(text)
        assert count == 0

    def test_count_json_spans_nested(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        count = _count_json_spans(text)
        assert count >= 1


# ---------------------------------------------------------------------------
# LlamaBenchmarkRunner integration tests
# ---------------------------------------------------------------------------

class TestLlamaBenchmarkRunner:
    def test_run_evaluation(self, tool_call_fixture):
        runner = LlamaBenchmarkRunner()
        result = runner.run_evaluation(
            tool_call_fixture,
            tool_call_fixture.messages,
            tool_call_fixture.messages,
            backend="llama_cpp",
        )
        assert "fixture" in result
        assert "alignment_score" in result
        assert "alignment_passed" in result
        assert result["fixture"] == "tool_test"

    def test_run_suite_structure(self, tool_call_fixture):
        runner = LlamaBenchmarkRunner()
        fixtures = [tool_call_fixture]
        strategies = {
            "test_strategy": {
                "backend": "llama_cpp",
                "protect_recent": 2,
                "cache_align_enabled": True,
                "name": "test_strategy",
            }
        }
        results = runner.run_suite(
            fixtures=fixtures,
            model="gpt-4o",
            strategies=strategies,
            repeat=1,
        )
        assert "test_strategy" in results
        assert len(results["test_strategy"]) == 1
        assert "tokens_before" in results["test_strategy"][0]
        assert "tokens_after" in results["test_strategy"][0]
        assert "compression_ratio" in results["test_strategy"][0]
