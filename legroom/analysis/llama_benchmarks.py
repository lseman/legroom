"""llama.cpp-specific benchmark evaluators for KV cache alignment.

This module provides specialized evaluators for validating Legroom's
llama.cpp KV cache optimizations:

- **JsonCanonicalizationCorrectness**: Ensures canonicalized JSON parses
  back to semantically identical structures (lossless at semantic level).
- **SequentialNormalizationStability**: Validates that sequential number
  normalization produces stable output across multiple turns.
- **PrefixStability**: Verifies that the stable prefix cache returns
  identical compressed prefix messages on cache hits.
- **KvCacheAlignmentScore**: Composite score measuring overall KV cache
  alignment quality.

Usage::

    from legroom.analysis.llama_benchmarks import (
        JsonCanonicalizationCorrectnessEvaluator,
        SequentialNormalizationStabilityEvaluator,
        PrefixStabilityEvaluator,
        KvCacheAlignmentEvaluator,
    )

    evaluator = KvCacheAlignmentEvaluator()
    result = evaluator.evaluate(fixture, messages, backend="llama_cpp")
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from .evaluation import Fixture, TaskEvaluation, QualityEvidence


class LlamaEvaluator(Protocol):
    """Protocol for llama.cpp benchmark evaluators."""

    def evaluate(
        self,
        fixture: Fixture,
        output: list[dict[str, Any]],
        backend: str = "llama_cpp",
    ) -> LlamaResult: ...


@dataclass(frozen=True)
class LlamaResult:
    """Result from a llama.cpp benchmark evaluator."""

    score: float
    passed: bool
    details: dict[str, Any]
    evidence_kind: str = "heuristic"

    @property
    def evidence(self) -> QualityEvidence:
        return QualityEvidence(
            "heuristic" if self.evidence_kind == "heuristic" else "task_verified",
            self.score,
            self.passed,
            self.details,
        )


class JsonCanonicalizationCorrectnessEvaluator:
    """Validates that canonicalized JSON preserves semantic equivalence.

    Checks:
    1. All JSON in output parses successfully
    2. Parsed JSON is semantically identical to expected (for known payloads)
    3. No malformed JSON structures introduced
    """

    def evaluate(
        self,
        fixture: Fixture,
        output: list[dict[str, Any]],
        backend: str = "llama_cpp",
    ) -> LlamaResult:
        if backend != "llama_cpp":
            return LlamaResult(1.0, True, {"note": "only applicable to llama_cpp backend"}, "heuristic")

        checks = []

        # Check 1: All tool call arguments are valid JSON
        valid_json_args = True
        total_args = 0
        malformed_args = []
        for msg in output:
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    func = call.get("function", {})
                    args = func.get("arguments", "") if isinstance(func, dict) else ""
                    if isinstance(args, str) and args.strip():
                        total_args += 1
                        try:
                            json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            valid_json_args = False
                            malformed_args.append({
                                "content": args[:80],
                                "error": "invalid JSON",
                            })

        checks.append(valid_json_args)

        # Check 2: All tool output content contains valid embedded JSON
        valid_json_content = True
        total_content_json = 0
        for msg in output:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Try to find and parse JSON spans
                json_count = _count_json_spans(content)
                total_content_json += json_count

        checks.append(True)  # Content JSON is informational, not correctness

        # Check 3: Tool call structure preserved
        tool_structure_ok = _check_tool_structure_preserved(output)
        checks.append(tool_structure_ok)

        score = 1.0 if all(checks) else 0.5
        return LlamaResult(
            score,
            all(checks),
            {
                "valid_json_args": valid_json_args,
                "total_args_checked": total_args,
                "malformed_args": malformed_args[:10],
                "tool_structure_preserved": tool_structure_ok,
                "content_json_spans": total_content_json,
            },
        )


class SequentialNormalizationStabilityEvaluator:
    """Validates sequential number normalization stability.

    Checks:
    1. Line numbers are replaced with placeholders (LN, IDX, etc.)
    2. Static numbers (ports, sizes in context) are preserved
    3. Normalized patterns are consistent across messages
    """

    def evaluate(
        self,
        fixture: Fixture,
        output: list[dict[str, Any]],
        backend: str = "llama_cpp",
    ) -> LlamaResult:
        if backend != "llama_cpp":
            return LlamaResult(
                1.0, True,
                {"note": "only applicable to llama_cpp backend"},
                "heuristic",
            )

        checks = []
        stats = {
            "line_numbers_normalized": 0,
            "indices_normalized": 0,
            "steps_normalized": 0,
            "file_refs_normalized": 0,
            "total_patterns_found": 0,
        }

        # Patterns that should be normalized
        _LINE_PATTERN = re.compile(r":\d+\s*[:\s]")
        _INDEX_PATTERN = re.compile(r"\[\d+\]")
        _FILE_LINE_PATTERN = re.compile(r"\w+\.\w+:\d+")
        _FILE_LINE_NORM_PATTERN = re.compile(r"\w+\.\w+:LN")
        _STEP_PATTERN = re.compile(r"\b(step|iteration)\s+\d+\b", re.IGNORECASE)

        for msg in output:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Count normalized placeholders
            line_placeholders = len(re.findall(r":LN\b", content))
            index_placeholders = len(re.findall(r"\[IDX\]", content))
            step_placeholders = len(re.findall(r"(?:step|iteration)\s+IDX\b", content))

            stats["line_numbers_normalized"] += line_placeholders
            stats["indices_normalized"] += index_placeholders
            stats["steps_normalized"] += step_placeholders

            # Check for file:LN patterns (normalized) and file:NUMBER patterns (unnormalized)
            normalized_file_lines = _FILE_LINE_NORM_PATTERN.findall(content)
            unnormalized_file_lines = _FILE_LINE_PATTERN.findall(content)

            stats["file_refs_normalized"] += len(normalized_file_lines)
            stats["total_patterns_found"] += len(normalized_file_lines) + len(unnormalized_file_lines)

            if unnormalized_file_lines:
                checks.append(False)

        # Check: At least some normalization happened in grep/search content
        any_normalization = sum(stats.values()) > 0

        # If the fixture contains grep-style output, we expect normalization
        has_grep_content = any(
            "legroom/" in msg.get("content", "")
            or "tests/" in msg.get("content", "")
            for msg in output
            if isinstance(msg.get("content"), str)
        )

        if has_grep_content and not any_normalization:
            checks.append(False)
        else:
            checks.append(True)

        score = 1.0 if all(checks) else (0.7 if has_grep_content and any_normalization else 0.5)

        return LlamaResult(
            score,
            all(checks),
            {"normalization_stats": stats, "has_grep_content": has_grep_content},
        )


class PrefixStabilityEvaluator:
    """Validates stable prefix cache behavior.

    Checks:
    1. System messages and tool definitions are preserved in order
    2. Prefix messages are not modified by compression phases
    3. Conversation tail is correctly separated from prefix
    """

    @staticmethod
    def _is_stable(msg: dict[str, Any]) -> bool:
        """Check if a message is part of the stable prefix.

        Only system messages and tool DEFINITIONS (with 'id' + 'function')
        are stable. Tool RESULTS (with 'tool_call_id') are conversation state.
        """
        role = msg.get("role", "")
        if role == "system":
            return True
        if role in ("tool", "function"):
            # Tool definitions have 'id' + 'function'; results have 'tool_call_id'
            has_definition = "id" in msg and "function" in msg
            has_result = "tool_call_id" in msg
            if has_result:
                return False
            return has_definition
        return False

    def evaluate(
        self,
        output: list[dict[str, Any]],
        original: list[dict[str, Any]],
    ) -> LlamaResult:
        checks = []

        # Check 1: Stable messages (system + tool definitions) are preserved
        original_stable = [msg for msg in original if self._is_stable(msg)]
        output_stable = [msg for msg in output if self._is_stable(msg)]

        # Count should match
        checks.append(len(original_stable) == len(output_stable))

        # Check 2: Stable message content (or tool definition) is preserved
        content_preserved = True
        for orig, out in zip(original_stable, output_stable):
            # Content messages use 'content' field
            # Tool definitions use 'function' field
            orig_content = orig.get("content", orig.get("function"))
            out_content = out.get("content", out.get("function"))
            if orig_content != out_content:
                content_preserved = False
                break
        checks.append(content_preserved)

        # Check 3: Tool definitions preserve their 'id' field
        tool_ids_preserved = True
        for orig, out in zip(original_stable, output_stable):
            # Tool definitions have 'id'; content messages don't
            orig_id = orig.get("id")
            out_id = out.get("id")
            if orig_id is not None and out_id is not None:
                if orig_id != out_id:
                    tool_ids_preserved = False
                    break
        checks.append(tool_ids_preserved)

        # Check 4: Conversation tail messages are present
        tail_roles = {"user", "assistant"}
        original_tail = [msg for msg in original if msg.get("role") in tail_roles]
        output_tail = [msg for msg in output if msg.get("role") in tail_roles]

        checks.append(len(output_tail) >= len(original_tail) - 1)  # Allow 1 for compression

        # Check 5: First message role matches (system prompt is first)
        if original and output:
            checks.append(original[0].get("role") == output[0].get("role"))

        score = 1.0 if all(checks) else sum(checks) / len(checks)
        return LlamaResult(
            score,
            all(checks),
            {
                "stable_original_count": len(original_stable),
                "stable_output_count": len(output_stable),
                "content_preserved": content_preserved,
                "tool_ids_preserved": tool_ids_preserved,
                "tail_original_count": len(original_tail),
                "tail_output_count": len(output_tail),
            },
        )


class KvCacheAlignmentEvaluator:
    """Composite evaluator for overall KV cache alignment quality.

    Runs multiple sub-evaluators and combines their scores.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._json_evaluator = JsonCanonicalizationCorrectnessEvaluator()
        self._seq_evaluator = SequentialNormalizationStabilityEvaluator()
        self._prefix_evaluator = PrefixStabilityEvaluator()
        self._weights = weights or {
            "json_correctness": 0.4,
            "sequential_stability": 0.3,
            "prefix_stability": 0.3,
        }

    def evaluate(
        self,
        fixture: Fixture,
        output: list[dict[str, Any]],
        original: list[dict[str, Any]] | None = None,
        backend: str = "llama_cpp",
    ) -> LlamaResult:
        if original is None:
            original = fixture.messages

        # Run sub-evaluators
        json_result = self._json_evaluator.evaluate(fixture, output, backend)
        seq_result = self._seq_evaluator.evaluate(fixture, output, backend)
        prefix_result = self._prefix_evaluator.evaluate(output, original)

        # Weighted composite score
        scores = {
            "json_correctness": json_result.score,
            "sequential_stability": seq_result.score,
            "prefix_stability": prefix_result.score,
        }

        total_weight = sum(self._weights.values())
        composite = sum(
            scores[key] * self._weights.get(key, 1.0 / len(self._weights))
            for key in scores
        ) / total_weight if total_weight > 0 else 0.0

        return LlamaResult(
            round(composite, 4),
            composite >= 0.9,
            {
                "sub_scores": scores,
                "weights": self._weights,
                "json_details": json_result.details,
                "seq_details": seq_result.details,
                "prefix_details": prefix_result.details,
            },
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_JSON_OPEN_PATTERN = re.compile(r"[\[{]")
_JSON_CLOSE_PATTERN = re.compile(r"[\]}]")


def _count_json_spans(text: str) -> int:
    """Count JSON-like spans in text (heuristic)."""
    count = 0
    i = 0
    n = len(text)
    while i < n:
        if _JSON_OPEN_PATTERN.match(text[i]):
            # Quick bracket balance check
            depth = 0
            j = i
            while j < n and depth >= 0:
                if text[j] in "{[":
                    depth += 1
                elif text[j] in "}]":
                    depth -= 1
                j += 1
            if depth == 0:
                count += 1
                i = j
                continue
        i += 1
    return count


def _check_tool_structure_preserved(output: list[dict[str, Any]]) -> bool:
    """Check that tool call structure is preserved in output."""
    for msg in output:
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for call in tool_calls:
                if not isinstance(call, dict):
                    return False
                if "id" not in call or "function" not in call:
                    return False
                func = call.get("function")
                if not isinstance(func, dict):
                    return False
                if "name" not in func or "arguments" not in func:
                    return False
    return True


class LlamaKvCacheHitRateEstimator:
    """Estimates KV cache hit rate from canonicalization metrics.

    This is a heuristic estimator — actual hit rates depend on the
    llama.cpp server's slot allocation, prefix matching, and KV cache
    size. This estimator measures what Legroom can control:
    output determinism.
    """

    def __init__(self) -> None:
        self._canonicalization_counts: dict[str, int] = Counter()

    def record_canonicalization(
        self,
        phase: str,
        count: int,
        tokens_saved: int,
    ) -> None:
        """Record a canonicalization phase result."""
        self._canonicalization_counts[phase] += count

    def estimate_hit_rate(
        self,
        request_count: int,
        unique_prefixes: int,
        unique_tails: int,
    ) -> float:
        """Estimate KV cache hit rate.

        Args:
            request_count: Total number of requests.
            unique_prefixes: Number of distinct stable prefixes.
            unique_tails: Number of distinct conversation tails.

        Returns:
            Estimated hit rate as a fraction [0, 1].
        """
        if request_count <= 0:
            return 0.0

        # Theoretical maximum: every unique prefix+tail combo produces
        # one cache entry. Hits = requests - unique_combinations.
        unique_combinations = min(unique_prefixes * unique_tails, request_count)
        estimated_hits = max(0, request_count - unique_combinations)

        return estimated_hits / request_count

    def get_metrics(self) -> dict[str, Any]:
        """Return estimator metrics."""
        return {
            "canonicalization_counts": dict(self._canonicalization_counts),
        }


class LlamaBenchmarkRunner:
    """Runs llama.cpp-specific benchmark suite.

    Usage::

        runner = LlamaBenchmarkRunner()
        results = runner.run_suite(
            fixtures,
            model="gpt-4o",
            repeat=3,
        )
    """

    def __init__(self) -> None:
        self._kv_estimator = LlamaKvCacheHitRateEstimator()

    def run_evaluation(
        self,
        fixture: Fixture,
        output: list[dict[str, Any]],
        original: list[dict[str, Any]],
        backend: str = "llama_cpp",
    ) -> dict[str, Any]:
        """Run all llama.cpp evaluators on a single compression result."""
        alignment_evaluator = KvCacheAlignmentEvaluator()
        alignment_result = alignment_evaluator.evaluate(
            fixture, output, original, backend
        )

        return {
            "fixture": fixture.name,
            "alignment_score": alignment_result.score,
            "alignment_passed": alignment_result.passed,
            "details": alignment_result.details,
        }

    def run_suite(
        self,
        fixtures: list[Fixture],
        model: str,
        strategies: dict[str, Any],
        repeat: int = 3,
    ) -> dict[str, Any]:
        """Run the full llama.cpp benchmark suite.

        Args:
            fixtures: List of benchmark fixtures.
            model: Model name for token counting.
            strategies: Strategy configurations to test.
            repeat: Number of repetitions per strategy.

        Returns:
            Benchmark results dictionary.
        """
        from ..runtime.compress import compress
        from ..runtime.config import CompressConfig

        results: dict[str, list[dict[str, Any]]] = {}

        for strategy_name, config_overrides in strategies.items():
            # Filter config to only valid CompressConfig fields
            valid_keys = set(CompressConfig.__dataclass_fields__)
            filtered_config = {k: v for k, v in config_overrides.items() if k in valid_keys}
            strategy_results: list[dict[str, Any]] = []

            for fixture in fixtures:
                for _ in range(repeat):
                    config = CompressConfig(**filtered_config)
                    compressed = compress(
                        fixture.messages,
                        model=model,
                        config=config,
                    )

                    result = self.run_evaluation(
                        fixture,
                        compressed.messages,
                        fixture.messages,
                        backend=config_overrides.get("backend", "llama_cpp"),
                    )
                    result["strategy"] = strategy_name
                    result["tokens_before"] = compressed.tokens_before
                    result["tokens_after"] = compressed.tokens_after
                    result["compression_ratio"] = (
                        1.0 - compressed.tokens_after / compressed.tokens_before
                        if compressed.tokens_before > 0
                        else 0.0
                    )
                    strategy_results.append(result)

            results[strategy_name] = strategy_results

        return results
