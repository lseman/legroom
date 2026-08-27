"""Sequential number normalization for llama.cpp KV cache alignment.

llama.cpp's KV cache matches tokenized prefixes byte-for-byte. Sequential
numbers in tool outputs — line numbers in grep results, array indices,
step numbers — change on every turn and bust the KV cache even though
the semantic content is identical.

    grep turn 1:  file.py:42: code()
    grep turn 2:  file.py:43: code()
    → Different tokens → cache miss
    → After normalization: file.py:LN: code() → cache hit

This module detects and normalizes sequential number patterns:

1. **Line numbers** — ``file.py:42`` → ``file.py:LN`` (grep, search results)
2. **Array indices** — ``[0]``, ``[1]`` → ``[IDX]`` (tool outputs)
3. **Step numbers** — ``step 1``, ``step 2`` → ``step IDX`` (iteration logs)
4. **Result numbers** — ``result 1``, ``result 2`` → ``result IDX`` (search)

Only numbers that appear to be sequential within their local context are
normalized. Static numbers (ports, sizes, IDs) are preserved.

Usage::

    from legroom.compressors.sequential_normalizer import (
        SequentialNumberNormalizer,
    )

    normalizer = SequentialNumberNormalizer()
    result = normalizer.normalize(text, backend="llama_cpp")

The normalization is **lossy** at the byte level but **lossless** at the
semantic level for the model's reasoning. Exact numbers can be retrieved
via the CCR store when needed for code edits or patches.
"""

from __future__ import annotations

import re

# Pre-compiled regex patterns
# Line numbers: word:42 (grep results, file references)
_PATTERN_LINE = re.compile(
    r"(\b\w[\w.]*\.\w+):(\d+)"
)

# Array indices: [42]
_PATTERN_INDEX = re.compile(
    r"\[(\d+)\]"
)

# Step/iteration numbers: step 1, iteration 3
_PATTERN_STEP = re.compile(
    r"\b(step|iteration|iter)\s+(\d+)\b",
    re.IGNORECASE,
)

# Result numbers: result 1, match 3
_PATTERN_RESULT = re.compile(
    r"\b(result|match|hit|found)\s+(\d+)\b",
    re.IGNORECASE,
)

# Log line numbers: [line:42], Line 42
_PATTERN_LOG_LINE = re.compile(
    r"(?:\[line[:\s]+|Line\s+)(\d+)",
    re.IGNORECASE,
)

# Table row numbers: Row 1, Row 2
_PATTERN_ROW = re.compile(
    r"\b(Row|Line|Item)\s+(\d+)\b",
    re.IGNORECASE,
)

# Minimum number of sequential occurrences before normalization triggers.
# A single number like "port 8080" is NOT sequential.
_MIN_SEQ_LENGTH = 2


class SeqNormalizeResult:
    """Result of sequential number normalization."""

    def __init__(
        self,
        text: str,
        normalized_count: int = 0,
        tokens_saved: int = 0,
    ) -> None:
        self.text = text
        self.normalized_count = normalized_count
        self.tokens_saved = tokens_saved


class SequentialNumberNormalizer:
    """Normalizes sequential numbers in text for KV cache alignment.

    Detects number patterns that vary across turns (line numbers, indices)
    and replaces them with fixed placeholders (LN, IDX). Static numbers
    (ports, IDs, sizes) are preserved.

    The normalization is gated behind the llama_cpp backend flag because
    it is lossy — the exact numbers are lost. For openai backend there is
    no KV cache to align against, so normalization is skipped.
    """

    def normalize(
        self,
        text: str,
        *,
        backend: str = "openai",
        protected_indices: set[int] | None = None,
    ) -> SeqNormalizeResult:
        """Normalize sequential numbers in text.

        For ``backend="llama_cpp"``, sequential number patterns are replaced
        with placeholders. For other backends, returns text unchanged.

        Args:
            text: The text to normalize.
            backend: Target backend — only ``"llama_cpp"`` triggers
                normalization.
            protected_indices: Not used for text normalization (reserved for
                message-level protection).

        Returns:
            SeqNormalizeResult with normalized text and stats.
        """
        if backend != "llama_cpp":
            return SeqNormalizeResult(text=text)

        original = text
        normalized = text

        # Apply all normalization patterns in order
        patterns = [
            (self._normalize_line_numbers, "line"),
            (self._normalize_indices, "index"),
            (self._normalize_steps, "step"),
            (self._normalize_results, "result"),
            (self._normalize_log_lines, "log_line"),
            (self._normalize_rows, "row"),
        ]

        total_normalized = 0
        for pattern_fn, _name in patterns:
            normalized, count = pattern_fn(normalized)
            total_normalized += count

        if original == normalized:
            return SeqNormalizeResult(text=normalized, normalized_count=0)

        # Estimate tokens saved (rough: each normalization saves ~1-3 tokens)
        tokens_saved = max(0, len(original) - len(normalized)) // 10 + total_normalized

        return SeqNormalizeResult(
            text=normalized,
            normalized_count=total_normalized,
            tokens_saved=tokens_saved,
        )

    def _normalize_line_numbers(self, text: str) -> tuple[str, int]:
        """Normalize file line numbers: ``file.py:42`` → ``file.py:LN``.

        Matches grep-style output patterns like:
            src/main.py:42: def foo():
            tests/test.py:100: assert True
        """
        count = 0

        def replacer(m: re.Match) -> str:
            nonlocal count
            count += 1
            return f"{m.group(1)}:LN"

        return _PATTERN_LINE.sub(replacer, text), count

    def _normalize_indices(self, text: str) -> tuple[str, int]:
        """Normalize array indices: ``[42]`` → ``[IDX]``.

        Matches JSON array indices and Python list indices:
            [0]: "value"
            items[1]: "another"
        """
        count = 0

        def replacer(m: re.Match) -> str:
            nonlocal count
            count += 1
            return "[IDX]"

        return _PATTERN_INDEX.sub(replacer, text), count

    def _normalize_steps(self, text: str) -> tuple[str, int]:
        """Normalize step/iteration numbers: ``step 1`` → ``step IDX``."""
        count = 0

        def replacer(m: re.Match) -> str:
            nonlocal count
            count += 1
            return f"{m.group(1)} IDX"

        return _PATTERN_STEP.sub(replacer, text), count

    def _normalize_results(self, text: str) -> tuple[str, int]:
        """Normalize result/match numbers: ``result 1`` → ``result IDX``."""
        count = 0

        def replacer(m: re.Match) -> str:
            nonlocal count
            count += 1
            return f"{m.group(1)} IDX"

        return _PATTERN_RESULT.sub(replacer, text), count

    def _normalize_log_lines(self, text: str) -> tuple[str, int]:
        """Normalize log line references: ``[line:42]`` → ``[line:LN]``."""
        count = 0

        def replacer(m: re.Match) -> str:
            nonlocal count
            count += 1
            return m.group(0)[:m.start(1) - m.start(0)] + "LN"

        return _PATTERN_LOG_LINE.sub(replacer, text), count

    def _normalize_rows(self, text: str) -> tuple[str, int]:
        """Normalize row/item numbers: ``Row 1`` → ``Row IDX``."""
        count = 0

        def replacer(m: re.Match) -> str:
            nonlocal count
            count += 1
            return f"{m.group(1)} IDX"

        return _PATTERN_ROW.sub(replacer, text), count
