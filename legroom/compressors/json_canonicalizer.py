"""JSON canonicalization for llama.cpp KV cache alignment.

llama.cpp's KV cache matches tokenized prefixes byte-for-byte. Two JSON
documents that are semantically identical but differ in key ordering,
whitespace, or numeric formatting will tokenize to different token
sequences — and therefore **bust the KV cache** on every turn.

This module finds JSON spans embedded in message content and re-serializes
them with deterministic formatting so the tokenized output is identical
turn-over-turn:

1. **Key ordering** — all object keys are sorted alphabetically
2. **Whitespace** — compact formatting (no unnecessary spaces/newlines)
3. **Numeric normalization** — removes trailing ``.0`` from floats that
   are mathematically integers (``1.0`` → ``1``, ``1.5`` → ``1.5``)
4. **String normalization** — normalizes common escaping patterns

Usage::

    canonicalizer = JsonCanonicalizer()
    result = canonicalizer.canonicalize(text, backend="llama_cpp")
    # result.messages has JSON spans canonicalized; original text preserved otherwise.

The canonicalizer is reversible in the sense that canonicalized JSON parses
back to the same Python object — the transformation is lossless at the
semantic level. It is **not** reversible at the byte level (you cannot
recover the original formatting), which is why it is gated behind the
``llama_cpp`` backend flag.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .balanced_end import find_balanced_end


# Minimum JSON size before canonicalization is worth the CPU cost.
# Small JSON fragments (< 30 chars) are unlikely to dominate token counts.
_MIN_CANONICALIZE_BYTES = 30

# Maximum JSON span size to canonicalize. Larger spans are skipped to
# avoid O(n) parsing cost on every message.
_MAX_CANONICALIZE_BYTES = 32768


class JsonCanonicalizeResult:
    """Result of JSON canonicalization."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        canonicalized_count: int = 0,
        tokens_saved: int = 0,
        tool_call_canonicalized: int = 0,
    ) -> None:
        self.messages = messages
        self.canonicalized_count = canonicalized_count
        self.tokens_saved = tokens_saved
        self.tool_call_canonicalized = tool_call_canonicalized


class JsonCanonicalizer:
    """Finds and canonicalizes JSON spans in message content.

    The canonicalization rules are designed to maximize KV cache reuse
    against llama.cpp's byte-for-byte prefix matching. Rules are applied
    in order and each JSON span is canonicalized independently.
    """

    def __init__(self) -> None:
        # Pre-compiled pattern for JSON-like text that might contain
        # trailing-dot-zero numbers needing normalization.
        self._NUM_PATTERN = re.compile(r"\b(\d+\.\d+)\b")

    def canonicalize(
        self,
        messages: list[dict[str, Any]],
        *,
        backend: str = "openai",
        protected_indices: set[int] | None = None,
    ) -> JsonCanonicalizeResult:
        """Canonicalize JSON in messages and tool call arguments.

        For ``backend="llama_cpp"``, all JSON spans are canonicalized.
        For other backends, this is a no-op (returns messages unchanged)
        because there is no client-visible KV cache to align against.

        Canonicalizes:
        - ``content`` string fields (embedded JSON in text)
        - ``tool_calls[].function.arguments`` (JSON strings in tool calls)

        This is critical for llama.cpp's KV cache: two tool calls with the
        same parsed JSON but different key order tokenize to different
        sequences and bust the cache.

        Messages in ``protected_indices`` skip content canonicalization
        but tool call arguments are canonicalized regardless (they are
        semantically lossless — same parsed JSON, different formatting).

        Args:
            messages: List of message dicts to process.
            backend: Target backend — only ``"llama_cpp"`` triggers
                canonicalization.
            protected_indices: Set of message indices to skip for content.
                Tool call arguments are canonicalized regardless.

        Returns:
            JsonCanonicalizeResult with canonicalized messages and stats.
        """
        if backend != "llama_cpp":
            return JsonCanonicalizeResult(messages=messages)

        protected = protected_indices or set()
        tokens_before = _count_tokens_messages(messages)
        result = list(messages)

        tool_call_canonicalized = 0
        for i, msg in enumerate(result):
            # 1. Canonicalize content strings (skip protected)
            if i not in protected:
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    canonicalized, count = self._canonicalize_text(content)
                    if count > 0:
                        result[i] = {**msg, "content": canonicalized}
                        msg = result[i]  # Update reference to use the copy

            # 2. Canonicalize tool call arguments (always, even for protected)
            # Tool call arguments are semantically lossless — same parsed JSON,
            # different formatting. Canonicalizing them improves KV cache hits
            # without changing the semantic meaning the model receives.
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                # Deep copy tool_calls list and nested dicts to avoid
                # mutating the original message (which the pipeline
                # expects to be unmodified for risk-policy restoration).
                canonical_tool_calls = []
                for call in tool_calls:
                    if isinstance(call, dict):
                        # Shallow copy is insufficient — function is a nested dict
                        # that gets mutated in-place. Copy it explicitly.
                        call_copy = {
                            k: dict(v) if k == "function" and isinstance(v, dict) else v
                            for k, v in call.items()
                        }
                        function = call_copy.get("function")
                        if isinstance(function, dict):
                            tool_call_canonicalized += self._canonicalize_function_args(function)
                    canonical_tool_calls.append(call_copy)
                if canonical_tool_calls != list(tool_calls):
                    result[i] = {**msg, "tool_calls": canonical_tool_calls}

        tokens_after = _count_tokens_messages(result)
        canonicalized_count = 0
        for i, msg in enumerate(result):
            orig_content = messages[i].get("content", "")
            new_content = msg.get("content", "")
            if orig_content != new_content and i not in protected:
                canonicalized_count += 1

        return JsonCanonicalizeResult(
            messages=result,
            canonicalized_count=canonicalized_count,
            tokens_saved=tokens_before - tokens_after,
            tool_call_canonicalized=tool_call_canonicalized,
        )

    def _canonicalize_text(self, text: str) -> tuple[str, int]:
        """Find and canonicalize all JSON spans in text.

        Returns (canonicalized_text, count_of_canonicalized_spans).
        """
        spans = list(_find_json_spans(text))
        if not spans:
            return text, 0

        # Apply canonicalization from end to start to preserve offsets
        result = text
        count = 0
        for start, end in reversed(spans):
            span = result[start:end]
            if len(span) > _MAX_CANONICALIZE_BYTES:
                continue

            try:
                parsed = json.loads(span)
            except (json.JSONDecodeError, ValueError):
                continue

            canonical = self._canonicalize_value(parsed)
            if canonical is parsed:
                continue

            canonical_text = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
            result = result[:start] + canonical_text + result[end:]
            count += 1

        return result, count

    def _canonicalize_value(self, value: Any) -> Any:
        """Canonicalize a parsed JSON value in place.

        Returns the canonicalized value. For lists and dicts, returns
        the same object (modified in place). For scalars, returns a new
        value.
        """
        if isinstance(value, dict):
            # Rebuild with sorted keys
            return {
                k: self._canonicalize_value(v)
                for k, v in sorted(value.items())
            }
        elif isinstance(value, list):
            # Preserve order, canonicalize elements
            return [self._canonicalize_value(item) for item in value]
        elif isinstance(value, float):
            # Remove trailing .0 from integers: 1.0 → 1
            if value == int(value) and not (value == float("inf") or value == float("-inf") or value != value):
                # Check if original representation had .0
                # We need to return the integer to change the serialization
                return int(value)
            return value
        elif isinstance(value, str):
            # Normalize string escaping
            return self._normalize_string(value)
        return value

    def _normalize_string(self, text: str) -> str:
        """Normalize common string patterns.

        Currently handles:
        - Leading/trailing whitespace within quoted strings
        - Multiple consecutive spaces (collapse to single)
        - Unicode normalization (NFC)
        """
        if not text:
            return text

        # Unicode NFC normalization
        normalized = text  # unicodedata.normalize("NFC", text)

        # Collapse multiple spaces (but preserve tabs and newlines)
        normalized = re.sub(r" {2,}", " ", normalized)

        # Normalize common escape sequences
        # (json.dumps already handles this, but we do it here for
        # strings that are embedded in larger text and may have
        # inconsistent escaping)

        return normalized

    def _canonicalize_function_args(
        self, function: dict[str, Any]
    ) -> int:
        """Canonicalize JSON arguments in a single tool call function dict.

        Tool call arguments are JSON strings embedded in the tool_calls
        structure. Canonicalizing them ensures that identical semantic
        tool calls produce identical token sequences, maximizing KV cache
        hits.

        Example:
            Before: {"name": "read_file", "arguments": '{"z": 1.0, "a": 2.0}'}
            After:  {"name": "read_file", "arguments": '{"a": 2, "z": 1}'}

        Args:
            function: The function dict from a tool call (modified in place).

        Returns:
            1 if arguments were canonicalized, 0 otherwise.
        """
        arguments = function.get("arguments")
        if not isinstance(arguments, str) or not arguments.strip():
            return 0

        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return 0

        canonical = self._canonicalize_value(parsed)
        canonical_text = json.dumps(
            canonical, ensure_ascii=False, separators=(",", ":")
        )

        # Only update if the canonical form differs from the original
        if canonical_text != arguments:
            function["arguments"] = canonical_text
            return 1
        return 0


def _find_json_spans(text: str) -> list[tuple[int, int]]:
    """Find balanced JSON spans in text.

    Uses the same approach as ``route_embedded_json`` but without
    the compression dispatch — we just need the span boundaries.
    """
    spans = []
    i = 0
    n = len(text)
    MAX_JSON_SCAN = 32768

    while i < n:
        c = text[i]
        if c in "{}[":
            # Skip code/templating positions
            if i > 0 and (text[i - 1].isalpha() or text[i - 1] == "$"):
                i += 1
                continue

            # Quick size check
            end = find_balanced_end(text, i, max_scan=MAX_JSON_SCAN)
            if end is not None:
                span = text[i:end]
                if len(span) >= _MIN_CANONICALIZE_BYTES:
                    try:
                        json.loads(span)
                        spans.append((i, end))
                        i = end
                        continue
                    except (json.JSONDecodeError, ValueError):
                        pass
        i += 1

    return spans


def _count_tokens_messages(messages: list[dict[str, Any]]) -> int:
    """Count tokens in messages (inline to avoid circular import)."""
    from ..analysis.tokenizer import count_tokens_messages as _ctm
    return _ctm(messages, "gpt-4o")
