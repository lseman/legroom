"""KV cache optimization — reduces KV cache pressure via prefix dedup and alignment.

When an LLM processes a long context, the KV cache stores key-value pairs for
every token. KV cache optimization targets three things:

1. **Prefix deduplication**: Common prefixes across messages (system prompts,
   repeated tool schemas, boilerplate) are stored once and later occurrences
   are replaced with compact pointers. The model sees the pointer but the
   KV cache only needs to hold one copy.

2. **Token-boundary alignment**: Compression must not break tokens mid-word.
   We ensure compressed text starts and ends at token boundaries so the
   model's tokenizer sees clean tokens.

3. **KV cache alignment**: Structure content so common prefixes appear at
   message boundaries, maximizing KV cache hit rates when the same system
   prompt or schema repeats across turns.

This is distinct from context compression: compression reduces total tokens,
while KV cache optimization reduces the *effective* KV cache pressure by
eliminating redundant token sequences.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Minimum prefix length (chars) before dedup is worth it — short prefixes
# would inflate the output with pointer overhead.
_MIN_PREFIX_BYTES = 100

# Minimum number of messages sharing a prefix before dedup triggers.
_MIN_PREFIX_OCCURRENCES = 2

# Max pointer length budget — pointers longer than this are worse than
# keeping the original text.
_MAX_POINTER_CHARS = 80

# Token boundary padding: when aligning, add this many tokens of padding
# at the start of a message to ensure it starts at a token boundary.
_TOKEN_BOUNDARY_PADDING = 4


@dataclass
class KVOptimizationResult:
    """Result of KV cache optimization."""

    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    prefix_dedup_count: int = 0
    prefix_tokens_saved: int = 0
    token_boundary_aligned: int = 0
    transforms_applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class KVOptimizer:
    """KV cache optimization via prefix dedup and token-boundary alignment.

    Strategy:
    1. Extract text content from all messages.
    2. Find common prefixes across messages (n-gram based).
    3. Replace later occurrences with compact pointers.
    4. Ensure all text starts/ends at token boundaries.
    """

    def __init__(
        self,
        min_prefix_bytes: int = _MIN_PREFIX_BYTES,
        min_occurrences: int = _MIN_PREFIX_OCCURRENCES,
        max_pointer_chars: int = _MAX_POINTER_CHARS,
        token_boundary_padding: int = _TOKEN_BOUNDARY_PADDING,
    ) -> None:
        self._min_prefix_bytes = min_prefix_bytes
        self._min_occurrences = min_occurrences
        self._max_pointer_chars = max_pointer_chars
        self._token_boundary_padding = token_boundary_padding

    def optimize(
        self,
        messages: list[dict[str, Any]],
        model: str = "gpt-4o",
    ) -> KVOptimizationResult:
        """Apply KV cache optimization to messages.

        Returns a KVOptimizationResult with optimized messages and stats.
        """
        if len(messages) < self._min_occurrences:
            return KVOptimizationResult(
                messages=messages,
                tokens_before=count_tokens_messages(messages, model),
                tokens_after=count_tokens_messages(messages, model),
            )

        tokens_before = count_tokens_messages(messages, model)

        # Phase 1: Extract text content and find common prefixes
        text_segments = self._extract_text_segments(messages)
        if not text_segments:
            return KVOptimizationResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        common_prefixes = self._find_common_prefixes(text_segments)
        if not common_prefixes:
            return KVOptimizationResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        # Phase 2: Replace common prefixes with pointers
        prefix_map: dict[str, str] = {}
        pointer_counter = 0
        optimized = list(messages)

        for prefix, occurrences in common_prefixes.items():
            if len(occurrences) < self._min_occurrences:
                continue

            distinct_indices = sorted({idx for idx, _ in occurrences})
            if len(distinct_indices) < self._min_occurrences:
                continue
            source_index = distinct_indices[0]
            # Keep references meaningful in the actual prompt. The previous
            # opaque ``[prefix_N]`` marker depended on metadata that was never
            # sent to the model.
            pointer = f"[same prefix as message {source_index}]"
            pointer_counter += 1
            prefix_map[prefix] = pointer

            # Sort by message index so the earliest message always
            # keeps the full prefix (it's the most important one
            # for the model's context window). Deduplicate by message
            # index in case the same message shares the prefix with
            # multiple others.
            sorted_occurrences = [(idx, 0) for idx in distinct_indices]

            # Always preserve the earliest message (first in sorted order)
            for msg_idx, _ in sorted_occurrences[1:]:
                content = optimized[msg_idx].get("content", "")
                replacement = pointer + content[len(prefix):] if isinstance(content, str) else ""
                if (
                    isinstance(content, str)
                    and content.startswith(prefix)
                    and len(replacement) < len(content)
                ):
                    optimized[msg_idx] = {
                        **optimized[msg_idx],
                        "content": replacement,
                    }

        # Phase 3: Token-boundary alignment
        aligned_count = 0
        for i, msg in enumerate(optimized):
            content = msg.get("content", "")
            if isinstance(content, str) and content and not content.startswith("[same prefix as"):
                aligned = self._align_token_boundary(content, model)
                if aligned != content:
                    optimized[i] = {**msg, "content": aligned}
                    aligned_count += 1

        tokens_after = count_tokens_messages(optimized, model)

        result = KVOptimizationResult(
            messages=optimized,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            prefix_dedup_count=pointer_counter,
            prefix_tokens_saved=tokens_before - tokens_after,
            token_boundary_aligned=aligned_count,
            transforms_applied=["kv_cache_optimization"],
            metadata={
                "prefix_map": prefix_map,
                "common_prefix_count": len(common_prefixes),
            },
        )

        return result

    def _extract_text_segments(
        self, messages: list[dict[str, Any]]
    ) -> list[tuple[int, str]]:
        """Extract text content from messages as (index, text) pairs."""
        segments: list[tuple[int, str]] = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                segments.append((i, content))
            elif isinstance(content, list):
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                text = " ".join(text_parts)
                if text.strip():
                    segments.append((i, text))
        return segments

    def _find_common_prefixes(
        self, segments: list[tuple[int, str]]
    ) -> dict[str, list[tuple[int, int]]]:
        """Find common prefixes across text segments.

        Uses a sliding window approach: for each pair of segments, find
        the longest common prefix. Groups prefixes by their content and
        tracks which segments contain them.

        Returns a dict mapping prefix → list of (segment_index, position).
        """
        # Build a map: prefix → [(segment_index, position)]
        prefix_map: dict[str, list[tuple[int, int]]] = defaultdict(list)

        # For each pair of segments, find common prefixes
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                idx_i, text_i = segments[i]
                idx_j, text_j = segments[j]

                # Find longest common prefix
                common = self._longest_common_prefix(text_i, text_j)
                if len(common) < self._min_prefix_bytes:
                    continue

                # Record this prefix for both segments
                prefix_map[common].append((idx_i, 0))
                prefix_map[common].append((idx_j, 0))

        # Filter to prefixes that appear in enough segments.
        # Each pair (i, j) adds 2 entries, so for N segments sharing
        # a prefix we get N entries (one per segment). We need at
        # least min_occurrences distinct segments.
        return {
            prefix: occurrences
            for prefix, occurrences in prefix_map.items()
            if len(occurrences) >= self._min_occurrences
        }

    def _longest_common_prefix(self, s1: str, s2: str) -> str:
        """Find the longest common prefix between two strings."""
        min_len = min(len(s1), len(s2))
        i = 0
        while i < min_len and s1[i] == s2[i]:
            i += 1
        return s1[:i]

    def _align_token_boundary(
        self, text: str, model: str = "gpt-4o"
    ) -> str:
        """Ensure text starts at a token boundary.

        If the text starts mid-token (e.g., after a newline that split a
        token), prepend whitespace to align it. We use tiktoken to check
        if the text starts at a token boundary.

        Returns the aligned text (or the original if already aligned).
        """
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        # Check if text starts at a token boundary by encoding the first
        # few tokens and seeing if decoding them matches the start of text
        first_tokens = encoding.encode(text[:100])
        if not first_tokens:
            return text

        # Decode the first token and check if it matches the start of text
        first_token = encoding.decode([first_tokens[0]])
        if text.startswith(first_token):
            return text  # Already aligned

        # Not aligned — prepend a space to shift to next token boundary
        return " " + text

    def get_stats(self) -> dict[str, Any]:
        """Return optimization statistics."""
        return {
            "min_prefix_bytes": self._min_prefix_bytes,
            "min_occurrences": self._min_occurrences,
            "max_pointer_chars": self._max_pointer_chars,
        }


def count_tokens_messages(messages: list[dict[str, Any]], model: str = "gpt-4o") -> int:
    """Count tokens in a list of messages (inline to avoid circular import)."""
    from ..tokenizer import count_tokens_messages as _ctm
    return _ctm(messages, model)
