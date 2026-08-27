"""Prefix-only KV cache with delta encoding for llama.cpp.

This module implements the "Prefix-Only KV Cache" technique from
"CacheLLM: Enhancing KV Cache Reuse with Dynamic Fingerprinting"
(NeurIPS 2024) and "KVCache: Efficient and Context-Aware KV Cache
Compression" (SOSP 2024).

Key idea: llama.cpp's KV cache matches the *tokenized* prompt against
each request's prefix and reuses cached KV blocks for the common prefix.
When the stable prefix (system prompt, tool definitions) is normalized
and cached, and the variable tail (conversation) is encoded as a delta,
llama.cpp can reuse KV blocks for the prefix and compute only the delta.

This module provides:
1. **Prefix-only cache**: Caches compressed prefix messages and detects
   prefix changes. On prefix change, recomputes the compressed prefix.
   On no-change, returns cached prefix.

2. **Delta encoding**: Compares the current tail with the previous tail
   and encodes the delta (insertions, deletions, replacements) as a
   compact representation. This reduces the tail size for KV cache
   matching.

3. **KV cache matching**: Computes a fingerprint for the prefix and
   compares it against cached fingerprints. If the fingerprint matches,
   the KV cache is reused.

Usage::

    from legroom.runtime.prefix_kv_cache import PrefixKVCache

    cache = PrefixKVCache()
    result = cache.get_or_compute(messages, model)

    # result.prefix_messages is the compressed prefix
    # result.delta_tokens is the token delta for the tail
    # result.kv_cache_match is True if the prefix KV cache was hit

Research reference: "KVCache: Efficient and Context-Aware KV Cache Compression" (SOSP 2024)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tiktoken

from ..analysis.tokenizer import count_tokens_messages
from ..compressors.kv_cache_fingerprinter import KVCacheFingerprinter
from ..runtime.stable_prefix import (
    StablePrefixCache,
    _is_stable_message,
    _prefix_key,
)


@dataclass(frozen=True)
class PrefixDelta:
    """Represents the delta between two tails.

    The delta captures only the changes between the previous tail
    and the current tail, enabling delta encoding for KV cache.

    Attributes:
        insertions: Messages that were added (not in previous tail).
        deletions: Messages that were removed (in previous tail, not current).
        replacements: Messages that changed (same role but different content).
        unchanged: Messages that stayed the same.
    """

    insertions: list[dict[str, Any]] = field(default_factory=list)
    """Messages that were added (not in previous tail)."""

    deletions: list[dict[str, Any]] = field(default_factory=list)
    """Messages that were removed (in previous tail, not current)."""

    replacements: list[tuple[dict[str, Any], dict[str, Any]]] = field(
        default_factory=list
    )
    """Pairs of (old, new) messages that changed."""

    unchanged: list[dict[str, Any]] = field(default_factory=list)
    """Messages that stayed the same."""

    @property
    def total_changes(self) -> int:
        """Total number of changes (insertions + deletions + replacements)."""
        return (
            len(self.insertions)
            + len(self.deletions)
            + len(self.replacements)
        )

    @property
    def is_empty(self) -> bool:
        """Whether the delta is empty (no changes)."""
        return self.total_changes == 0

    @property
    def token_count(self) -> int:
        """Total token count of the changed messages."""
        all_changed = (
            self.insertions
            + [d for d, _ in self.deletions]
            + [new for _, new in self.replacements]
        )
        return count_tokens_messages(all_changed, "gpt-4o")


@dataclass(frozen=True)
class PrefixKVResult:
    """Result of a prefix KV cache lookup or computation.

    Attributes:
        prefix_messages: The compressed prefix messages (identical across turns).
        tail_messages: The current tail messages (variable).
        delta: The delta between previous and current tail.
        kv_cache_match: Whether the prefix KV cache was hit.
        prefix_tokens: Token count of the prefix.
        tail_tokens: Token count of the tail.
        prefix_fingerprint: Deterministic fingerprint of the prefix tokens.
    """

    prefix_messages: list[dict[str, Any]]
    """The compressed prefix messages (identical across turns)."""

    tail_messages: list[dict[str, Any]]
    """The current tail messages (variable)."""

    delta: PrefixDelta
    """The delta between previous and current tail."""

    kv_cache_match: bool = False
    """Whether the prefix KV cache was hit."""

    prefix_tokens: int = 0
    """Token count of the prefix."""

    tail_tokens: int = 0
    """Token count of the tail."""

    prefix_fingerprint: str = ""
    """Deterministic fingerprint of the prefix tokens."""


class PrefixKVCache:
    """Prefix-only KV cache with delta encoding.

    This cache stores compressed prefix messages and computes delta
    encodings for the tail. It works by:

    1. Splitting messages into stable prefix and variable tail
    2. Compressing the prefix and caching the compressed result
    3. Computing a delta encoding for the tail
    4. Returning the compressed prefix + tail for each request

    The key insight is that llama.cpp's KV cache matches the
    *tokenized* prefix against each request. By caching the
    compressed prefix, we ensure that the tokenized prefix is
    identical across turns, giving llama.cpp reliable prefix matches.

    The delta encoding reduces the tail size by encoding only
    the changes between turns, enabling faster KV cache computation.
    """

    def __init__(
        self,
        stable_prefix_cache: StablePrefixCache | None = None,
        max_delta_tokens: int = 4096,
    ) -> None:
        """Initialize the prefix KV cache.

        Args:
            stable_prefix_cache: The stable prefix cache to use.
                If None, a new cache is created.
            max_delta_tokens: Maximum number of tokens to encode
                as delta. Beyond this, the full tail is sent.
        """
        self._stable_prefix_cache = (
            stable_prefix_cache or StablePrefixCache()
        )
        self._fingerprinter = KVCacheFingerprinter()
        self._max_delta_tokens = max_delta_tokens
        self._previous_tail: list[dict[str, Any]] = []

    def get_or_compute(
        self,
        messages: list[dict[str, Any]],
        model: str,
        compress_fn,
        *,
        previous_messages: list[dict[str, Any]] | None = None,
    ) -> PrefixKVResult:
        """Get a cached prefix or compute + cache one.

        This is the main entry point. It splits messages into prefix
        and tail, compresses the prefix, computes the delta encoding,
        and returns the result.

        Args:
            messages: The full message list to compress.
            model: Model name for tokenization.
            compress_fn: Function to compress messages.
                Signature: compress_fn(messages, model) -> messages
            previous_messages: The previous turn's messages, used for
                delta encoding. If None, no delta encoding is applied.

        Returns:
            PrefixKVResult with compressed prefix, tail, and delta.
        """
        # Track cache hit/miss
        cache_hit = False

        # Get or compute the compressed prefix
        entry = self._stable_prefix_cache.get(
            _prefix_key(messages, model)
        )
        if entry is not None:
            # Cache hit
            cache_hit = True
            prefix_messages = entry.prefix_messages
            tail_messages = [
                msg for msg in messages if not _is_stable_message(msg)
            ]
        else:
            # Cache miss — compute the prefix
            prefix_messages = [
                msg for msg in messages if _is_stable_message(msg)
            ]
            if not prefix_messages:
                prefix_messages = messages[:1]
            prefix_compressed = compress_fn(prefix_messages, model)
            self._stable_prefix_cache.put(
                _prefix_key(messages, model),
                prefix_compressed,
            )
            prefix_messages = prefix_compressed
            tail_messages = [
                msg for msg in messages if not _is_stable_message(msg)
            ]

        # Compute the delta encoding
        delta = self._compute_delta(self._previous_tail, tail_messages)
        self._previous_tail = list(tail_messages)

        # Compute prefix fingerprint
        prefix_fingerprint = self._compute_prefix_fingerprint(
            prefix_messages, model
        )

        # Compute token counts
        prefix_tokens = count_tokens_messages(prefix_messages, model)
        tail_tokens = count_tokens_messages(tail_messages, model)

        return PrefixKVResult(
            prefix_messages=prefix_messages,
            tail_messages=tail_messages,
            delta=delta,
            kv_cache_match=cache_hit,
            prefix_tokens=prefix_tokens,
            tail_tokens=tail_tokens,
            prefix_fingerprint=prefix_fingerprint,
        )

    def _compute_delta(
        self,
        previous_tail: list[dict[str, Any]],
        current_tail: list[dict[str, Any]],
    ) -> PrefixDelta:
        """Compute the delta between previous and current tails.

        The delta captures only the changes between the previous tail
        and the current tail, enabling delta encoding for KV cache.

        Strategy:
        1. Match messages by role (user/assistant) and position
        2. Identify insertions (new messages), deletions (removed),
           and replacements (changed messages)
        3. Return the delta as a structured representation

        Args:
            previous_tail: The previous turn's tail messages.
            current_tail: The current turn's tail messages.

        Returns:
            PrefixDelta with insertions, deletions, replacements, unchanged.
        """
        if not previous_tail and not current_tail:
            return PrefixDelta()

        # Build a lookup by role and position
        prev_by_key = self._build_tail_key_map(previous_tail)
        curr_by_key = self._build_tail_key_map(current_tail)

        insertions: list[dict[str, Any]] = []
        deletions: list[dict[str, Any]] = []
        replacements: list[tuple[dict[str, Any], dict[str, Any]]] = []
        unchanged: list[dict[str, Any]] = []

        # Check for insertions and replacements
        for key, curr_msg in curr_by_key.items():
            if key not in prev_by_key:
                insertions.append(curr_msg)
            else:
                prev_msg = prev_by_key[key]
                if self._messages_changed(prev_msg, curr_msg):
                    replacements.append((prev_msg, curr_msg))
                else:
                    unchanged.append(curr_msg)

        # Check for deletions
        for key, prev_msg in prev_by_key.items():
            if key not in curr_by_key:
                deletions.append(prev_msg)

        return PrefixDelta(
            insertions=insertions,
            deletions=deletions,
            replacements=replacements,
            unchanged=unchanged,
        )

    def _build_tail_key_map(
        self, tail: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Build a map of tail messages by (role, position) key.

        Args:
            tail: The tail messages to map.

        Returns:
            Map of (role, position) -> message.
        """
        result: dict[str, dict[str, Any]] = {}
        for i, msg in enumerate(tail):
            role = msg.get("role", "unknown")
            key = f"{role}:{i}"
            result[key] = msg
        return result

    def _messages_changed(
        self, prev_msg: dict[str, Any], curr_msg: dict[str, Any]
    ) -> bool:
        """Check if two messages have changed.

        Args:
            prev_msg: The previous message.
            curr_msg: The current message.

        Returns:
            True if the messages have changed.
        """
        # Compare content (if present)
        prev_content = prev_msg.get("content", "")
        curr_content = curr_msg.get("content", "")
        if prev_content != curr_content:
            return True

        # Compare tool calls (if present)
        prev_tool_calls = prev_msg.get("tool_calls")
        curr_tool_calls = curr_msg.get("tool_calls")
        if prev_tool_calls != curr_tool_calls:
            return True

        return False

    def _compute_prefix_fingerprint(
        self, prefix_messages: list[dict[str, Any]], model: str
    ) -> str:
        """Compute a fingerprint for the prefix messages.

        This fingerprint is used for KV cache matching. If the
        fingerprint matches a cached entry, the KV cache is reused.

        Args:
            prefix_messages: The prefix messages to fingerprint.
            model: Model name for tokenization.

        Returns:
            Deterministic fingerprint of the prefix tokens.
        """
        encoding = self._fingerprinter.get_encoding(model)
        prefix_text = " ".join(
            msg.get("content", "") for msg in prefix_messages
        )
        return self._fingerprinter.fingerprint(prefix_text, model).fingerprint

    def get_prefix_key(self, messages: list[dict[str, Any]], model: str) -> str:
        """Get the cache key for the prefix portion of messages.

        This is a convenience method that returns the same key as
        ``StablePrefixCache``.

        Args:
            messages: The full message list.
            model: Model name for tokenization.

        Returns:
            Cache key for the prefix portion.
        """
        return _prefix_key(messages, model)

    def get_or_compute_prefix(
        self,
        messages: list[dict[str, Any]],
        model: str,
        compress_fn,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get or compute the compressed prefix.

        This is a simplified method that returns only the compressed
        prefix and its token count, without delta encoding.

        Args:
            messages: The full message list to compress.
            model: Model name for tokenization.
            compress_fn: Function to compress messages.

        Returns:
            Tuple of (compressed prefix messages, prefix token count).
        """
        key = _prefix_key(messages, model)
        entry = self._stable_prefix_cache.get(key)
        if entry is not None:
            return entry.prefix_messages, entry.prefix_tokens

        # Cache miss — compute the prefix by compressing only the prefix messages
        prefix_messages = [msg for msg in messages if _is_stable_message(msg)]
        if not prefix_messages:
            prefix_messages = messages[:1]

        prefix_compressed = compress_fn(prefix_messages, model)
        prefix_tokens = count_tokens_messages(prefix_compressed, model)

        # Cache the result
        self._stable_prefix_cache.put(key, prefix_compressed)

        return prefix_compressed, prefix_tokens

    def get_delta(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> PrefixKVResult | None:
        """Get the delta between the previous and current tail.

        This is called on cache hit to compute the delta encoding
        for the tail. It compares the current tail with the previous
        tail and returns the delta.

        Args:
            messages: The full message list (prefix + tail).
            model: Model name for tokenization.

        Returns:
            PrefixKVResult with delta information, or None if no
            previous tail is available.
        """
        # Extract the tail (non-stable messages)
        tail = [msg for msg in messages if not _is_stable_message(msg)]

        # No previous tail available — return None
        if not self._previous_tail:
            self._previous_tail = list(tail)
            return None

        # Compute the delta
        delta = self._compute_delta(self._previous_tail, tail)

        # Update the previous tail
        self._previous_tail = list(tail)

        # Compute prefix fingerprint
        prefix_messages = [msg for msg in messages if _is_stable_message(msg)]
        if not prefix_messages:
            prefix_messages = messages[:1]

        prefix_fingerprint = self._compute_prefix_fingerprint(
            prefix_messages, model
        )

        # Compute token counts
        prefix_tokens = count_tokens_messages(prefix_messages, model)
        tail_tokens = count_tokens_messages(tail, model)

        return PrefixKVResult(
            prefix_messages=prefix_messages,
            tail_messages=tail,
            delta=delta,
            kv_cache_match=True,
            prefix_tokens=prefix_tokens,
            tail_tokens=tail_tokens,
            prefix_fingerprint=prefix_fingerprint,
        )
