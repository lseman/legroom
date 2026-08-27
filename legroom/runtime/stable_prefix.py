"""Stable-prefix compression cache for llama.cpp KV-cache alignment.

llama.cpp's OpenAI-compatible server matches the *tokenized* prompt against
each request's prefix and reuses cached KV blocks for the common prefix.
When the proxy compresses the *entire* message sequence as one batch,
the compressed output of the stable system prompt varies depending on what
the conversation tail looks like — different surrounding context triggers
different compression decisions, so the tokenized prefix is not byte-identical
turn-over-turn, defeating the server-side KV cache.

This module solves that by decomposing the prompt into two parts:

1. **Stable prefix** — system prompt, tool definitions, instructions. This
   is the part that repeats verbatim across turns. It is compressed *once*
   and cached so every request that shares the same prefix gets the same
   compressed prefix messages.

2. **Conversation tail** — user messages and assistant responses. This part
   varies each turn and is compressed independently.

When the stable prefix is unchanged, the cache returns the pre-compressed
prefix messages so the tokenized prefix is identical turn-over-turn,
giving llama.cpp a reliable prefix match and avoiding full reprocessing.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ..analysis.tokenizer import count_tokens_messages

logger = logging.getLogger(__name__)


# Maximum cache size — enough to hold one entry per distinct model + tool
# configuration combination without exhausting memory.
_DEFAULT_MAX_ENTRIES = 64


def _is_stable_message(msg: dict[str, Any]) -> bool:
    """Return True if this message is part of the stable prefix.

    A message is considered stable if it's a system message or a tool
    definition (tool schema). Tool **results** (with tool_call_id) are
    conversation state that varies every turn and must NOT be stable.

    The conversation tail starts at the first user or assistant message.
    """
    role = msg.get("role", "")
    if role == "system":
        return True
    # Tool definitions have an 'id' field (the tool call ID) and a
    # 'function' dict. Tool **results** have a 'tool_call_id' field
    # (referring to a prior assistant tool call). Only definitions
    # are stable — results change every turn.
    if role == "tool" or role == "function":
        # Tool definitions have 'id' + 'function'; results have 'tool_call_id'
        has_definition = "id" in msg and "function" in msg
        has_result = "tool_call_id" in msg
        if has_result:
            return False
        return has_definition
    return False


def _prefix_key(messages: list[dict[str, Any]], model: str) -> str:
    """Compute a stable key for the prefix portion of the messages.

    The key includes the model and every byte of the prefix content so that
    different models or different system prompts always produce independent
    cache entries.
    """
    import json

    prefix = [msg for msg in messages if _is_stable_message(msg)]
    if not prefix:
        prefix = messages[:1]  # Fallback: at least the first message
    document = json.dumps(
        [model] + prefix,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return "spc:" + hashlib.sha256(document).hexdigest()[:32]


@dataclass(frozen=True)
class _PrefixEntry:
    """A cached compressed prefix.

    Stores the *compressed* prefix messages and the *original* tail
    messages (the conversation portion after the prefix). On a cache
    hit, the pipeline reconstructs the full prompt as
    ``compressed_prefix + original_tail``, letting the tail be
    compressed by the normal pipeline while the prefix stays
    byte-identical across requests.
    """
    key: str
    prefix_messages: list[dict[str, Any]]
    tail_messages: list[dict[str, Any]]
    prefix_tokens: int


@dataclass
class PrefixCacheMetrics:
    """Metrics for stable-prefix cache activity."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate * 100, 1),
            "evictions": self.evictions,
        }


class StablePrefixCache:
    """LRU cache for compressed stable prefixes.

    Keys are derived from the model name + all prefix messages (system,
    tools, instructions). Values are the *compressed* prefix messages
    — the exact output that would come from running the compression
    pipeline on just the prefix.

    Usage pattern:
        cache = StablePrefixCache()
        prefix_entry = cache.get_or_compute(messages, model, compress_fn)
        # prefix_entry.prefix_messages is the pre-compressed prefix
        # prefix_entry.prefix_tokens is the token count
    """

    def __init__(self, maxsize: int = _DEFAULT_MAX_ENTRIES) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self._entries: OrderedDict[str, _PrefixEntry] = OrderedDict()
        self._maxsize = maxsize
        self._metrics = PrefixCacheMetrics()

    def get(self, key: str) -> _PrefixEntry | None:
        """Return a cached prefix entry or None."""
        entry = self._entries.get(key)
        if entry is None:
            self._metrics.misses += 1
            return None
        self._entries.move_to_end(key)
        self._metrics.hits += 1
        return entry

    def key_for_messages(
        self, messages: list[dict[str, Any]], model: str
    ) -> str:
        """Compute the cache key for a message list without storing it.

        Convenience method for callers that need the key before deciding
        whether to store a result. Equivalent to ``_prefix_key`` but
        exposed as a public method on the cache instance.
        """
        return _prefix_key(messages, model)

    def put(
        self,
        key: str,
        prefix_messages: list[dict[str, Any]],
        tail_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Cache a compressed prefix entry."""
        prefix_tokens = count_tokens_messages(prefix_messages, "gpt-4o")
        entry = _PrefixEntry(
            key=key,
            prefix_messages=prefix_messages,
            tail_messages=tail_messages or [],
            prefix_tokens=prefix_tokens,
        )
        # Check if key already exists (update)
        if key in self._entries:
            self._entries.move_to_end(key)
            self._entries[key] = entry
        else:
            self._entries[key] = entry
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)
                self._metrics.evictions += 1

    def get_or_compute(
        self,
        messages: list[dict[str, Any]],
        model: str,
        compress_fn,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Get a cached prefix or compute + cache one.

        Returns (prefix_messages, tail_messages) where prefix_messages is
        the pre-compressed stable prefix and tail_messages is the original
        conversation tail (to be compressed separately).
        """
        key = _prefix_key(messages, model)
        entry = self.get(key)
        if entry is not None:
            # Return cached prefix + original tail
            tail = [msg for msg in messages if not _is_stable_message(msg)]
            return deepcopy(entry.prefix_messages), tail

        # Cache miss — compute the prefix by compressing only the prefix messages
        prefix_messages = [msg for msg in messages if _is_stable_message(msg)]
        # Empty prefix? Use first message
        if not prefix_messages:
            prefix_messages = messages[:1]

        prefix_compressed = compress_fn(prefix_messages, model)

        # Split original messages
        tail = [msg for msg in messages if not _is_stable_message(msg)]

        self.put(key, prefix_compressed, tail)
        return prefix_compressed, tail

    @property
    def metrics(self) -> PrefixCacheMetrics:
        return self._metrics

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._metrics = PrefixCacheMetrics()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._metrics.to_dict(),
            "size": self.size,
            "maxsize": self._maxsize,
        }
