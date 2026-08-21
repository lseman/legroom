"""Token counting utilities with LRU caching.

Non-OpenAI models (e.g. Claude) don't have a public local tokenizer, so
their counts here are a cl100k_base approximation, not an exact count.
That's good enough for before/after compression ratios (both sides use
the same approximation) but not for billing-accurate token counts — for
those, use the provider's own counting API (e.g. Anthropic's
``client.messages.count_tokens``).
"""

from __future__ import annotations

import hashlib
from typing import Any

import tiktoken

_MODEL_TO_ENCODING: dict[str, str] = {
    "gpt-4o": "cl100k_base",
    "gpt-4o-mini": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    # Approximated via cl100k_base — see module docstring.
    "claude-3-5-sonnet": "cl100k_base",
    "claude-3-opus": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
}


class _TokenCache:
    """LRU cache for token counts keyed by (model, text hash)."""

    def __init__(self, maxsize: int = 512) -> None:
        self._cache: dict[str, int] = {}
        self._order: list[str] = []
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> int | None:
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: int) -> None:
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self._maxsize:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)
        self._cache[key] = value
        self._order.append(key)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def ratio(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()


# Module-level singleton cache
token_cache = _TokenCache()


def get_encoding(model: str) -> tiktoken.Encoding:
    """Get the encoding for a model."""
    encoding_name = _MODEL_TO_ENCODING.get(model, "cl100k_base")
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text with LRU caching.

    Cached by (model, sha256(text)) so the same text+model pair
    avoids re-encoding on repeated calls.
    Uses tiktoken for accurate counting instead of the rough
    ``len(text) // 4`` heuristic.
    """
    if not text:
        return 0
    key = f"{model}:{hashlib.sha256(text.encode()).hexdigest()[:24]}"
    cached = token_cache.get(key)
    if cached is not None:
        return cached
    encoding = get_encoding(model)
    result = len(encoding.encode(text))
    token_cache.put(key, result)
    return result


def count_tokens_messages(messages: list[dict[str, Any]], model: str = "gpt-4o") -> int:
    """Count tokens in a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += count_tokens(block.get("text", ""), model)
    return total
