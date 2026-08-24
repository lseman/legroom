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
import json
from collections import OrderedDict
from threading import RLock
from typing import Any, Literal

# Use MD5 instead of SHA256 for cache keys — fast enough for token
# counting (collision risk is negligible for this use case) and ~2x faster.
_hash_func = hashlib.md5

import tiktoken

_MODEL_TO_ENCODING: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    # Approximated via cl100k_base — see module docstring.
    "claude-3-5-sonnet": "cl100k_base",
    "claude-3-opus": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
}

# Pre-loaded encodings to avoid repeated registry lookups.
_MODEL_TO_ENCODING_OBJ: dict[str, tiktoken.Encoding] = {}


class _TokenCache:
    """LRU cache for token counts keyed by (model, text hash).

    Uses OrderedDict for O(1) removal (vs O(n) with a list) since
    in a real conversation the cache sees many unique texts and
    eviction is frequent. Increased to 2048 for proxy mode where
    tool outputs repeat frequently.
    """

    def __init__(self, maxsize: int = 2048) -> None:
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    def get(self, key: str) -> int | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: int) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

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
        with self._lock:
            self._cache.clear()


# Module-level singleton cache
token_cache = _TokenCache()


def get_encoding(model: str) -> tiktoken.Encoding:
    """Get the encoding for a model."""
    if model in _MODEL_TO_ENCODING_OBJ:
        return _MODEL_TO_ENCODING_OBJ[model]
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding_name = _MODEL_TO_ENCODING.get(model, "cl100k_base")
        enc = tiktoken.get_encoding(encoding_name)
    _MODEL_TO_ENCODING_OBJ[model] = enc
    return enc


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text with LRU caching.

    Cached by (model, md5(text)) so the same text+model pair
    avoids re-encoding on repeated calls. Uses tiktoken for accurate
    counting instead of the rough ``len(text) // 4`` heuristic.
    Non-string content (e.g. Anthropic content blocks) returns 0;
    use ``count_tokens_messages`` which handles list content.

    Fast paths:
      - Empty/None text → 0 (no hashing overhead)
      - Short text (< 50 chars) → direct encode, no cache (overhead
        of hashing + cache ops exceeds the encode cost)
    """
    if not isinstance(text, str) or not text:
        return 0
    # Short text: hashing + cache overhead exceeds encode cost.
    if len(text) < 50:
        return len(get_encoding(model).encode(text))
    key = f"{model}:{_hash_func(text.encode()).hexdigest()[:16]}"
    cached = token_cache.get(key)
    if cached is not None:
        return cached
    encoding = get_encoding(model)
    result = len(encoding.encode(text))
    token_cache.put(key, result)
    return result


TokenProtocol = Literal["auto", "openai_chat", "openai_responses", "content_only"]


def _detect_protocol(messages: list[dict[str, Any]]) -> TokenProtocol:
    if any(message.get("type") == "message" for message in messages):
        return "openai_responses"
    return "openai_chat"


def count_tokens_messages(
    messages: list[dict[str, Any]],
    model: str = "gpt-4o",
    *,
    protocol: TokenProtocol = "auto",
) -> int:
    """Estimate complete request-message tokens for a provider protocol.

    The estimator counts framing, roles, structured content, tool calls, and
    tool-call identifiers. It remains an estimate because providers may apply
    private serialization and image/audio tokenization rules.
    """
    resolved_protocol = _detect_protocol(messages) if protocol == "auto" else protocol
    framing_per_message = 3 if resolved_protocol in {"openai_chat", "openai_responses"} else 0
    total = 3 if framing_per_message and messages else 0
    for msg in messages:
        total += framing_per_message
        role = msg.get("role")
        if isinstance(role, str) and resolved_protocol != "content_only":
            total += count_tokens(role, model)
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    total += count_tokens(block["text"], model)
                    remainder = {key: value for key, value in block.items() if key != "text"}
                    if remainder and resolved_protocol != "content_only":
                        total += count_tokens(
                            json.dumps(remainder, sort_keys=True, separators=(",", ":")), model
                        )
                elif resolved_protocol != "content_only":
                    total += count_tokens(
                        json.dumps(block, sort_keys=True, separators=(",", ":")), model
                    )
        if resolved_protocol != "content_only":
            for key in ("name", "tool_call_id", "call_id"):
                value = msg.get(key)
                if isinstance(value, str):
                    total += count_tokens(value, model)
            for key in ("tool_calls", "function_call"):
                value = msg.get(key)
                if value is not None:
                    total += count_tokens(
                        json.dumps(value, sort_keys=True, separators=(",", ":")), model
                    )
    return total
