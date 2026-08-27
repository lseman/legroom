"""Stable, bounded, TTL-aware compression-result cache."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any


@dataclass(frozen=True)
class CachedCompression:
    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    transforms: list[str]
    ccr_hashes: tuple[str, ...] = ()
    tail_hash: str = ""

    @property
    def has_ccr(self) -> bool:
        return bool(self.ccr_hashes)


@dataclass
class _Entry:
    expires_at: float
    value: CachedCompression


class CompressionResultCache:
    """LRU cache whose key covers every input that changes compression output."""

    def __init__(self, maxsize: int = 256, ttl_seconds: float = 300.0) -> None:
        if maxsize < 1 or ttl_seconds <= 0:
            raise ValueError("cache size and TTL must be positive")
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        *, protocol: str, model: str, mode: str, messages: list[dict[str, Any]], policy: str
    ) -> str:
        document = json.dumps(
            [protocol, model, mode, policy, messages],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(document).hexdigest()

    # ------------------------------------------------------------------ #
    # Prefix-aware key (llama.cpp / StablePrefixCache)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_tail(
        messages: list[dict[str, Any]], model: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return (prefix_hash, tail_messages) for prefix-aware caching.

        The stable prefix consists of system / tool / function messages
        — the part that repeats verbatim across turns. Everything from
        the first user or assistant message onward is the "tail".

        Returns a SHA-256 hex prefix of the original prefix content so
        that cache entries are partitioned by prefix. Two requests with
        the same tail but different prefixes get independent cache slots.
        """
        import hashlib as _hashlib

        prefix: list[dict[str, Any]] = []
        tail: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            if role in ("system", "tool", "function"):
                prefix.append(msg)
            else:
                tail.append(msg)

        # Fallback: if no prefix messages exist, use the first message
        if not prefix and messages:
            prefix = [messages[0]]
            tail = list(messages[1:])

        prefix_hash = (
            "sp:" + _hashlib.sha256(
                json.dumps(
                    [model] + prefix,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )
        return prefix_hash, tail

    @staticmethod
    def tail_key(
        *, model: str, tail_messages: list[dict[str, Any]]
    ) -> str:
        """Cache key for prefix-aware compression (tail-only).

        When the StablePrefixCache is active and hits, the tail messages
        are compressed independently against the fixed compressed prefix.
        The compression result is a pure function of the tail, so the key
        needs only the model + tail content.

        Two requests that share the same tail get a cache hit regardless
        of what prefix they were paired with — the prefix cache handles
        prefix partitioning separately.

        This is the key optimization for llama.cpp: repeated conversation
        patterns (e.g. "list files", "show error", "run tests") get cache
        hits across different turns, even though the full message lists
        (including prefix) are completely different.
        """
        document = json.dumps(
            [model, tail_messages],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(document).hexdigest()

    def get_or_compute(
        self,
        *,
        model: str,
        tail_messages: list[dict[str, Any]],
        compute: Callable[[], CachedCompression],
    ) -> CachedCompression:
        """Lookup or compute with a tail-based cache key.

        When the StablePrefixCache is active and hits, the tail messages
        are compressed independently against the fixed prefix. This method
        keys the cache by tail alone, so repeated conversation patterns
        get cache hits regardless of what prefix they were paired with.

        If a cache hit occurs, returns the cached result immediately.
        Otherwise calls `compute()` and stores the result before returning.
        """
        key = self.tail_key(model=model, tail_messages=tail_messages)
        result = self.get(key)
        if result is not None:
            return result
        result = compute()
        self.put(key, result)
        return result

    def get(self, key: str) -> CachedCompression | None:
        entry = self._entries.get(key)
        now = time.monotonic()
        if entry is None or entry.expires_at <= now:
            if entry is not None:
                del self._entries[key]
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return copy.deepcopy(entry.value)

    def put(self, key: str, value: CachedCompression) -> None:
        self._entries[key] = _Entry(time.monotonic() + self._ttl, copy.deepcopy(value))
        self._entries.move_to_end(key)
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def clear(self) -> None:
        self._entries.clear()

    def discard(self, key: str) -> None:
        self._entries.pop(key, None)
