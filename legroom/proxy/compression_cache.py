"""Stable, bounded, TTL-aware compression-result cache."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CachedCompression:
    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    transforms: list[str]
    ccr_hashes: tuple[str, ...] = ()

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
