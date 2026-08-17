"""Compression store — stores and retrieves compressed content."""

from __future__ import annotations

import json
import hashlib
from typing import Any


class CompressionStore:
    """Stores compressed content for retrieval via CCR."""

    def __init__(self, max_entries: int = 1000) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._max_entries = max_entries

    def store(self, original: str, compressed: str) -> str:
        """Store compressed content, return hash key."""
        content_hash = hashlib.sha256(original.encode()).hexdigest()[:16]

        self._store[content_hash] = {
            "original": original,
            "compressed": compressed,
            "size_before": len(original),
            "size_after": len(compressed),
        }

        # Evict oldest if full (simple FIFO)
        if len(self._store) > self._max_entries:
            oldest = next(iter(self._store))
            del self._store[oldest]

        return content_hash

    def retrieve(self, hash_key: str) -> str | None:
        """Retrieve original content by hash key."""
        entry = self._store.get(hash_key)
        if entry:
            return entry["original"]
        return None

    def get_stats(self) -> dict[str, Any]:
        """Return store statistics."""
        total_before = sum(e["size_before"] for e in self._store.values())
        total_after = sum(e["size_after"] for e in self._store.values())
        return {
            "entries": len(self._store),
            "max_entries": self._max_entries,
            "total_bytes_before": total_before,
            "total_bytes_after": total_after,
            "savings": total_before - total_after if total_before > 0 else 0,
        }
