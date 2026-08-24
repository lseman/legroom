"""Compression store — stores and retrieves compressed content."""

from __future__ import annotations

import hashlib
import threading
from typing import Any


class CompressionStore:
    """Stores compressed content for retrieval via CCR."""

    def __init__(self, max_entries: int = 1000) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._max_entries = max_entries
        self._lock = threading.RLock()

    def store(
        self,
        original: str,
        compressed: str,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        compression_strategy: str | None = None,
        explicit_hash: str | None = None,
    ) -> str:
        """Store compressed content, return hash key.

        Callers may pass `explicit_hash` (e.g. a hash already quoted to the
        model in a marker) so the stored key matches what the model was told
        to retrieve. Otherwise a hash is derived from the content.
        """
        content_hash = explicit_hash or hashlib.sha256(original.encode()).hexdigest()[:16]

        with self._lock:
            self._store[content_hash] = {
                "original": original,
                "compressed": compressed,
                "size_before": len(original),
                "size_after": len(compressed),
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "compression_strategy": compression_strategy,
            }

            if len(self._store) > self._max_entries:
                oldest = next(iter(self._store))
                del self._store[oldest]

        return content_hash

    def retrieve(self, hash_key: str) -> str | None:
        """Retrieve original content by hash key."""
        with self._lock:
            entry = self._store.get(hash_key)
        if entry:
            return entry["original"]
        return None

    def contains_all(self, hash_keys: tuple[str, ...]) -> bool:
        """Return whether every cached CCR reference is still retrievable."""
        with self._lock:
            return all(hash_key in self._store for hash_key in hash_keys)

    def get_stats(self) -> dict[str, Any]:
        """Return store statistics."""
        with self._lock:
            total_before = sum(e["size_before"] for e in self._store.values())
            total_after = sum(e["size_after"] for e in self._store.values())
            return {
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "total_bytes_before": total_before,
                "total_bytes_after": total_after,
                "savings": total_before - total_after if total_before > 0 else 0,
            }
