"""Proxy state — central tracking for compression metrics and history.

All compression events, CCR store operations, and read lifecycle transforms
flow through this object, making stats and history available to both the
proxy handler and the dashboard API.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _event_to_dict(event: RequestEvent) -> dict[str, Any]:
    """Convert a RequestEvent to a plain dict."""
    return {
        "request_id": event.request_id,
        "timestamp": event.timestamp,
        "model": event.model,
        "messages_before": event.messages_before,
        "tokens_before": event.tokens_before,
        "tokens_after": event.tokens_after,
        "tokens_saved": event.tokens_saved,
        "transforms_applied": list(event.transforms_applied),
        "warnings": list(event.warnings),
        "read_lifecycle_stats": dict(event.read_lifecycle_stats),
        "compression_details": list(event.compression_details),
    }


@dataclass
class RequestEvent:
    """Single compression request event."""

    request_id: str
    timestamp: float  # Unix timestamp
    model: str
    messages_before: int
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    transforms_applied: list[str]
    warnings: list[str]
    read_lifecycle_stats: dict[str, Any] = field(default_factory=dict)
    compression_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProxyState:
    """Central state tracker for the proxy server."""

    max_history: int = 1000  # Keep last N requests in memory

    # Request history (deque for O(1) append, bounded)
    _history: deque[RequestEvent] = field(default_factory=deque)

    # Aggregate stats
    total_requests: int = 0
    total_tokens_saved: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    total_messages: int = 0

    # Read lifecycle stats (accumulated)
    total_reads_compressed: int = 0
    total_reads_stale: int = 0
    total_reads_superseded: int = 0
    total_reads_fresh: int = 0

    # CCR store stats
    total_ccr_stored: int = 0
    total_ccr_retrieved: int = 0

    # Compression strategy counts
    strategy_counts: Counter[str] = field(default_factory=Counter)

    # Each live client owns a bounded queue so events are broadcast rather
    # than divided among consumers. Slow clients lose their oldest event.
    _subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)
    dropped_live_events: int = 0

    # Last updated timestamp
    last_updated: float = field(default_factory=time.time)

    def record_request(
        self,
        request_id: str,
        model: str,
        messages_before: int,
        tokens_before: int,
        tokens_after: int,
        transforms_applied: list[str],
        warnings: list[str],
        read_lifecycle_stats: dict[str, Any] | None = None,
        compression_details: list[dict[str, Any]] | None = None,
    ) -> RequestEvent:
        """Record a compression request event."""
        tokens_saved = tokens_before - tokens_after
        now = time.time()

        event = RequestEvent(
            request_id=request_id,
            timestamp=now,
            model=model,
            messages_before=messages_before,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            transforms_applied=transforms_applied,
            warnings=warnings,
            read_lifecycle_stats=read_lifecycle_stats or {},
            compression_details=compression_details or [],
        )

        # Append to history (bounded)
        self._history.append(event)
        if len(self._history) > self.max_history:
            self._history.popleft()

        # Update aggregate stats
        self.total_requests += 1
        self.total_tokens_saved += tokens_saved
        self.total_tokens_before += tokens_before
        self.total_tokens_after += tokens_after
        self.total_messages += messages_before

        # Update strategy counts
        for transform in transforms_applied:
            self.strategy_counts[transform] += 1

        # Update read lifecycle stats
        if read_lifecycle_stats:
            self.total_reads_compressed += read_lifecycle_stats.get("reads_stale", 0) + read_lifecycle_stats.get("reads_superseded", 0)
            self.total_reads_stale += read_lifecycle_stats.get("reads_stale", 0)
            self.total_reads_superseded += read_lifecycle_stats.get("reads_superseded", 0)
            self.total_reads_fresh += read_lifecycle_stats.get("reads_fresh", 0)

        self.last_updated = now

        # Push to live event queue (for dashboard WebSocket)
        self._emit_event(event)

        return event

    def _emit_event(self, event: RequestEvent) -> None:
        """Emit a live event to the queue."""
        payload = {
            "type": "request",
            "data": {
                "request_id": event.request_id,
                "timestamp": event.timestamp,
                "model": event.model,
                "tokens_before": event.tokens_before,
                "tokens_after": event.tokens_after,
                "tokens_saved": event.tokens_saved,
                "transforms": event.transforms_applied,
            },
        }
        for queue in tuple(self._subscribers):
            if queue.full():
                queue.get_nowait()
                self.dropped_live_events += 1
            queue.put_nowait(payload)

    def record_ccr_store(self, count: int = 1) -> None:
        """Record a CCR store operation."""
        self.total_ccr_stored += count
        self.last_updated = time.time()

    def record_ccr_retrieve(self, count: int = 1) -> None:
        """Record a CCR retrieval operation."""
        self.total_ccr_retrieved += count
        self.last_updated = time.time()

    def get_stats(self) -> dict[str, Any]:
        """Get current aggregate stats."""
        avg_tokens_before = self.total_tokens_before / max(self.total_requests, 1)
        avg_tokens_after = self.total_tokens_after / max(self.total_requests, 1)
        avg_tokens_saved = self.total_tokens_saved / max(self.total_requests, 1)

        # Calculate overall compression ratio
        if self.total_requests == 0 or self.total_tokens_before == 0:
            compression_ratio = 0.0
        else:
            compression_ratio = 1.0 - (self.total_tokens_after / max(self.total_tokens_before, 1))

        return {
            "total_requests": self.total_requests,
            "total_tokens_saved": self.total_tokens_saved,
            "total_tokens_before": self.total_tokens_before,
            "total_tokens_after": self.total_tokens_after,
            "total_messages": self.total_messages,
            "avg_tokens_before": round(avg_tokens_before, 1),
            "avg_tokens_after": round(avg_tokens_after, 1),
            "avg_tokens_saved": round(avg_tokens_saved, 1),
            "compression_ratio": round(compression_ratio * 100, 1),
            "total_reads_stale": self.total_reads_stale,
            "total_reads_superseded": self.total_reads_superseded,
            "total_reads_fresh": self.total_reads_fresh,
            "total_reads_compressed": self.total_reads_compressed,
            "total_ccr_stored": self.total_ccr_stored,
            "total_ccr_retrieved": self.total_ccr_retrieved,
            "strategy_counts": dict(self.strategy_counts),
            "last_updated": self.last_updated,
            "live_subscribers": len(self._subscribers),
            "dropped_live_events": self.dropped_live_events,
        }

    def get_history(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Get recent request history (most recent first)."""
        entries = list(self._history)
        # Most recent first
        entries = list(reversed(entries))
        # Serialize dataclass events to dicts for JSON/API consumption
        return [_event_to_dict(e) for e in entries[offset:offset + limit]]

    def get_read_lifecycle_stats(self) -> dict[str, Any]:
        """Get read lifecycle specific stats."""
        return {
            "total_reads_compressed": self.total_reads_compressed,
            "total_reads_stale": self.total_reads_stale,
            "total_reads_superseded": self.total_reads_superseded,
            "total_reads_fresh": self.total_reads_fresh,
            "compression_rate": (
                round(self.total_reads_compressed / max(self.total_reads_fresh + self.total_reads_compressed, 1) * 100, 1)
                if (self.total_reads_fresh + self.total_reads_compressed) > 0
                else 0
            ),
        }

    def subscribe(self, max_events: int = 128) -> asyncio.Queue[dict[str, Any]]:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=max_events)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def get_live_event(
        self, queue: asyncio.Queue[dict[str, Any]], timeout: float = 10
    ) -> dict[str, Any] | None:
        """Get live events from the queue (for WebSocket)."""
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
