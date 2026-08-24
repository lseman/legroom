"""Provider prompt-cache controls, usage normalization, and cost accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

CacheMode = Literal["off", "implicit", "explicit"]


@dataclass(frozen=True)
class CachePricing:
    """Prices per million tokens; callers supply current provider rates."""

    uncached_input: float = 0.0
    cache_write: float = 0.0
    cache_read: float = 0.0

    def __post_init__(self) -> None:
        if min(self.uncached_input, self.cache_write, self.cache_read) < 0:
            raise ValueError("cache prices must be non-negative")


@dataclass(frozen=True)
class ProviderCacheUsage:
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cached_tokens: int = 0

    @property
    def uncached_tokens(self) -> int:
        return max(0, self.input_tokens - self.cache_write_tokens - self.cached_tokens)

    def cost(self, pricing: CachePricing) -> float:
        return (
            self.uncached_tokens * pricing.uncached_input
            + self.cache_write_tokens * pricing.cache_write
            + self.cached_tokens * pricing.cache_read
        ) / 1_000_000


@dataclass(frozen=True)
class ProviderCachePolicy:
    """Apply cache controls without overriding fields explicitly set by callers."""

    mode: CacheMode = "implicit"
    key: str | None = None
    ttl: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"off", "implicit", "explicit"}:
            raise ValueError("cache mode must be 'off', 'implicit', or 'explicit'")
        if self.ttl is not None and self.ttl != "24h":
            raise ValueError("OpenAI prompt cache retention currently supports only '24h'")

    def apply(self, body: dict[str, Any], *, protocol: str) -> bool:
        if self.mode == "off" or not protocol.startswith("openai_"):
            return False
        changed = False
        key = self.key or self._stable_key(body, protocol)
        if "prompt_cache_key" not in body:
            body["prompt_cache_key"] = key
            changed = True
        if self.mode == "explicit" and "prompt_cache_retention" not in body:
            body["prompt_cache_retention"] = self.ttl or "24h"
            changed = True
        return changed

    @staticmethod
    def _stable_key(body: dict[str, Any], protocol: str) -> str:
        # Deliberately excludes dynamic conversation content. The model and
        # stable tool/schema prefix identify a cache affinity group.
        stable = {
            "protocol": protocol,
            "model": body.get("model"),
            "tools": body.get("tools"),
            "instructions": body.get("instructions"),
        }
        encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
        return "legroom-" + hashlib.sha256(encoded).hexdigest()[:24]


def parse_cache_usage(document: dict[str, Any]) -> ProviderCacheUsage:
    """Normalize Chat Completions and Responses usage documents."""
    usage = document.get("usage")
    if not isinstance(usage, dict):
        return ProviderCacheUsage()
    input_tokens = _integer(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
    details = usage.get("input_tokens_details", usage.get("prompt_tokens_details", {}))
    if not isinstance(details, dict):
        details = {}
    return ProviderCacheUsage(
        input_tokens=input_tokens,
        cache_write_tokens=_integer(details.get("cache_write_tokens", 0)),
        cached_tokens=_integer(details.get("cached_tokens", 0)),
    )


def _integer(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


class StreamingUsageParser:
    """Incrementally extract usage from fragmented Responses/Chat SSE events."""

    def __init__(self, max_buffer_bytes: int = 1_000_000) -> None:
        self._buffer = b""
        self._max_buffer_bytes = max_buffer_bytes
        self.usage = ProviderCacheUsage()

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        if len(self._buffer) > self._max_buffer_bytes:
            self._buffer = self._buffer[-self._max_buffer_bytes :]
        lines = self._buffer.split(b"\n")
        self._buffer = lines.pop()
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                document = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(document, dict):
                continue
            candidate = document.get("response", document)
            if isinstance(candidate, dict):
                usage = parse_cache_usage(candidate)
                if usage.input_tokens or usage.cached_tokens or usage.cache_write_tokens:
                    self.usage = usage
