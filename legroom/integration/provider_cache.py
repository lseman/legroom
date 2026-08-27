"""Provider prompt-cache controls, usage normalization, and cost accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

CacheMode = Literal["off", "implicit", "explicit"]
Backend = Literal["openai", "llama_cpp"]


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
    """Apply cache controls without overriding fields explicitly set by callers.

    ``backend`` selects which wire fields are meaningful:

    - ``openai``: injects ``prompt_cache_key`` / ``prompt_cache_retention``,
      the server-side prompt-cache controls for OpenAI's Chat/Responses APIs.
    - ``llama_cpp``: injects ``id_slot`` / ``cache_prompt`` for llama.cpp's
      OpenAI-compatible server, which reuses a request's KV cache by slot
      affinity and longest-common-prefix match against what a slot already
      holds. llama.cpp compares the *raw token prefix byte-for-byte*, so this
      policy never touches message content — only routing fields.
    """

    mode: CacheMode = "implicit"
    key: str | None = None
    ttl: str | None = None
    backend: Backend = "openai"

    def __post_init__(self) -> None:
        if self.mode not in {"off", "implicit", "explicit"}:
            raise ValueError("cache mode must be 'off', 'implicit', or 'explicit'")
        if self.backend not in {"openai", "llama_cpp"}:
            raise ValueError("backend must be 'openai' or 'llama_cpp'")
        if self.ttl is not None:
            if self.backend != "openai":
                raise ValueError("prompt-cache TTL is only meaningful for the 'openai' backend")
            if self.ttl != "24h":
                raise ValueError("OpenAI prompt cache retention currently supports only '24h'")

    def apply(self, body: dict[str, Any], *, protocol: str) -> bool:
        if self.mode == "off":
            return False
        if self.backend == "llama_cpp":
            return self._apply_llama_cpp(body)
        if not protocol.startswith("openai_"):
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

    def _apply_llama_cpp(self, body: dict[str, Any]) -> bool:
        changed = False
        # cache_prompt tells llama.cpp to keep and reuse this slot's KV cache
        # across requests instead of recomputing the prompt from scratch.
        if "cache_prompt" not in body:
            body["cache_prompt"] = True
            changed = True
        # id_slot pins a logical conversation to one server slot so its KV
        # cache stays resident between turns. Without a stable id, llama.cpp
        # picks the least-recently-used slot, which still allows prefix reuse
        # but is not guaranteed to land on the same slot as the prior turn.
        if "id_slot" not in body:
            slot_key = self.key or self._stable_key(body)
            body["id_slot"] = _stable_slot_id(slot_key)
            changed = True
        return changed

    @staticmethod
    def _stable_key(body: dict[str, Any], protocol: str | None = None) -> str:
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


# llama.cpp's id_slot is a non-negative integer (-1 means "let the server
# choose"), not an opaque string like OpenAI's prompt_cache_key. Hash the
# stable key down into a fixed, small positive range so distinct cache
# affinity groups tend to land on distinct slots without needing to know how
# many slots the server was started with.
#
# 8192 gives good collision resistance for typical deployments (dozens to
# hundreds of concurrent conversations) while keeping slot IDs small
# enough for llama.cpp's internal arrays. If the server reports slot
# exhaustion, reduce this number to match the server's --contiguous-slots
# count.
_LLAMA_CPP_SLOT_SPACE = 8196


def _stable_slot_id(key: str) -> int:
    # Use 8 bytes for better avalanche — 4 bytes has a 1-in-65536 collision
    # probability with 8196 slots (birthday paradox), while 8 bytes pushes
    # that to effectively zero for any realistic conversation count.
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % _LLAMA_CPP_SLOT_SPACE


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
