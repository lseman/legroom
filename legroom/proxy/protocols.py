"""Native-document adapters for proxy compression protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..integration.provider_adapters import AdaptedConversation, adapter_for_path
from ..runtime.ir import Conversation

ProxyMode = Literal["token", "cache"]


@dataclass(frozen=True)
class CompressionView:
    """A compressible message view over a provider-native request document."""

    protocol: str
    model: str
    messages: list[dict[str, Any]]
    field: str
    offset: int = 0
    indices: tuple[int, ...] | None = None
    adapted: AdaptedConversation | None = None

    def apply(self, body: dict[str, Any], compressed: list[dict[str, Any]]) -> bool:
        if self.adapted is not None:
            return self.adapted.apply(body, Conversation.from_mappings(compressed))
        original = body[self.field]
        if self.indices is not None:
            if len(compressed) != len(self.indices):
                raise ValueError("Responses compression must preserve message item count")
            replacement = list(original)
            for index, item in zip(self.indices, compressed, strict=True):
                replacement[index] = item
        elif self.offset:
            replacement = [*original[: self.offset], *compressed]
        else:
            replacement = compressed
        if replacement == original:
            return False
        body[self.field] = replacement
        return True


def compression_view(
    path: str,
    body: dict[str, Any],
    mode: ProxyMode,
) -> CompressionView | None:
    """Return a lossless view only for request shapes Legroom understands."""
    adapter = adapter_for_path(path)
    if adapter is None:
        return None
    adapted = adapter.parse(body, cache_mode=mode == "cache")
    if adapted is None:
        return None

    return CompressionView(
        protocol=adapted.provider,
        model=str(body.get("model", "gpt-4o")),
        messages=adapted.conversation.to_mappings(),
        field=adapted.field,
        indices=adapted.indices,
        adapted=adapted,
    )


def normalize_mode(value: str) -> ProxyMode:
    normalized = value.strip().lower()
    if normalized not in {"token", "cache"}:
        raise ValueError("mode must be 'token' or 'cache'")
    return normalized  # type: ignore[return-value]
