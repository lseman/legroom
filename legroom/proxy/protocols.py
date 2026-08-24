"""Native-document adapters for proxy compression protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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

    def apply(self, body: dict[str, Any], compressed: list[dict[str, Any]]) -> bool:
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
    if path == "/v1/chat/completions":
        field = "messages"
        protocol = "openai_chat"
    elif path == "/v1/responses":
        field = "input"
        protocol = "openai_responses"
    else:
        return None

    native = body.get(field)
    if not isinstance(native, list) or not all(isinstance(item, dict) for item in native):
        return None

    messages = native
    offset = 0
    indices = None
    if protocol == "openai_responses":
        eligible = tuple(
            index
            for index, item in enumerate(native)
            if item.get("type", "message") == "message"
            and item.get("role") in {"system", "developer", "user", "assistant"}
            and "content" in item
        )
        if mode == "cache" and eligible:
            eligible = eligible[-1:]
        indices = eligible
        messages = [native[index] for index in eligible]
    elif mode == "cache" and messages:
        offset = len(messages) - 1
        messages = messages[offset:]

    return CompressionView(
        protocol=protocol,
        model=str(body.get("model", "gpt-4o")),
        messages=messages,
        field=field,
        offset=offset,
        indices=indices,
    )


def normalize_mode(value: str) -> ProxyMode:
    normalized = value.strip().lower()
    if normalized not in {"token", "cache"}:
        raise ValueError("mode must be 'token' or 'cache'")
    return normalized  # type: ignore[return-value]
