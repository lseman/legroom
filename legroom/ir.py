"""Typed, provider-neutral representation of compressible conversations.

The IR deliberately owns only the concepts Legroom needs. Original provider
documents remain attached to each value so adapters can round-trip fields that
Legroom does not understand yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Literal

type JSONValue = (
    bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None
)
ContentShape = Literal["missing", "string", "blocks", "opaque"]


class Provenance(StrEnum):
    TRUSTED = "trusted"
    USER = "user"
    MODEL = "model"
    TOOL = "tool"
    PROVIDER = "provider"


class CompressionRisk(StrEnum):
    IMMUTABLE = "immutable"
    EXACT = "exact"
    HIGH = "high"
    REVERSIBLE = "reversible"
    NORMAL = "normal"


@dataclass(frozen=True)
class TextBlock:
    """Textual content plus the provider fields surrounding it."""

    text: str
    raw: Mapping[str, Any] | None = None
    text_field: str = "text"


@dataclass(frozen=True)
class OpaqueBlock:
    """Provider content that must survive but must not be rewritten."""

    raw: Any


type ContentBlock = TextBlock | OpaqueBlock


@dataclass(frozen=True)
class Message:
    """A role-bearing message with lossless provider provenance."""

    role: str
    blocks: tuple[ContentBlock, ...]
    raw: Mapping[str, Any]
    content_shape: ContentShape
    provenance: Provenance = Provenance.PROVIDER
    risk: CompressionRisk = CompressionRisk.NORMAL

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Message:
        raw = deepcopy(dict(value))
        role = str(value.get("role", ""))
        provenance, risk = _classify_message(role, value)
        if "content" not in value:
            return cls(role, (), raw, "missing", provenance, risk)
        content = value["content"]
        if isinstance(content, str):
            return cls(role, (TextBlock(content),), raw, "string", provenance, risk)
        if isinstance(content, list):
            blocks: list[ContentBlock] = []
            for block in content:
                if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                    blocks.append(TextBlock(block["text"], deepcopy(dict(block))))
                else:
                    blocks.append(OpaqueBlock(deepcopy(block)))
            return cls(role, tuple(blocks), raw, "blocks", provenance, risk)
        return cls(role, (OpaqueBlock(deepcopy(content)),), raw, "opaque", provenance, risk)

    def to_mapping(self) -> dict[str, Any]:
        result = deepcopy(dict(self.raw))
        result["role"] = self.role
        if self.content_shape == "missing":
            result.pop("content", None)
        elif self.content_shape == "string":
            result["content"] = "".join(
                block.text for block in self.blocks if isinstance(block, TextBlock)
            )
        elif self.content_shape == "blocks":
            rendered: list[Any] = []
            for block in self.blocks:
                if isinstance(block, OpaqueBlock):
                    rendered.append(deepcopy(block.raw))
                else:
                    item = deepcopy(dict(block.raw or {"type": "text"}))
                    item[block.text_field] = block.text
                    rendered.append(item)
            result["content"] = rendered
        elif self.blocks and isinstance(self.blocks[0], OpaqueBlock):
            result["content"] = deepcopy(self.blocks[0].raw)
        return result

    def with_text(self, text: str) -> Message:
        """Replace a scalar text body while retaining provider metadata."""
        if self.content_shape != "string":
            raise ValueError("with_text requires scalar string content")
        return replace(self, blocks=(TextBlock(text),))


@dataclass(frozen=True)
class Conversation:
    """Ordered provider-neutral messages."""

    messages: tuple[Message, ...]

    @classmethod
    def from_mappings(cls, values: list[dict[str, Any]]) -> Conversation:
        return cls(tuple(Message.from_mapping(value) for value in values))

    def to_mappings(self) -> list[dict[str, Any]]:
        return [message.to_mapping() for message in self.messages]


def _classify_message(
    role: str, value: Mapping[str, Any]
) -> tuple[Provenance, CompressionRisk]:
    if role in {"system", "developer"}:
        return Provenance.TRUSTED, CompressionRisk.IMMUTABLE
    if role == "tool" or value.get("type") == "function_call_output":
        return Provenance.TOOL, CompressionRisk.REVERSIBLE
    if role == "user":
        return Provenance.USER, CompressionRisk.HIGH
    if role == "assistant":
        if value.get("tool_calls") or value.get("function_call"):
            return Provenance.MODEL, CompressionRisk.EXACT
        return Provenance.MODEL, CompressionRisk.NORMAL
    return Provenance.PROVIDER, CompressionRisk.EXACT
