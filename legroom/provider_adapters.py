"""Lossless provider-document adapters for the Legroom IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .ir import Conversation

ProviderName = Literal["openai_chat", "openai_responses"]


@dataclass(frozen=True)
class AdaptedConversation:
    provider: ProviderName
    conversation: Conversation
    field: str
    indices: tuple[int, ...]

    def apply(self, body: dict[str, Any], conversation: Conversation) -> bool:
        if len(conversation.messages) != len(self.indices):
            raise ValueError("Compression must preserve provider message count")
        original = body[self.field]
        replacement = list(original)
        for index, message in zip(self.indices, conversation.messages, strict=True):
            replacement[index] = message.to_mapping()
        if replacement == original:
            return False
        body[self.field] = replacement
        return True


class ProviderAdapter(Protocol):
    """Seam for turning provider documents into the Legroom IR."""

    def parse(self, body: dict[str, Any], *, cache_mode: bool = False) -> AdaptedConversation | None:
        ...


class OpenAIChatAdapter:
    def parse(self, body: dict[str, Any], *, cache_mode: bool = False) -> AdaptedConversation | None:
        native = body.get("messages")
        if not isinstance(native, list) or not all(isinstance(item, dict) for item in native):
            return None
        indices = tuple(range(len(native)))
        if cache_mode and indices:
            indices = indices[-1:]
        selected = [native[index] for index in indices]
        return AdaptedConversation(
            "openai_chat", Conversation.from_mappings(selected), "messages", indices
        )


class OpenAIResponsesAdapter:
    def parse(self, body: dict[str, Any], *, cache_mode: bool = False) -> AdaptedConversation | None:
        native = body.get("input")
        if not isinstance(native, list) or not all(isinstance(item, dict) for item in native):
            return None
        indices = tuple(
            index
            for index, item in enumerate(native)
            if item.get("type", "message") == "message"
            and item.get("role") in {"system", "developer", "user", "assistant"}
            and "content" in item
        )
        if cache_mode and indices:
            indices = indices[-1:]
        selected = [native[index] for index in indices]
        return AdaptedConversation(
            "openai_responses", Conversation.from_mappings(selected), "input", indices
        )


def adapter_for_path(path: str) -> ProviderAdapter | None:
    if path == "/v1/chat/completions":
        return OpenAIChatAdapter()
    if path == "/v1/responses":
        return OpenAIResponsesAdapter()
    return None
