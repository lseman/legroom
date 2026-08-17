"""Token counting utilities."""

from __future__ import annotations

import tiktoken

_MODEL_TO_ENCODING: dict[str, str] = {
    "gpt-4o": "cl100k_base",
    "gpt-4o-mini": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "claude-3-5-sonnet": "cl100k_base",
    "claude-3-opus": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
}


def get_encoding(model: str) -> tiktoken.Encoding:
    """Get the encoding for a model."""
    encoding_name = _MODEL_TO_ENCODING.get(model, "cl100k_base")
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text."""
    if not text:
        return 0
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def count_tokens_messages(messages: list[dict[str, Any]], model: str = "gpt-4o") -> int:
    """Count tokens in a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += count_tokens(block.get("text", ""), model)
    return total
