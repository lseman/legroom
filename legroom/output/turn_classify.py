"""Turn classification — classifies message turns for output shaping."""

from __future__ import annotations

from typing import Any


def classify_turn(message: dict[str, Any]) -> str:
    """Classify a message turn type."""
    role = message.get("role", "")
    content = message.get("content", "")

    if role == "system":
        return "system"
    elif role == "user":
        if isinstance(content, str) and ("```" in content or "json" in content.lower()):
            return "user_request"
        return "user_query"
    elif role == "assistant":
        if isinstance(content, str) and "<think" in content:
            return "assistant_thinking"
        return "assistant_response"
    elif role == "tool":
        return "tool_result"

    return "unknown"
