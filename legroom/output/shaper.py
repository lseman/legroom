"""Output shaper — reduces verbosity in model outputs."""

from __future__ import annotations

from typing import Any


class OutputShaper:
    """Reduces verbosity in model outputs by steering toward concise responses."""

    def __init__(self, verbosity_level: int = 2, protect_recent: int = 0) -> None:
        self.verbosity_level = verbosity_level
        self.protect_recent = protect_recent

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply output shaping to messages."""
        if not messages or self.protect_recent > 0:
            return messages
        latest_user = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"),
            None,
        )
        if latest_user is None:
            return messages
        content = messages[latest_user].get("content")
        directive = (
            "(Return only the requested content.)"
            if self.verbosity_level <= 1
            else "(Be concise — no preamble. Return only the requested content.)"
        )
        if not isinstance(content, str) or directive in content:
            return messages
        result = list(messages)
        result[latest_user] = {**result[latest_user], "content": f"{content.rstrip()}\n\n{directive}"}
        return result
