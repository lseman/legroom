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
        if not messages or self.protect_recent <= 0:
            return messages

        result = list(messages)
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "assistant":
                content = result[i].get("content", "")
                if not isinstance(content, str) or not content.strip() or self._is_concise_enough(content):
                    break
                # Steer the last user message only if not protected
                if self.protect_recent > 0 and i < len(result) - self.protect_recent:
                    continue
                for j in range(i + 1, len(result)):
                    if result[j].get("role") == "user" and j >= len(result) - self.protect_recent:
                        break
                    if result[j].get("role") == "user" and j == len(result) - 1:
                        existing = result[j].get("content", "")
                        result[j]["content"] = existing + "\n\n(Be concise — no preamble. Return only the requested content.)"
                        break
                break

        return result

    def _is_concise_enough(self, content: str) -> bool:
        """Check if content is already concise."""
        if not content:
            return True
        words = content.split()
        return len(words) < 100
