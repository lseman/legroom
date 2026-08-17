"""Output shaper — reduces verbosity in model outputs."""

from __future__ import annotations

from typing import Any


class OutputShaper:
    """Reduces verbosity in model outputs by steering toward concise responses."""

    def __init__(self, verbosity_level: int = 2) -> None:
        self.verbosity_level = verbosity_level

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply output shaping to messages."""
        if not messages:
            return messages

        # Find the last assistant message (where we can steer output)
        result = list(messages)
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "assistant":
                content = result[i].get("content", "")
                if self._is_concise_enough(content):
                    break
                # Add steering hint to next user message
                for j in range(i + 1, len(result)):
                    if result[j].get("role") == "user":
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
