"""Content type detection for the router."""

from __future__ import annotations

import re
import json
from typing import Any


class ContentDetector:
    """Detects content type: json, code, log, search, mixed."""

    LOG_PATTERN = re.compile(
        r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
        r"(\.\d+)?(Z|[+-]\d{2}:?\d{2})?"
        r"\s+(DEBUG|INFO|WARN|ERROR|FATAL|CRITICAL|TRACE|NOTICE|WARNING)",
        re.MULTILINE,
    )
    SEARCH_PATTERN = re.compile(r"^\S+:\d+:", re.MULTILINE)
    CODE_KEYWORDS = {"def ", "class ", "import ", "from ", "func ", "struct ", "fn ", "let "}
    JSON_PATTERN = re.compile(r"^\s*[\{\[]")

    def detect(self, content: str) -> str:
        """Detect the type of content."""
        if not content or not content.strip():
            return "text"

        stripped = content.strip()

        # Try JSON first (balanced braces/brackets)
        if self.JSON_PATTERN.match(stripped):
            try:
                json.loads(stripped)
                return "json"
            except (json.JSONDecodeError, ValueError):
                pass

        # Check for log format
        if self.LOG_PATTERN.search(stripped):
            return "log"

        # Check for search/grep format (path:line:content)
        if self.SEARCH_PATTERN.search(stripped):
            return "search"

        # Check for code
        keywords_found = sum(1 for kw in self.CODE_KEYWORDS if kw in stripped)
        if keywords_found >= 1 or "```" in stripped:
            return "code"

        return "text"

    def detect_sections(self, content: str) -> list[tuple[str, str]]:
        """Split mixed content into typed sections."""
        sections = []
        lines = content.split("\n")
        current_type = "text"
        current_lines = []

        for line in lines:
            section_type = self._classify_line(line)
            if section_type and section_type != current_type:
                if current_lines:
                    sections.append((current_type, "\n".join(current_lines)))
                current_type = section_type
                current_lines = [line]
            elif current_type == "text":
                current_type = self._classify_line(line) or "text"
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_type, "\n".join(current_lines)))

        return sections if sections else [("text", content)]

    def _classify_line(self, line: str) -> str | None:
        """Classify a single line."""
        if self.LOG_PATTERN.match(line):
            return "log"
        if self.SEARCH_PATTERN.match(line):
            return "search"
        if any(kw in line for kw in self.CODE_KEYWORDS) or line.strip().startswith("```"):
            return "code"
        return None
