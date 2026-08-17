"""CCR tool injection — injects retrieval tool and instructions into LLM requests."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_MARKERS = [
    re.compile(r"\[(\d+) items? compressed\. hash=([a-f0-9]+)\]"),
    re.compile(r"<<ccr:([a-f0-9]+)>>"),
]


def create_ccr_tool_definition(provider: str = "anthropic") -> dict[str, Any]:
    """Create the CCR retrieval tool definition."""
    tool_def = {
        "name": "ccr_retrieve",
        "description": "Retrieve original content for a compressed output using its hash.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hash": {
                    "type": "string",
                    "description": "The hash key of the compressed content to retrieve.",
                },
            },
            "required": ["hash"],
        },
    }
    return tool_def


def create_system_instructions(hash_list: list[str]) -> str:
    """Create system instructions about compressed content."""
    hash_str = ", ".join(f"`{h}`" for h in hash_list[:10])
    more = "" if len(hash_list) <= 10 else f" (and {len(hash_list) - 10} more)"
    return f"""Some tool outputs have been compressed to reduce context size. If you need
the full uncompressed data, you can retrieve it using the `ccr_retrieve` tool.

**How to retrieve:**
- Call `ccr_retrieve(hash="<hash>")` to get the full original content back

**Available hashes:** {hash_str}{more}

Look for markers like `[N items compressed. hash=abc123]` or `<<ccr:abc123>>`
in tool results to find the hash for each compressed output.
"""


@dataclass
class CCRToolInjector:
    """Manages CCR tool injection into LLM requests."""

    provider: str = "anthropic"
    inject_tool: bool = True
    _inject_system_instructions: bool = True

    _detected_hashes: list[str] = field(default_factory=list, repr=False)

    @property
    def has_compressed_content(self) -> bool:
        return len(self._detected_hashes) > 0

    @property
    def detected_hashes(self) -> list[str]:
        return list(self._detected_hashes)

    def scan_for_markers(self, messages: list[dict[str, Any]]) -> list[str]:
        """Scan messages for compression markers and extract hashes."""
        self._detected_hashes = []
        seen: set[str] = set()

        for message in messages:
            content = message.get("content", "")

            if isinstance(content, str):
                self._scan_text(content, seen)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            self._scan_text(block.get("text", ""), seen)
                        elif btype == "tool_result":
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, str):
                                self._scan_text(tool_content, seen)
            elif isinstance(content, dict):
                for value in content.values():
                    if isinstance(value, str):
                        self._scan_text(value, seen)

            for part in message.get("parts", []):
                if isinstance(part, dict):
                    if "text" in part:
                        self._scan_text(part["text"], seen)

        return list(seen)

    def _scan_text(self, text: str, seen: set[str]) -> None:
        """Scan text for compression markers."""
        for pattern in _MARKERS:
            for match in pattern.finditer(text):
                hash_key = match.group(match.lastindex)
                if hash_key and hash_key not in seen:
                    seen.add(hash_key)
                    self._detected_hashes.append(hash_key)

    def inject_tool_definition(self, tools: list[dict] | None) -> list[dict]:
        """Inject the CCR retrieval tool into the tools array."""
        if not self.inject_tool or not tools:
            return tools or []

        tool_def = create_ccr_tool_definition(self.provider)
        existing_names = {
            t.get("function", {}).get("name")
            if isinstance(t, dict) and "function" in t
            else t.get("name")
            for t in tools
            if isinstance(t, dict)
        }
        if "ccr_retrieve" not in existing_names:
            tools.append(tool_def)
        return tools

    def inject_system_instructions(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Inject CCR instructions into the system message."""
        if not self._inject_system_instructions or not self.has_compressed_content:
            return messages

        instructions = create_system_instructions(self.detected_hashes)

        for msg in messages:
            if msg.get("role") == "system":
                existing = msg.get("content", "")
                msg["content"] = existing.rstrip() + "\n\n" + instructions
                return messages

        return [
            {"role": "system", "content": instructions},
            *messages,
        ]

    def process_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict] | None]:
        """Process a request: scan for markers, inject tool + instructions."""
        self.scan_for_markers(messages)
        if self.has_compressed_content:
            if tools is not None:
                tools = self.inject_tool_definition(tools)
            messages = self.inject_system_instructions(messages)
        return messages, tools
