"""Tool schema canonicalization for llama.cpp KV cache alignment.

llama.cpp's KV cache matches tokenized prefixes byte-for-byte. Tool
definitions are often 10-50KB and sent in full in every request. If the
tool schema JSON has different key ordering or formatting across requests,
the tokenized prefix changes and the KV cache misses — even though the
tool schema is semantically identical.

This module canonicalizes the ``tools`` field in request bodies:

1. **Key ordering** — all JSON keys sorted alphabetically at every nesting level
2. **Whitespace** — compact formatting (no unnecessary spaces/newlines)
3. **Numeric normalization** — ``1.0`` → ``1``, preserving non-integer floats

Usage::

    from legroom.compressors.tool_schema_canonicalizer import (
        ToolSchemaCanonicalizer,
    )

    canon = ToolSchemaCanonicalizer()
    # Returns a new body dict with canonicalized tool schemas
    new_body = canon.canonicalize_body(body)

The canonicalization is **semantically lossless**: the model receives the
same tool definitions, just with deterministic formatting.
"""

from __future__ import annotations

import json
from typing import Any


class ToolSchemaCanonicalizeResult:
    """Result of tool schema canonicalization."""

    def __init__(
        self,
        body: dict[str, Any],
        canonicalized_count: int = 0,
    ) -> None:
        self.body = body
        self.canonicalized_count = canonicalized_count


class ToolSchemaCanonicalizer:
    """Canonicalizes tool schemas in request bodies.

    Applies the same canonicalization rules as JsonCanonicalizer:
    sorted keys, compact formatting, numeric normalization.
    """

    def canonicalize_body(
        self,
        body: dict[str, Any],
    ) -> ToolSchemaCanonicalizeResult:
        """Canonicalize the tools field in a request body.

        Args:
            body: The request body dict (not modified in place).

        Returns:
            ToolSchemaCanonicalizeResult with canonicalized body and stats.
        """
        import copy

        body_copy = copy.deepcopy(body)
        tools = body_copy.get("tools")
        if not isinstance(tools, list) or not tools:
            return ToolSchemaCanonicalizeResult(
                body=body_copy,
                canonicalized_count=0,
            )

        canonicalized = False
        canonical_tools = []
        for tool in tools:
            if not isinstance(tool, dict):
                canonical_tools.append(tool)
                continue
            canon = self._canonicalize_tool(tool)
            # Python dict equality ignores key order, so we compare
            # JSON serialization to detect actual reformatting changes.
            if json.dumps(canon, sort_keys=False) != json.dumps(tool, sort_keys=False):
                canonicalized = True
            canonical_tools.append(canon)

        body_copy["tools"] = canonical_tools

        return ToolSchemaCanonicalizeResult(
            body=body_copy,
            canonicalized_count=len(canonical_tools),
        ) if canonicalized else ToolSchemaCanonicalizeResult(
            body=body_copy,
            canonicalized_count=0,
        )

    def _canonicalize_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize a single tool definition."""
        if "function" not in tool:
            return dict(tool)

        function = tool["function"]
        if not isinstance(function, dict):
            return dict(tool)

        canon_func = self._canonicalize_value(function)
        canon_tool = {**tool, "function": canon_func}

        # Also canonicalize the top-level keys
        canon_tool = dict(sorted(canon_tool.items()))

        return canon_tool

    def _canonicalize_value(self, value: Any) -> Any:
        """Canonicalize a parsed JSON value."""
        if isinstance(value, dict):
            return {
                k: self._canonicalize_value(v)
                for k, v in sorted(value.items())
            }
        elif isinstance(value, list):
            return [self._canonicalize_value(item) for item in value]
        elif isinstance(value, float):
            if value == int(value) and not (
                value == float("inf") or value == float("-inf")
                or value != value  # NaN check
            ):
                return int(value)
            return value
        return value

    def canonicalize_json_string(self, json_str: str) -> str:
        """Parse, canonicalize, and re-serialize a JSON string.

        Useful for normalizing tool schemas that arrive as JSON strings
        (some providers serialize tools as strings).

        Args:
            json_str: A JSON string containing a tool schema.

        Returns:
            Canonicalized JSON string with sorted keys and compact format.
        """
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return json_str

        canonical = self._canonicalize_value(parsed)
        return json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
