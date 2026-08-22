"""Recursive JSON routing — finds and compresses embedded JSON spans."""

from __future__ import annotations

import re
import json
from typing import Callable


def route_embedded_json(
    text: str,
    dispatch: Callable[[str], str | None],
    tok: Callable[[str], int] | None = None,
) -> str | None:
    """Find balanced JSON spans in text and route them through a compressor.

    Returns the modified text if any JSON was compressed, or None if
    no compression was needed.
    """
    if not text or not isinstance(text, str):
        return None

    # Skip if already compressed (has markers)
    if _has_ccr_markers(text):
        return None

    result = []
    last_end = 0
    compressed_any = False

    for match in _find_json_spans(text):
        start, end = match
        json_span = text[start:end]

        # Try to dispatch through compressor
        dispatch_result = dispatch(json_span)

        if dispatch_result and dispatch_result != json_span:
            result.append(text[last_end:start])
            result.append(dispatch_result)
            last_end = end
            compressed_any = True

    if not compressed_any:
        return None

    result.append(text[last_end:])
    return "".join(result)


def _find_json_spans(text: str) -> list[tuple[int, int]]:
    """Find balanced JSON spans in text."""
    spans = []
    i = 0

    while i < len(text):
        if text[i] in "{[":
            # Found potential JSON start
            start = i
            end = _find_balanced_end(text, i)
            if end is not None:
                span = text[start:end]
                # Verify it's valid JSON
                try:
                    json.loads(span)
                    spans.append((start, end))
                    i = end
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass
        i += 1

    return spans


def _find_balanced_end(text: str, start: int) -> int | None:
    """Find the end of a balanced JSON span starting at start."""
    if start >= len(text):
        return None

    open_char = text[start]
    close_char = "{" if open_char == "{" else "}"

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]

        if escape_next:
            escape_next = False
            continue

        if c == "\\":
            escape_next = True
            continue

        if c == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return i + 1

    return None


def _has_ccr_markers(text: str) -> bool:
    """Check if text has CCR compression markers."""
    return "[N items compressed" in text or "<<ccr:" in text
