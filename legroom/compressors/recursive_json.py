"""Recursive JSON routing — finds and compresses embedded JSON spans."""

from __future__ import annotations

import json
from collections.abc import Callable

from .balanced_end import find_balanced_end


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
    """Find balanced JSON spans in text with fast pre-filters.

    Two pre-filters skip expensive balanced-end scan + json.loads:
      1. Previous char is a letter or '$' — likely code/templating, not JSON.
      2. Balanced-end scan capped at 8 KB — real JSON is rarely larger.
    """
    spans = []
    i = 0
    n = len(text)
    MAX_JSON_SCAN = 8192

    while i < n:
        c = text[i]
        if c in "{}[":
            # Code/templating pre-filter: skip positions preceded by letters
            # or '$' (e.g. "def foo{bar}", "${x}")
            if i > 0 and (text[i - 1].isalpha() or text[i - 1] == "$"):
                i += 1
                continue

            end = find_balanced_end(text, i, max_scan=MAX_JSON_SCAN)
            if end is not None:
                span = text[i:end]
                try:
                    json.loads(span)
                    spans.append((i, end))
                    i = end
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass
        i += 1

    return spans


def _has_ccr_markers(text: str) -> bool:
    """Check if text has CCR compression markers."""
    return "[N items compressed" in text or "<<ccr:" in text
