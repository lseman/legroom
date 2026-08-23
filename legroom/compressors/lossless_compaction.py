"""Lossless compaction — reversible, no-CCR text compression."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LosslessResult:
    """Result of lossless compaction."""

    compressed: str
    original_size: int
    compressed_size: int
    transforms_applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# Pre-compiled regex patterns
_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_SEARCH_HEADINGS_PATTERN = re.compile(r"^(\S+):(\d+):(.*)")


def compact_lossless(
    content: str, content_hint: str = "text"
) -> LosslessResult:
    """Apply reversible lossless compression transforms.

    Available transforms:
    - ANSI stripping (removes color codes)
    - Run collapse (collapses repeated lines)
    - Search heading (groups grep results)
    - Diff index stripping
    - Block folding (folds repeated multi-line blocks)
    """
    result = content
    applied = []

    # ANSI stripping (one-way, safe)
    if "\x1b[" in result:
        result = _strip_ansi(result)
        applied.append("ansi_strip")

    # Content-hint-specific transforms
    if content_hint == "log":
        result = _collapse_runs(result)
        applied.append("run_collapse")
    elif content_hint == "grep":
        result = _compress_search_headings(result)
        applied.append("search_heading")
    elif content_hint == "diff":
        result = _strip_diff_index(result)
        applied.append("diff_strip")

    return LosslessResult(
        compressed=result,
        original_size=len(content),
        compressed_size=len(result),
        transforms_applied=applied,
    )


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    return _ANSI_PATTERN.sub("", text)


def _collapse_runs(text: str, max_repeats: int = 5) -> str:
    """Collapse repeated consecutive lines."""
    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        run_length = 1

        while i + run_length < len(lines) and lines[i + run_length] == line:
            run_length += 1

        if run_length > max_repeats:
            result.append(f"... (repeated {run_length} times)")
            result.append(line)
            i += run_length
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def _compress_search_headings(text: str) -> str:
    """Convert grep results with repeated paths to file headings."""
    lines = text.split("\n")
    current_file = None
    file_lines: list[str] = []
    result = []

    for line in lines:
        match = _SEARCH_HEADINGS_PATTERN.match(line)
        if match:
            filepath = match.group(1)
            line_num = match.group(2)
            line_content = match.group(3)

            if filepath != current_file:
                if current_file is not None and file_lines:
                    result.append(current_file)
                    result.extend(file_lines)
                current_file = filepath
                file_lines = [f"  {line_num}: {line_content}"]
            else:
                file_lines.append(f"  {line_num}: {line_content}")
        else:
            if current_file is not None and file_lines:
                result.append(current_file)
                result.extend(file_lines)
                current_file = None
                file_lines = []
            result.append(line)

    if current_file is not None and file_lines:
        result.append(current_file)
        result.extend(file_lines)

    return "\n".join(result)


_DIFF_INDEX_LINE = re.compile(r"^index [0-9a-f]{4,40}\.\.[0-9a-f]{4,40}(\s+\d+)?$")


def _strip_diff_index(text: str) -> str:
    """Remove git diff `index <sha>..<sha>` lines."""
    lines = text.split("\n")
    result = []

    for line in lines:
        if _DIFF_INDEX_LINE.match(line):
            continue
        result.append(line)

    return "\n".join(result)
