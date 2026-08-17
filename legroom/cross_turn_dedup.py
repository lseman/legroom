"""Cross-turn deduplication — replaces identical spans across messages with pointers."""

from __future__ import annotations

import hashlib


class DedupBlock:
    """Represents a message content block for deduplication."""

    def __init__(self, index: int, content: str) -> None:
        self.index = index
        self.content = content
        self._hash = hashlib.sha256(content.encode()).hexdigest()[:12]

    @property
    def hash(self) -> str:
        return self._hash

    def __repr__(self) -> str:
        return f"DedupBlock({self.index}, hash={self._hash}, len={len(self.content)})"


def dedup_blocks(blocks: list[DedupBlock]) -> list[DedupBlock]:
    """Deduplicate content spans across blocks.

    Replaces identical content in later blocks with a pointer to the
    earliest occurrence. Priority-monotonic: appending a turn never
    mutates earlier turns.
    """
    seen: dict[str, int] = {}  # hash -> first index
    result = []

    for block in blocks:
        content = block.content
        if not content or not content.strip():
            result.append(block)
            continue

        # Split content into lines for block-level dedup
        lines = content.split("\n")

        # Try multi-line dedup (check for repeated blocks)
        deduped_lines = _dedup_lines(lines, seen, block.index)
        new_content = "\n".join(deduped_lines)

        result.append(DedupBlock(block.index, new_content))

    return result


def _dedup_lines(
    lines: list[str], seen: dict[str, int], current_index: int
) -> list[str]:
    """Deduplicate lines across blocks."""
    # Group consecutive identical lines
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        line_hash = hashlib.sha256(line.encode()).hexdigest()[:12]

        if line_hash in seen and seen[line_hash] < current_index:
            # This line was seen before — replace with pointer
            first_index = seen[line_hash]
            result.append(f"[see block {first_index} for context]")
        else:
            result.append(line)
            if line.strip():
                seen[line_hash] = current_index

        i += 1

    return result
