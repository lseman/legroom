"""Cross-turn deduplication — replaces identical spans across messages with pointers."""

from __future__ import annotations

import hashlib

# Use MD5 instead of SHA256 for faster hashing (sufficient for dedup)
# MD5 is ~2x faster than SHA256 and collision risk is negligible for this use case
_hash_func = hashlib.md5


def _fast_hash(content: str) -> str:
    """Fast hash function for deduplication."""
    return _hash_func(content.encode()).hexdigest()[:12]


class DedupBlock:
    """Represents a message content block for deduplication."""

    def __init__(self, index: int, content: str) -> None:
        self.index = index
        # Preserve the original content regardless of type.
        # Non-string content (e.g. Anthropic multi-block content) is stored
        # as-is and will be skipped during dedup hashing so it always
        # passes through unchanged.
        self.content = content
        if isinstance(content, str):
            self._hash = _fast_hash(content)
        else:
            self._hash = ""

    @property
    def hash(self) -> str:
        return self._hash

    def __repr__(self) -> str:
        return f"DedupBlock({self.index}, hash={self._hash}, len={len(self.content)})"


# Minimum content size before a whole-block match is worth collapsing —
# guards against wasting a pointer on a short block that happens to recur
# (e.g. a one-line "OK" tool result) where the marker itself would be as
# large as what it's replacing.
_MIN_DEDUP_BLOCK_BYTES = 200


def dedup_blocks(blocks: list[DedupBlock]) -> list[DedupBlock]:
    """Deduplicate content spans across blocks.

    Replaces a later block with a pointer only when its *entire* content is
    byte-for-byte identical to an earlier block's entire content (e.g. the
    same large tool-result dump appearing twice) — not on a per-line basis.
    Per-line hashing is unsafe here: it would collapse ordinary shared
    boilerplate (a common "status": "ok" line, a shared import, a blank
    separator) that recurs across completely unrelated tool calls, and the
    resulting "[see block N]" pointer is never dereferenced by anything
    downstream — the model has no way to resolve it, so any content it
    replaces is gone for good. Whole-block matching only fires on content
    that is actually a full duplicate, so nothing distinct is ever lost.
    Priority-monotonic: appending a turn never mutates earlier turns.
    """
    seen: dict[str, int] = {}  # hash -> first index
    result = []

    for block in blocks:
        content = block.content
        if (
            isinstance(content, str)
            and len(content.encode("utf-8")) >= _MIN_DEDUP_BLOCK_BYTES
            and block.hash in seen
        ):
            first_index = seen[block.hash]
            pointer = f"[identical to block {first_index} — {len(content)} chars omitted]"
            result.append(DedupBlock(block.index, pointer))
            continue

        result.append(block)
        if isinstance(content, str) and content.strip():
            seen.setdefault(block.hash, block.index)

    return result
