"""Read Lifecycle — track file tool operations and compress stale/superseded reads.

A Read becomes STALE when its file is subsequently edited — the content
in context is factually wrong. A Read becomes SUPERSEDED when the same file
is re-Read — the content is redundant. Both are provably safe to replace.

Real-world data from headroom's audit-reads shows:
- 67% stale (file edited after Read)
- 12% superseded (file re-Read later)
- Only 20% are fresh (untouched)

This module replaces stale/superseded Read outputs with compact markers
that include CCR hashes for retrieval.

Read Maturation (not yet implemented): holds fresh Reads out of the cache
while the file is active, then matures them to markers once the file has
been quiet for `quiesce_turns`. This gives an additional 10-15% savings
because the verbatim form never enters the cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Tool names that produce Read outputs
_READ_TOOL_NAMES = frozenset({"Read", "read", "read_file"})

# Tool names that mutate files (make previous Reads stale)
_MUTATING_TOOL_NAMES = frozenset(
    {
        "Edit", "edit", "edit_file",
        "Write", "write", "write_file",
        "MultiEdit", "NotebookEdit", "apply_patch",
    }
)


class ReadState(str, Enum):
    """Lifecycle state of a Read output."""

    FRESH = "fresh"
    STALE = "stale"
    SUPERSEDED = "superseded"


@dataclass
class FileOperation:
    """A single file operation observed in the conversation."""

    msg_index: int
    tool_call_id: str
    tool_name: str
    file_path: str
    operation: str  # "read" | "edit" | "write"
    content_size: int = 0
    read_offset: Optional[int] = None
    read_limit: Optional[int] = None


@dataclass
class ReadClassification:
    """Classification of a single Read output."""

    msg_index: int
    tool_call_id: str
    file_path: str
    state: ReadState
    content_size: int = 0


@dataclass
class ReadLifecycleConfig:
    """Configuration for Read Lifecycle management."""

    enabled: bool = True
    compress_stale: bool = True
    compress_superseded: bool = True
    min_size_bytes: int = 50  # Skip replacing tiny outputs
    protect_recent: int = 0  # Never compress a Read in the last N messages


@dataclass
class ReadLifecycleResult:
    """Output of lifecycle management pass."""

    messages: list[dict[str, Any]]
    reads_total: int = 0
    reads_stale: int = 0
    reads_superseded: int = 0
    reads_fresh: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    transforms_applied: list[str] = field(default_factory=list)
    ccr_hashes: list[str] = field(default_factory=list)


def classify_reads(
    messages: list[dict[str, Any]],
    config: ReadLifecycleConfig,
    compression_store: Optional[Any] = None,
) -> ReadLifecycleResult:
    """Apply read lifecycle management to messages.

    Scans for tool calls (Read, Edit, Write), tracks file operations,
    classifies reads as fresh/stale/superseded, and replaces stale content
    with compact markers.

    Args:
        messages: Conversation messages.
        config: Lifecycle configuration.
        compression_store: Optional CCR store for persisting original content.

    Returns:
        ReadLifecycleResult with replaced messages and stats.
    """
    if not config.enabled or not messages:
        return ReadLifecycleResult(messages=messages)

    # Phase 1: Build tool metadata
    tool_metadata = _build_tool_metadata(messages)

    # Phase 2: Build file operation index
    file_ops = _build_file_operation_index(messages, tool_metadata)

    # Phase 3: Classify each Read. Reads inside the protected tail are never
    # eligible for compression, regardless of stale/superseded status — the
    # model may need their exact content (e.g. as `old_string` for an edit)
    # on the very next turn.
    protected_from = (
        len(messages) - config.protect_recent if config.protect_recent > 0 else None
    )
    classifications = _classify_reads(file_ops, config, protected_from)

    if not classifications:
        return ReadLifecycleResult(
            messages=messages,
            reads_total=0,
        )

    # Phase 4: Apply replacements
    return _apply_lifecycle(messages, classifications, config, compression_store)


def _build_tool_metadata(
    messages: list[dict[str, Any]],
) -> dict[str, tuple[str, str | None, int | None, int | None]]:
    """Build tool_call_id → (tool_name, file_path, offset, limit) mapping.

    Handles both OpenAI and Anthropic message formats.
    """
    metadata: dict[str, tuple[str, str | None, int | None, int | None]] = {}

    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        # OpenAI format: tool_calls array
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id", "")
            func = tc.get("function", {})
            name = func.get("name", "")
            if not tc_id or not name:
                continue

            file_path = None
            offset = None
            limit = None
            try:
                args = json.loads(func.get("arguments", "{}"))
                file_path = args.get("file_path") or args.get("path")
                offset = args.get("offset")
                limit = args.get("limit")
            except (json.JSONDecodeError, TypeError):
                pass

            metadata[tc_id] = (name, file_path, offset, limit)

        # Anthropic format: content blocks with type=tool_use
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tc_id = block.get("id", "")
            name = block.get("name", "")
            if not tc_id or not name:
                continue

            inp = block.get("input", {})
            file_path = None
            offset = None
            limit = None
            if isinstance(inp, dict):
                file_path = inp.get("file_path") or inp.get("path")
                offset = inp.get("offset")
                limit = inp.get("limit")

            metadata[tc_id] = (name, file_path, offset, limit)

    return metadata


def _build_file_operation_index(
    messages: list[dict[str, Any]],
    tool_metadata: dict[str, tuple[str, str | None, int | None, int | None]],
) -> dict[str, list[FileOperation]]:
    """Build file_path → [FileOperation] index in a single pass."""
    file_ops: dict[str, list[FileOperation]] = defaultdict(list)

    for tc_id, (name, file_path, offset, limit) in tool_metadata.items():
        if not file_path:
            continue

        if name in _READ_TOOL_NAMES:
            operation = "read"
        elif name in _MUTATING_TOOL_NAMES:
            operation = "edit"
        else:
            continue

        # Find the message index containing this tool_call
        msg_idx = _find_tool_call_msg_index(messages, tc_id)
        if msg_idx is None:
            continue

        file_ops[file_path].append(
            FileOperation(
                msg_index=msg_idx,
                tool_call_id=tc_id,
                tool_name=name,
                file_path=file_path,
                operation=operation,
                read_offset=offset if operation == "read" else None,
                read_limit=limit if operation == "read" else None,
            )
        )

    return dict(file_ops)


def _find_tool_call_msg_index(
    messages: list[dict[str, Any]],
    tool_call_id: str,
) -> int | None:
    """Find the message index containing a specific tool_call_id."""
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        # OpenAI format
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("id") == tool_call_id:
                return i

        # Anthropic format
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("id") == tool_call_id
                ):
                    return i

    return None


def _read_covers(later: FileOperation, earlier: FileOperation) -> bool:
    """Check if `later` read fully covers the line range of `earlier`.

    A full-file read (no offset/limit) covers everything.
    A partial read only covers another partial if its range is a superset.
    """
    # Full-file read supersedes anything
    if later.read_offset is None and later.read_limit is None:
        return True

    # If the earlier was a full-file read, a partial can't cover it
    if earlier.read_offset is None and earlier.read_limit is None:
        return False

    # Both are partial reads — check range containment
    later_start = later.read_offset or 0
    later_end = later_start + (later.read_limit or 2000)
    earlier_start = earlier.read_offset or 0
    earlier_end = earlier_start + (earlier.read_limit or 2000)

    return later_start <= earlier_start and later_end >= earlier_end


def _classify_reads(
    file_ops: dict[str, list[FileOperation]],
    config: ReadLifecycleConfig,
    protected_from: Optional[int] = None,
) -> list[ReadClassification]:
    """Classify each Read as fresh, stale, or superseded.

    protected_from: reads at or after this message index are always FRESH,
    win over stale/superseded status.
    """
    classifications: list[ReadClassification] = []

    for file_path, ops in file_ops.items():
        reads = [op for op in ops if op.operation == "read"]
        edits = [op for op in ops if op.operation == "edit"]

        if not reads:
            continue

        for read_op in reads:
            if protected_from is not None and read_op.msg_index >= protected_from:
                classifications.append(
                    ReadClassification(
                        msg_index=read_op.msg_index,
                        tool_call_id=read_op.tool_call_id,
                        file_path=file_path,
                        state=ReadState.FRESH,
                        content_size=read_op.content_size,
                    )
                )
                continue

            # Check stale: any edit/write after this read?
            is_stale = config.compress_stale and any(
                e.msg_index > read_op.msg_index for e in edits
            )

            # Check superseded: any later read that FULLY COVERS this read's range?
            is_superseded = config.compress_superseded and any(
                r.msg_index > read_op.msg_index and _read_covers(r, read_op)
                for r in reads
            )

            if is_stale:
                state = ReadState.STALE
            elif is_superseded:
                state = ReadState.SUPERSEDED
            else:
                state = ReadState.FRESH

            classifications.append(
                ReadClassification(
                    msg_index=read_op.msg_index,
                    tool_call_id=read_op.tool_call_id,
                    file_path=file_path,
                    state=state,
                    content_size=read_op.content_size,
                )
            )

    return classifications


def _apply_lifecycle(
    messages: list[dict[str, Any]],
    classifications: list[ReadClassification],
    config: ReadLifecycleConfig,
    store: Optional[Any],
) -> ReadLifecycleResult:
    """Replace stale/superseded Read content with markers."""
    # Build lookup: tool_call_id → classification (for non-fresh reads)
    replacements: dict[str, ReadClassification] = {
        c.tool_call_id: c for c in classifications if c.state != ReadState.FRESH
    }

    if not replacements:
        counts = {ReadState.FRESH: len(classifications), ReadState.STALE: 0, ReadState.SUPERSEDED: 0}
        return ReadLifecycleResult(
            messages=messages,
            reads_total=len(classifications),
            reads_fresh=counts[ReadState.FRESH],
            reads_stale=counts[ReadState.STALE],
            reads_superseded=counts[ReadState.SUPERSEDED],
        )

    result_messages: list[dict[str, Any]] = []
    transforms: list[str] = []
    ccr_hashes: list[str] = []
    bytes_before = 0
    bytes_after = 0
    counts = {ReadState.FRESH: 0, ReadState.STALE: 0, ReadState.SUPERSEDED: 0}

    for c in classifications:
        counts[c.state] += 1

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # OpenAI format: role=tool with tool_call_id
        if role == "tool":
            tc_id = msg.get("tool_call_id", "")
            classification = replacements.get(tc_id)
            if classification and isinstance(content, str):
                replaced, marker, ccr_hash = _replace_content(
                    content, classification, config, store
                )
                if replaced:
                    result_messages.append({**msg, "content": marker})
                    transforms.append(
                        f"read_lifecycle:{classification.state.value}:{classification.file_path}"
                    )
                    if ccr_hash:
                        ccr_hashes.append(ccr_hash)
                    bytes_before += len(content.encode("utf-8"))
                    bytes_after += len(marker.encode("utf-8"))
                    continue

        # Anthropic format: content blocks list
        if isinstance(content, list):
            new_blocks, block_replaced = _process_anthropic_blocks(
                content, replacements, transforms, ccr_hashes, config, store
            )
            if block_replaced:
                result_messages.append({**msg, "content": new_blocks})
                continue

        result_messages.append(msg)

    return ReadLifecycleResult(
        messages=result_messages,
        reads_total=len(classifications),
        reads_stale=counts[ReadState.STALE],
        reads_superseded=counts[ReadState.SUPERSEDED],
        reads_fresh=counts[ReadState.FRESH],
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        transforms_applied=transforms,
        ccr_hashes=ccr_hashes,
    )


def _process_anthropic_blocks(
    content_blocks: list[Any],
    replacements: dict[str, ReadClassification],
    transforms: list[str],
    ccr_hashes: list[str],
    config: ReadLifecycleConfig,
    store: Optional[Any],
) -> tuple[list[Any], bool]:
    """Process Anthropic-format content blocks for lifecycle replacement."""
    new_blocks = []
    any_replaced = False

    for block in content_blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            new_blocks.append(block)
            continue

        tc_id = block.get("tool_use_id", "")
        classification = replacements.get(tc_id)
        tool_content = block.get("content", "")

        if classification and isinstance(tool_content, str):
            replaced, marker, ccr_hash = _replace_content(
                tool_content, classification, config, store
            )
            if replaced:
                new_blocks.append({**block, "content": marker})
                transforms.append(
                    f"read_lifecycle:{classification.state.value}:{classification.file_path}"
                )
                if ccr_hash:
                    ccr_hashes.append(ccr_hash)
                any_replaced = True
                continue

        new_blocks.append(block)

    return new_blocks, any_replaced


def _replace_content(
    content: str,
    classification: ReadClassification,
    config: ReadLifecycleConfig,
    store: Optional[Any],
) -> tuple[bool, str, Optional[str]]:
    """Replace Read content with a lifecycle marker."""
    content_bytes = len(content.encode("utf-8"))

    # Skip tiny outputs
    if content_bytes < config.min_size_bytes:
        return False, content, None

    # Best-effort CCR persistence. Hash length must match CompressionStore's
    # own default (16 chars) so the key quoted to the model always matches
    # the key actually stored, whether or not a store is wired in.
    ccr_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    if store is not None:
        try:
            ccr_hash = store.store(
                original=content,
                compressed="",
                tool_name="Read",
                tool_call_id=classification.tool_call_id,
                compression_strategy=f"read_lifecycle:{classification.state.value}",
                explicit_hash=ccr_hash,
            )
        except Exception:
            pass  # Storage failure must not break compression

    file_display = classification.file_path or "unknown"

    if classification.state == ReadState.STALE:
        marker = (
            f"[Read content stale: {file_display} was modified after this read — "
            f"re-read for current content. "
            f"Retrieve original: hash={ccr_hash}]"
        )
    else:
        marker = (
            f"[Read content superseded: {file_display} was re-read later — "
            f"re-read if needed. "
            f"Retrieve original: hash={ccr_hash}]"
        )

    return True, marker, ccr_hash
