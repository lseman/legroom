"""Compression configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompressConfig:
    """Configuration for the compression pipeline."""

    optimize: bool = True
    protect_recent: int = 0
    max_output_tokens: int = 0
    max_input_tokens: int = 0
    verbosity_level: int = 2
    # Threshold for token-level retention (ML compressor)
    retention_threshold: float = 0.5
    # Minimum compression ratio to accept (0.0 = always accept)
    min_compression_ratio: float = 0.0
    # Enable/disable specific pipeline phases
    cache_align_enabled: bool = True
    compress_enabled: bool = True
    ccr_enabled: bool = True
    output_shaping: bool = True
    thinking_compact_enabled: bool = False
    cross_turn_dedup_enabled: bool = True
    read_lifecycle_enabled: bool = True
    # Read Lifecycle settings
    compress_stale: bool = True
    compress_superseded: bool = True
    min_read_lifecycle_bytes: int = 50
    # Read Maturation settings
    maturation_enabled: bool = False
    maturation_quiesce_turns: int = 5
    maturation_max_hold_turns: int = 50
    # Salience tracking
    track_salience: bool = True
    # Bias JSON array compression toward items relevant to the latest
    # user message instead of compressing purely on structural redundancy.
    query_aware: bool = True


@dataclass
class CompressResult:
    """Result of a compression operation."""

    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    transforms_applied: list[str]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_salience_before(self) -> float | None:
        """Average message salience before compression."""
        scores = self.metadata.get("salience_scores_before")
        if not scores:
            return None
        return sum(scores) / len(scores)

    @property
    def avg_salience_after(self) -> float | None:
        """Average message salience after compression."""
        scores = self.metadata.get("salience_scores_after")
        if not scores:
            return None
        return sum(scores) / len(scores)

    @property
    def information_preserved(self) -> float | None:
        """Ratio of information preserved (salience_after / salience_before)."""
        before = self.avg_salience_before
        after = self.avg_salience_after
        if before is None or after is None:
            return None
        return round(after / max(before, 0.001), 4)
