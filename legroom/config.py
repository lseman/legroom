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
