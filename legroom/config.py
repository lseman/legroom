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
