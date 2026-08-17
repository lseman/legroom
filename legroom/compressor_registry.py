"""Compressor registry and input/output types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompressInput:
    """Input to a compressor."""

    content: str
    content_type: str = "text"
    source_hint: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressOutput:
    """Output from a compressor."""

    compressed: str
    original_token_count: int = 0
    compressed_token_count: int = 0
    strategy: str = "unknown"
    routing_log: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        return self.original_token_count - self.compressed_token_count

    @property
    def compression_ratio(self) -> float:
        if self.original_token_count == 0:
            return 0.0
        return self.compressed_token_count / self.original_token_count
