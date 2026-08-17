"""High-level compression interface."""

from __future__ import annotations

from typing import Any

from .config import CompressConfig, CompressResult
from .pipeline import TransformPipeline, TransformResult
from .tokenizer import count_tokens_messages


def compress(
    messages: list[dict[str, Any]],
    model: str = "gpt-4o",
    config: CompressConfig | None = None,
    **kwargs: Any,
) -> CompressResult:
    """Compress a list of messages using the default pipeline.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Model name for token counting (affects encoding).
        config: Compression configuration.
        **kwargs: Additional config overrides (protect_recent, optimize, etc.)

    Returns:
        CompressResult with compressed messages and stats.
    """
    cfg = config or CompressConfig()

    # Apply kwargs as overrides
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    pipeline = TransformPipeline(
        compress_enabled=cfg.optimize,
        cache_align_enabled=cfg.cache_align_enabled,
        cross_turn_dedup_enabled=cfg.cross_turn_dedup_enabled,
        thinking_compact_enabled=cfg.thinking_compact_enabled,
        ccr_enabled=cfg.ccr_enabled,
        output_shaping=cfg.output_shaping,
        verbosity_level=cfg.verbosity_level,
    )

    result = pipeline.apply(messages, model=model, config=cfg)

    return CompressResult(
        messages=result.messages,
        tokens_before=result.tokens_before,
        tokens_after=result.tokens_after,
        tokens_saved=result.tokens_before - result.tokens_after,
        transforms_applied=result.transforms_applied,
        warnings=result.warnings,
        metadata=result.metadata,
    )
