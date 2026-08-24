"""High-level compression interface."""

from __future__ import annotations

from typing import Any

from .ccr.compression_store import CompressionStore
from .config import CompressConfig, CompressResult
from .pipeline import TransformPipeline
from .tokenizer import count_tokens_messages


def compress(
    messages: list[dict[str, Any]],
    model: str = "gpt-4o",
    config: CompressConfig | None = None,
    compression_store: CompressionStore | None = None,
    **kwargs: Any,
) -> CompressResult:
    """Compress a list of messages using the default pipeline.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Model name for token counting (affects encoding).
        config: Compression configuration.
        compression_store: Optional CCR store for persisting/retrieving
            content replaced by the read lifecycle phase. Without one,
            replaced content cannot actually be retrieved later.
        **kwargs: Additional config overrides (protect_recent, optimize, etc.)

    Returns:
        CompressResult with compressed messages and stats.
    """
    cfg = config or CompressConfig()

    # Apply kwargs as overrides
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    # ``optimize=False`` is the public passthrough contract: no phase may
    # rewrite caller-owned messages or inject steering/tool instructions.
    if not cfg.optimize:
        tokens = count_tokens_messages(messages, model)
        return CompressResult(
            messages=messages,
            tokens_before=tokens,
            tokens_after=tokens,
            tokens_saved=0,
            transforms_applied=[],
        )

    pipeline = TransformPipeline(
        compress_enabled=cfg.compress_enabled,
        cache_align_enabled=cfg.cache_align_enabled,
        cross_turn_dedup_enabled=cfg.cross_turn_dedup_enabled,
        thinking_compact_enabled=cfg.thinking_compact_enabled,
        ccr_enabled=cfg.ccr_enabled,
        output_shaping=cfg.output_shaping,
        verbosity_level=cfg.verbosity_level,
        adaptive_sizing=cfg.adaptive_sizing,
        size_bias=cfg.size_bias,
        ml_compress_enabled=cfg.ml_compress_enabled,
        ml_model_path=cfg.ml_model_path,
        ml_tokenizer_path=cfg.ml_tokenizer_path,
        ml_retention_threshold=cfg.retention_threshold,
        ml_min_compression_ratio=cfg.min_compression_ratio,
        compression_store=compression_store,
        semantic_dedup_enabled=cfg.semantic_dedup_enabled,
        semantic_dedup_threshold=cfg.semantic_dedup_threshold,
        semantic_dedup_model_path=cfg.semantic_dedup_model_path,
        semantic_dedup_config_path=cfg.semantic_dedup_config_path,
        semantic_dedup_vocab_path=cfg.semantic_dedup_vocab_path,
        strict=cfg.strict,
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
