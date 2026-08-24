"""Model-specific compression profiles.

Different models have different context utilization patterns and
resilience to compressed context. This module defines profiles that
tune compression aggressiveness per model.

Profiles:
- claude: handles long context well, more resilient to compression
- gpt-4o: moderate resilience, standard compression
- gpt-4o-mini: less context capacity, lighter compression
- gpt-4-turbo: similar to gpt-4o
- gpt-3.5-turbo: limited context, conservative compression
- default: fallback profile for unknown models
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CompressConfig

# Default profile used when no model-specific profile matches.
_DEFAULT_PROFILE = "default"


@dataclass(frozen=True)
class CompressionProfile:
    """Compression profile for a specific model.

    Attributes:
        protect_recent: Number of recent turns to protect from compression.
            Models with better context retention (Claude) can protect fewer.
        cache_align_enabled: Whether to normalize UUIDs/timestamps.
            Always True — this is free and helps all models.
        thinking_compact_enabled: Strip thinking/reasoning blocks.
            Always True — reasoning blocks are never useful across turns.
        adaptive_sizing: Use Kneedle+SimHash for JSON arrays.
            Always True — already the default.
        min_compression_ratio: Minimum ratio to accept compression.
            Higher for models less resilient to compression.
        size_bias: Bias for adaptive JSON sizing (lower = more aggressive).
    """

    protect_recent: int = 3
    cache_align_enabled: bool = True
    thinking_compact_enabled: bool = True
    adaptive_sizing: bool = True
    min_compression_ratio: float = 0.15
    size_bias: float = 1.0


# Profile definitions.
# Claude handles long context better and is more resilient to compression,
# so it can use slightly more aggressive settings.
# GPT-4o-mini has less context capacity, so we're more conservative.
MODEL_PROFILES: dict[str, CompressionProfile] = {
    # OpenAI models
    "gpt-4o": CompressionProfile(
        protect_recent=3,
        min_compression_ratio=0.15,
    ),
    "gpt-4o-mini": CompressionProfile(
        protect_recent=3,
        min_compression_ratio=0.20,  # More conservative
    ),
    "gpt-4-turbo": CompressionProfile(
        protect_recent=3,
        min_compression_ratio=0.15,
    ),
    "gpt-4": CompressionProfile(
        protect_recent=4,  # 8K context, protect more
        min_compression_ratio=0.20,
    ),
    "gpt-3.5-turbo": CompressionProfile(
        protect_recent=4,  # 16K context, protect more
        min_compression_ratio=0.20,
    ),
    # Anthropic models
    "claude-3-5-sonnet": CompressionProfile(
        protect_recent=3,
        min_compression_ratio=0.15,
    ),
    "claude-3-sonnet": CompressionProfile(
        protect_recent=3,
        min_compression_ratio=0.15,
    ),
    "claude-3-opus": CompressionProfile(
        protect_recent=3,
        min_compression_ratio=0.15,
    ),
    "claude-3-haiku": CompressionProfile(
        protect_recent=3,
        min_compression_ratio=0.15,
    ),
    "claude-2.1": CompressionProfile(
        protect_recent=3,
        min_compression_ratio=0.15,
    ),
    "claude-2": CompressionProfile(
        protect_recent=3,
        min_compression_ratio=0.15,
    ),
}


def get_profile(model: str) -> CompressionProfile:
    """Get the compression profile for a model.

    Falls back to the default profile for unknown models.
    """
    # Exact match first
    if model in MODEL_PROFILES:
        return MODEL_PROFILES[model]

    # Prefix match (e.g. "claude-3-5-sonnet-20241022" → claude-3-5-sonnet)
    for prefix, profile in MODEL_PROFILES.items():
        if model.startswith(prefix):
            return profile

    return MODEL_PROFILES[_DEFAULT_PROFILE]


def apply_profile(model: str, config: CompressConfig) -> CompressConfig:
    """Apply model-specific profile settings to a config.

    Only applies settings that are model-dependent (protect_recent,
    min_compression_ratio, size_bias). All other settings are preserved
    from the user's config — this is a soft default, not a hard override.
    """
    profile = get_profile(model)

    # Only override model-specific knobs; everything else stays as-is.
    return CompressConfig(
        **{
            **config.__dict__,
            "protect_recent": profile.protect_recent,
            "min_compression_ratio": profile.min_compression_ratio,
            "size_bias": profile.size_bias,
        }
    )
