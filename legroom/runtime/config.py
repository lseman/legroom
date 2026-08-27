"""Compression configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class CompressConfig:
    """Configuration for the compression pipeline."""

    optimize: bool = True
    # Number of recent turns to protect from compression.
    # The model's immediate context is the most critical for reasoning.
    protect_recent: int = 3
    verbosity_level: int = 2
    # Threshold for token-level retention (ML compressor)
    retention_threshold: float = 0.5
    # Minimum compression ratio to accept (0.0 = always accept).
    # Values below 0.15 are usually noise removal, not real compression.
    min_compression_ratio: float = 0.15
    # Enable/disable specific pipeline phases
    # Destructive normalization of volatile values can improve stable-prefix
    # cache reuse, but timestamps, UUIDs, and similar values may be task
    # evidence.  Keep it explicitly opt-in.
    cache_align_enabled: bool = False
    compress_enabled: bool = True
    ccr_enabled: bool = True
    output_shaping: bool = False
    thinking_compact_enabled: bool = True
    cross_turn_dedup_enabled: bool = True
    read_lifecycle_enabled: bool = True
    # Read Lifecycle settings
    compress_stale: bool = True
    compress_superseded: bool = True
    min_read_lifecycle_bytes: int = 50
    # Raise phase failures instead of falling back. Useful in tests and audits;
    # the proxy remains fail-open by default.
    strict: bool = False
    # Salience tracking
    track_salience: bool = True
    # Bias JSON array compression toward items relevant to the latest
    # user message instead of compressing purely on structural redundancy.
    query_aware: bool = True
    # Use compute_optimal_k (SimHash near-dup + Kneedle) to skip summarizing
    # JSON array groups that turn out to be too diverse to benefit from it.
    adaptive_sizing: bool = True
    size_bias: float = 1.0
    # Opt-in ML token-retention compression (Kompress-v2-base) for plain-text
    # content. Lossy and requires `pip install legroom[ml]` plus the model
    # files (see README) — falls back to the lossless TextCompressor when
    # the optional deps or model aren't available. Off by default.
    ml_compress_enabled: bool = False
    ml_model_path: str | None = None
    ml_tokenizer_path: str | None = None
    # Embedding-based semantic cross-turn dedup — detects paraphrased/
    # rephrased content across messages. Requires ONNX model files.
    # Off by default (opt-in) to avoid model download overhead.
    semantic_dedup_enabled: bool = False
    semantic_dedup_threshold: float = 0.85
    semantic_dedup_model_path: str | None = None
    semantic_dedup_config_path: str | None = None
    semantic_dedup_vocab_path: str | None = None
    # KV cache optimization — prefix deduplication across messages.
    # Finds common prefixes across messages and replaces later occurrences
    # with compact pointers, reducing *token count*. No external deps.
    # Off by default (opt-in) — adds a pipeline phase.
    #
    # This rewrites the shared prefix text itself, which breaks the
    # byte-identical prefix match a real inference-server KV cache (e.g.
    # llama.cpp) needs to reuse cached key/value state. It is automatically
    # skipped when ``backend="llama_cpp"`` regardless of this flag — use
    # ``cache_align_enabled`` instead for that backend, which only strips
    # volatile values (UUIDs/timestamps) rather than rewriting shared prefixes.
    kv_cache_optimization_enabled: bool = False
    kv_cache_min_prefix_bytes: int = 100
    kv_cache_min_occurrences: int = 2
    # Stable prefix cache — decomposes the prompt into a stable system-prefix
    # and a variable conversation tail. The compressed prefix is cached so that
    # every request sharing the same system prompt / tools gets an identical
    # tokenized prefix, which llama.cpp's server-side KV cache can match on.
    # Auto-enabled for llama_cpp backend.
    stable_prefix_cache_enabled: bool = False
    stable_prefix_cache_maxsize: int = 64
    # Target inference backend. Controls which KV-cache-affecting phases are
    # safe to run: "openai" (default) is a stateless API with no client-visible
    # KV cache, so prefix rewriting is harmless. "llama_cpp" targets a server
    # with a real, prefix-matched KV cache, so phases that would break prefix
    # identity are disabled regardless of their individual enabled flags.
    backend: Literal["openai", "llama_cpp"] = "openai"
    # Model profiles are named presets, not implicit overrides. Enabling one
    # deliberately replaces its model-dependent knobs; ordinary user config
    # always wins by leaving this disabled.
    use_model_profile: bool = False
    # Protect trusted instructions, exact tool-call structure, unknown provider
    # messages, and the current turn according to the typed IR risk labels.
    risk_policy_enabled: bool = True
    # Phase names disabled by an external calibration controller.
    disabled_phases: tuple[str, ...] = ()


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
