"""Transform pipeline — orchestrates compression transforms."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

from .config import CompressConfig, CompressResult
from .tokenizer import count_tokens_messages
from .compressor_registry import _compute_salience
from .content_router import ContentRouter
from .smart_crusher import SmartCrusher, SmartCrusherConfig
from .adaptive_sizer import compute_optimal_k
from .cross_turn_dedup import DedupBlock, dedup_blocks
from .recursive_json import route_embedded_json
from .lossless_compaction import compact_lossless
from .read_lifecycle import classify_reads, ReadLifecycleConfig, ReadLifecycleResult
from .output.shaper import OutputShaper
from .ccr.tool_injection import CCRToolInjector
from .query_relevance import latest_query_terms

logger = logging.getLogger(__name__)


class ContentHashCache:
    """LRU cache keyed by content hash → compressed result.

    Used by CompressPhase to avoid re-compressing identical content
    across messages and across pipeline invocations.
    """

    def __init__(self, maxsize: int = 512) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, content: str) -> str | None:
        h = hashlib.sha256(content.encode()).hexdigest()
        if h in self._cache:
            self._cache.move_to_end(h)
            self._hits += 1
            return self._cache[h]
        self._misses += 1
        return None

    def put(self, content: str, compressed: str) -> None:
        h = hashlib.sha256(content.encode()).hexdigest()
        if h in self._cache:
            self._cache.move_to_end(h)
            self._cache[h] = compressed
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[h] = compressed

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def ratio(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


class CompressPhase:
    """Phase that applies content compression."""

    def __init__(
        self,
        cache: ContentHashCache | None = None,
        adaptive_sizing: bool = False,
        size_bias: float = 1.0,
    ) -> None:
        self._router = ContentRouter(adaptive_sizing=adaptive_sizing, size_bias=size_bias)
        self._cache = cache or ContentHashCache()

    def apply(
        self, messages: list[dict[str, Any]], config: CompressConfig, model: str = "gpt-4o"
    ) -> list[dict[str, Any]]:
        """Apply compression to messages with cache lookups."""
        query_terms = latest_query_terms(messages) if getattr(config, "query_aware", True) else set()

        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Check compression cache (before lossless, on raw content).
                # Skipped under query-aware compression — see ContentRouter.compress.
                if not query_terms:
                    cached = self._cache.get(content)
                    if cached is not None:
                        result.append({**msg, "content": cached})
                        continue

                original_content = content

                # Try lossless compaction first
                compacted = compact_lossless(content, "text")
                if compacted.transforms_applied:
                    content = compacted.compressed

                # Route through content router with model
                compressed = self._router.compress(
                    content, source_hint="text", model=model, query_terms=query_terms
                )
                if compressed and compressed.tokens_saved > 0:
                    content = compressed.compressed

                # Try recursive JSON routing
                json_result = route_embedded_json(
                    content,
                    lambda span: self._router.compress(
                        span, source_hint="text", model=model, query_terms=query_terms
                    ),
                )
                if json_result is not None:
                    content = json_result

                # Cache the result keyed by the original raw content
                if content != original_content and not query_terms:
                    self._cache.put(original_content, content)

            result.append({**msg, "content": content})
        return result


class ThinkingCompactor:
    """Strips thinking/reasoning blocks from messages."""

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip thinking blocks."""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Remove <think>...</think> blocks
                import re
                content = re.sub(
                    r"<think>.*?</think>",
                    "",
                    content,
                    flags=re.DOTALL,
                )
                content = content.strip()
            result.append({**msg, "content": content})
        return result


class TransformResult:
    """Result from the transform pipeline."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        tokens_before: int,
        tokens_after: int,
        transforms_applied: list[str],
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.messages = messages
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.transforms_applied = transforms_applied
        self.warnings = warnings or []
        self.metadata = metadata or {}


class TransformPipeline:
    """Full compression pipeline orchestrating all transforms."""

    def __init__(
        self,
        compress_enabled: bool = True,
        cache_align_enabled: bool = True,
        cross_turn_dedup_enabled: bool = True,
        thinking_compact_enabled: bool = False,
        ccr_enabled: bool = True,
        output_shaping: bool = True,
        verbosity_level: int = 2,
        adaptive_sizing: bool = False,
        size_bias: float = 1.0,
    ) -> None:
        self.compress_enabled = compress_enabled
        self.cache_align_enabled = cache_align_enabled
        self.cross_turn_dedup_enabled = cross_turn_dedup_enabled
        self.thinking_compact_enabled = thinking_compact_enabled
        self.ccr_enabled = ccr_enabled
        self.output_shaping = output_shaping
        self.verbosity_level = verbosity_level

        self.cache_aligner = CacheAligner(enabled=cache_align_enabled)
        self.compressor = CompressPhase(adaptive_sizing=adaptive_sizing, size_bias=size_bias)
        self.thinking_compactor = ThinkingCompactor()
        self.output_shaper = OutputShaper(verbosity_level=verbosity_level)
        self._applied_transforms: list[str] = []
        self._warnings: list[str] = []

    def apply(
        self,
        messages: list[dict[str, Any]],
        model: str = "gpt-4o",
        config: CompressConfig | None = None,
    ) -> TransformResult:
        """Run the full compression pipeline on messages."""
        if not messages:
            return TransformResult(messages=[], tokens_before=0, tokens_after=0, transforms_applied=[])

        tokens_before = count_tokens_messages(messages, model)
        config = config or CompressConfig()

        # Compute salience scores before compression
        if getattr(config, "track_salience", True):
            salience_scores_before = [
                _compute_salience(msg.get("content", "")) for msg in messages
            ]
        else:
            salience_scores_before = None
        self._applied_transforms = []
        self._warnings = []

        current_messages = list(messages)

        # Phase 1: Output shaping (pass protect_recent to avoid modifying protected messages)
        if self.output_shaping:
            self.output_shaper.protect_recent = config.protect_recent
            current_messages = self.output_shaper.apply(current_messages)
            self._applied_transforms.append("output_shaper")

        # Phase 2: Cache alignment
        if self.cache_align_enabled:
            try:
                current_messages = self.cache_aligner.apply(current_messages)
                self._warnings.extend(self.cache_aligner.get_warnings())
                self._applied_transforms.append("cache_aligner")
            except Exception as e:
                logger.warning(f"CacheAligner failed: {e}")

        # Phase 2.5: Cross-turn dedup
        if self.cross_turn_dedup_enabled:
            try:
                blocks = [
                    DedupBlock(i, msg.get("content", ""))
                    for i, msg in enumerate(current_messages)
                ]
                deduped = dedup_blocks(blocks)
                for i, block in enumerate(deduped):
                    if i < len(current_messages):
                        current_messages[i] = {
                            **current_messages[i],
                            "content": block.content,
                        }
                self._applied_transforms.append("cross_turn_dedup")
            except Exception as e:
                logger.warning(f"CrossTurnDedup failed: {e}")

        # Phase 2.6: Read Lifecycle — compress stale/superseded Read outputs
        if getattr(config, "read_lifecycle_enabled", True):
            try:
                lifecycle_config = ReadLifecycleConfig(
                    enabled=True,
                    compress_stale=getattr(config, "compress_stale", True),
                    compress_superseded=getattr(config, "compress_superseded", True),
                    min_size_bytes=getattr(config, "min_read_lifecycle_bytes", 50),
                )
                lifecycle_result = classify_reads(
                    current_messages, lifecycle_config, None
                )
                current_messages = lifecycle_result.messages
                if lifecycle_result.reads_stale > 0 or lifecycle_result.reads_superseded > 0:
                    self._applied_transforms.append("read_lifecycle")
                    self._warnings.append(
                        f"read_lifecycle: {lifecycle_result.reads_stale} stale, "
                        f"{lifecycle_result.reads_superseded} superseded reads compressed"
                    )
            except Exception as e:
                logger.warning(f"ReadLifecycle failed: {e}")

        # Phase 3: Compression
        if self.compress_enabled:
            try:
                current_messages = self.compressor.apply(current_messages, config, model=model)
                self._applied_transforms.append("compress")
            except Exception as e:
                logger.warning(f"Compress phase failed: {e}")
                self._warnings.append(f"Compression failed: {e}")

        # Phase 4: Thinking compaction
        if self.thinking_compact_enabled:
            try:
                current_messages = self.thinking_compactor.apply(current_messages)
                self._applied_transforms.append("thinking_compactor")
            except Exception as e:
                logger.warning(f"ThinkingCompactor failed: {e}")

        # Phase 5: CCR tool injection
        if self.ccr_enabled:
            injector = CCRToolInjector(provider="anthropic")
            injector.scan_for_markers(current_messages)
            if injector.has_compressed_content:
                current_messages = injector.inject_system_instructions(current_messages)
                self._applied_transforms.append("ccr_tool_injection")

        tokens_after = count_tokens_messages(current_messages, model)

        # Inflation guard: if the pipeline made things worse, revert to the
        # original messages rather than ship an inflated result. Independent
        # of salience tracking, which is purely for the metadata below.
        if tokens_after > tokens_before:
            logger.warning("Compression inflated tokens; reverting to original")
            return TransformResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=["inflation_guard"],
                warnings=self._warnings,
                metadata={
                    "cache_metrics": self.cache_aligner.get_metrics(),
                    "salience_scores_before": salience_scores_before,
                },
            )

        # Compute salience scores after compression
        if getattr(config, "track_salience", True):
            salience_scores_after = [
                _compute_salience(msg.get("content", "")) for msg in current_messages
            ]
        else:
            salience_scores_after = None

        metadata = {"cache_metrics": self.cache_aligner.get_metrics()}
        if salience_scores_before is not None:
            metadata["salience_scores_before"] = salience_scores_before
        if salience_scores_after is not None:
            metadata["salience_scores_after"] = salience_scores_after

        return TransformResult(
            messages=current_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=self._applied_transforms,
            warnings=self._warnings,
            metadata=metadata,
        )


def create_default_pipeline(
    compress_enabled: bool = True,
    cache_align_enabled: bool = True,
) -> TransformPipeline:
    """Create a default TransformPipeline with sensible settings."""
    return TransformPipeline(
        compress_enabled=compress_enabled,
        cache_align_enabled=cache_align_enabled,
        ccr_enabled=True,
        output_shaping=True,
        verbosity_level=2,
    )


class CacheAligner:
    """Detects volatile content that busts KV cache alignment."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._warnings: list[str] = []
        self._metrics: dict[str, Any] = {"detected": 0, "aligned": 0}

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply cache alignment — replace volatile tokens with placeholders."""
        import re
        import uuid

        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Replace UUIDs with fixed placeholder
                uuid_pattern = re.compile(
                    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
                )
                if uuid_pattern.search(content):
                    content = uuid_pattern.sub("[UUID_PLACEHOLDER]", content)
                    self._metrics["detected"] += 1

                # Replace ISO timestamps
                iso_pattern = re.compile(
                    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
                )
                if iso_pattern.search(content):
                    content = iso_pattern.sub("[TIMESTAMP_PLACEHOLDER]", content)
                    self._metrics["detected"] += 1

                self._metrics["aligned"] += 1
            result.append({**msg, "content": content})
        return result

    def get_warnings(self) -> list[str]:
        return self._warnings

    def get_metrics(self) -> dict[str, Any]:
        return self._metrics.copy()
