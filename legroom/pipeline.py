"""Transform pipeline — orchestrates compression transforms."""

from __future__ import annotations

import logging
from typing import Any

from .config import CompressConfig, CompressResult
from .tokenizer import count_tokens_messages
from .content_router import ContentRouter, CompressionCache
from .smart_crusher import SmartCrusher, SmartCrusherConfig
from .adaptive_sizer import compute_optimal_k
from .cross_turn_dedup import DedupBlock, dedup_blocks
from .recursive_json import route_embedded_json
from .lossless_compaction import compact_lossless
from .output.shaper import OutputShaper
from .ccr.tool_injection import CCRToolInjector

logger = logging.getLogger(__name__)


class CompressPhase:
    """Phase that applies content compression."""

    def __init__(self) -> None:
        self._router = ContentRouter()

    def apply(
        self, messages: list[dict[str, Any]], config: CompressConfig
    ) -> list[dict[str, Any]]:
        """Apply compression to messages."""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Try lossless compaction first
                compacted = compact_lossless(content, "text")
                if compacted.transforms_applied:
                    content = compacted.compressed

                # Route through content router
                compressed = self._router.compress(content)
                if compressed and compressed.tokens_saved > 0:
                    content = compressed.compressed

                # Try recursive JSON routing
                json_result = route_embedded_json(
                    content,
                    lambda span: self._router.compress(span),
                )
                if json_result is not None:
                    content = json_result

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
    ) -> None:
        self.compress_enabled = compress_enabled
        self.cache_align_enabled = cache_align_enabled
        self.cross_turn_dedup_enabled = cross_turn_dedup_enabled
        self.thinking_compact_enabled = thinking_compact_enabled
        self.ccr_enabled = ccr_enabled
        self.output_shaping = output_shaping
        self.verbosity_level = verbosity_level

        self.cache_aligner = CacheAligner(enabled=cache_align_enabled)
        self.compressor = CompressPhase()
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

        # Phase 3: Compression
        if self.compress_enabled:
            try:
                current_messages = self.compressor.apply(current_messages, config)
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

        if tokens_after > tokens_before:
            logger.warning("Compression inflated tokens; reverting to original")
            return TransformResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                transforms_applied=["inflation_guard"],
                warnings=self._warnings,
            )

        return TransformResult(
            messages=current_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=self._applied_transforms,
            warnings=self._warnings,
            metadata={"cache_metrics": self.cache_aligner.get_metrics()},
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
