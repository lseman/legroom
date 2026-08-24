"""Transform pipeline — orchestrates compression transforms."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import OrderedDict
from copy import deepcopy
from typing import Any

from .ccr.compression_store import CompressionStore
from .ccr.tool_injection import CCRToolInjector
from .compressors.compressor_registry import _compute_salience
from .compressors.content_detector import ContentDetector
from .compressors.content_router import ContentRouter
from .compressors.lossless_compaction import compact_lossless
from .compressors.recursive_json import route_embedded_json
from .compressors.semantic_dedup import SemanticDedup, SemanticDedupResult
from .config import CompressConfig
from .cross_turn_dedup import DedupBlock, dedup_blocks
from .output.shaper import OutputShaper
from .query_relevance import latest_query_terms
from .read_lifecycle import ReadLifecycleConfig, classify_reads
from .tokenizer import count_tokens_messages

logger = logging.getLogger(__name__)


def _snapshot_read_results(
    messages: list[dict[str, Any]], tool_call_ids: set[str]
) -> dict[str, Any]:
    """Capture fresh Read result payloads so later phases cannot rewrite them."""
    snapshots: dict[str, Any] = {}
    for msg in messages:
        tool_call_id = msg.get("tool_call_id")
        if msg.get("role") == "tool" and tool_call_id in tool_call_ids:
            snapshots[tool_call_id] = deepcopy(msg.get("content"))

        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if tool_use_id in tool_call_ids:
                snapshots[tool_use_id] = deepcopy(block.get("content"))
    return snapshots


def _restore_read_results(
    messages: list[dict[str, Any]], snapshots: dict[str, Any]
) -> list[dict[str, Any]]:
    """Restore byte-faithful fresh Read payloads after a transform phase."""
    if not snapshots:
        return messages

    restored: list[dict[str, Any]] = []
    for msg in messages:
        tool_call_id = msg.get("tool_call_id")
        if msg.get("role") == "tool" and tool_call_id in snapshots:
            restored.append({**msg, "content": deepcopy(snapshots[tool_call_id])})
            continue

        content = msg.get("content")
        if isinstance(content, list):
            blocks: list[Any] = []
            changed = False
            for block in content:
                if isinstance(block, dict) and block.get("tool_use_id") in snapshots:
                    blocks.append({**block, "content": deepcopy(snapshots[block["tool_use_id"]])})
                    changed = True
                else:
                    blocks.append(block)
            if changed:
                restored.append({**msg, "content": blocks})
                continue
        restored.append(msg)
    return restored


class ContentHashCache:
    """LRU cache keyed by content hash → compressed result.

    Used by CompressPhase to avoid re-compressing identical content within
    one pipeline invocation. Cross-request reuse belongs to the proxy cache,
    whose key also includes protocol, model, mode, and policy.

    Default size is 2048 entries for better hit rates in proxy mode
    where the same tool outputs repeat frequently.
    """

    def __init__(self, maxsize: int = 2048) -> None:
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
        ml_compress_enabled: bool = False,
        ml_model_path: str | None = None,
        ml_tokenizer_path: str | None = None,
        ml_retention_threshold: float = 0.5,
        ml_min_compression_ratio: float = 0.1,
    ) -> None:
        self._router = ContentRouter(
            adaptive_sizing=adaptive_sizing,
            size_bias=size_bias,
            ml_compress_enabled=ml_compress_enabled,
            ml_model_path=ml_model_path,
            ml_tokenizer_path=ml_tokenizer_path,
            ml_retention_threshold=ml_retention_threshold,
            ml_min_compression_ratio=ml_min_compression_ratio,
        )
        self._cache = cache or ContentHashCache()
        self._detector = ContentDetector()

    def apply(
        self, messages: list[dict[str, Any]], config: CompressConfig, model: str = "gpt-4o"
    ) -> list[dict[str, Any]]:
        """Apply compression to messages with cache lookups."""
        query_terms = (
            latest_query_terms(messages) if getattr(config, "query_aware", True) else set()
        )

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

                # Try lossless compaction first. ContentDetector's vocabulary
                # (json/log/search/code/text) doesn't line up 1:1 with
                # compact_lossless's (log/grep/diff/text) — map the two
                # hints that do correspond so the log/grep-specific
                # transforms actually get a chance to run instead of always
                # falling through to the hardcoded "text" no-op branch.
                detected = self._detector.detect(content)
                lossless_hint = "grep" if detected == "search" else detected
                compacted = compact_lossless(content, lossless_hint)
                if compacted.transforms_applied:
                    content = compacted.compressed

                # Route through content router with model
                compressor_output = self._router.compress(
                    content, source_hint="text", model=model, query_terms=query_terms
                )
                if compressor_output and compressor_output.tokens_saved > 0:
                    content = compressor_output.compressed

                # Try recursive JSON routing — only for plain text.
                # JSON, code, log, and search content were already
                # handled by their dedicated compressors above; the
                # expensive balanced-brace scan would be wasted work.
                if compressor_output is not None and compressor_output.content_type == "text":
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


_THINKING_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


class ThinkingCompactor:
    """Strips thinking/reasoning blocks from messages."""

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip thinking blocks."""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Remove <think>...</think> blocks
                content = _THINKING_PATTERN.sub("", content).strip()
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
        cache_align_enabled: bool = False,
        cross_turn_dedup_enabled: bool = True,
        thinking_compact_enabled: bool = False,
        ccr_enabled: bool = True,
        output_shaping: bool = False,
        verbosity_level: int = 2,
        adaptive_sizing: bool = False,
        size_bias: float = 1.0,
        ml_compress_enabled: bool = False,
        ml_model_path: str | None = None,
        ml_tokenizer_path: str | None = None,
        ml_retention_threshold: float = 0.5,
        ml_min_compression_ratio: float = 0.1,
        compression_store: CompressionStore | None = None,
        semantic_dedup_enabled: bool = False,
        semantic_dedup_threshold: float = 0.85,
        semantic_dedup_model_path: str | None = None,
        semantic_dedup_config_path: str | None = None,
        semantic_dedup_vocab_path: str | None = None,
        strict: bool = False,
    ) -> None:
        self.compress_enabled = compress_enabled
        self.cache_align_enabled = cache_align_enabled
        self.cross_turn_dedup_enabled = cross_turn_dedup_enabled
        self.thinking_compact_enabled = thinking_compact_enabled
        self.ccr_enabled = ccr_enabled
        self.output_shaping = output_shaping
        self.verbosity_level = verbosity_level
        self.compression_store = compression_store
        self.strict = strict

        self.cache_aligner = CacheAligner(enabled=cache_align_enabled)
        self.compressor = CompressPhase(
            adaptive_sizing=adaptive_sizing,
            size_bias=size_bias,
            ml_compress_enabled=ml_compress_enabled,
            ml_model_path=ml_model_path,
            ml_tokenizer_path=ml_tokenizer_path,
            ml_retention_threshold=ml_retention_threshold,
            ml_min_compression_ratio=ml_min_compression_ratio,
        )
        self.thinking_compactor = ThinkingCompactor()
        self.output_shaper = OutputShaper(verbosity_level=verbosity_level)
        self._applied_transforms: list[str] = []
        self._warnings: list[str] = []

        # Semantic dedup — created lazily in apply() when config is known
        self._semantic_dedup_enabled = semantic_dedup_enabled
        self._semantic_dedup_threshold = semantic_dedup_threshold
        self._semantic_dedup_model_path = semantic_dedup_model_path
        self._semantic_dedup_config_path = semantic_dedup_config_path
        self._semantic_dedup_vocab_path = semantic_dedup_vocab_path
        self._semantic_dedup: SemanticDedup | None = None

    def apply(
        self,
        messages: list[dict[str, Any]],
        model: str = "gpt-4o",
        config: CompressConfig | None = None,
    ) -> TransformResult:
        """Run the full compression pipeline on messages."""
        if not messages:
            return TransformResult(
                messages=[], tokens_before=0, tokens_after=0, transforms_applied=[]
            )

        tokens_before = count_tokens_messages(messages, model)
        config = config or CompressConfig()

        # Compute salience scores before compression
        if getattr(config, "track_salience", True):
            salience_scores_before = [_compute_salience(msg.get("content", "")) for msg in messages]
        else:
            salience_scores_before = None
        self._applied_transforms = []
        self._warnings = []

        current_messages = list(messages)

        # Classify Read results before any phase that may alter their bytes.
        # Fresh reads are working copies: models commonly copy an exact span
        # from them into an Edit `old_text` argument, so every subsequent
        # transform must preserve their payload verbatim.
        ccr_hashes: list[str] = []
        fresh_read_snapshots: dict[str, Any] = {}
        lifecycle_enabled = getattr(config, "read_lifecycle_enabled", True)
        try:
            lifecycle_config = ReadLifecycleConfig(
                enabled=True,
                compress_stale=lifecycle_enabled and getattr(config, "compress_stale", True),
                compress_superseded=lifecycle_enabled
                and getattr(config, "compress_superseded", True),
                min_size_bytes=getattr(config, "min_read_lifecycle_bytes", 50),
                protect_recent=getattr(config, "protect_recent", 0),
            )
            lifecycle_result = classify_reads(
                current_messages,
                lifecycle_config,
                self.compression_store if lifecycle_enabled else None,
            )
            current_messages = lifecycle_result.messages
            ccr_hashes = lifecycle_result.ccr_hashes
            fresh_read_snapshots = _snapshot_read_results(
                current_messages, lifecycle_result.fresh_tool_call_ids
            )
            if lifecycle_result.reads_stale > 0 or lifecycle_result.reads_superseded > 0:
                self._applied_transforms.append("read_lifecycle")
                self._warnings.append(
                    f"read_lifecycle: {lifecycle_result.reads_stale} stale, "
                    f"{lifecycle_result.reads_superseded} superseded reads compressed"
                )
        except Exception as e:
            if self.strict:
                raise
            logger.warning(f"ReadLifecycle failed: {e}")
            self._warnings.append(f"Read lifecycle failed: {type(e).__name__}: {e}")

        # Phase 1: Output shaping (pass protect_recent to avoid modifying protected messages)
        if self.output_shaping:
            self.output_shaper.protect_recent = config.protect_recent
            shaped = self.output_shaper.apply(current_messages)
            if shaped != current_messages:
                self._applied_transforms.append("output_shaper")
            current_messages = _restore_read_results(shaped, fresh_read_snapshots)

        # Phase 2: Cache alignment
        if self.cache_align_enabled:
            try:
                aligned = self.cache_aligner.apply(current_messages)
                self._warnings.extend(self.cache_aligner.get_warnings())
                if aligned != current_messages:
                    self._applied_transforms.append("cache_aligner")
                current_messages = _restore_read_results(aligned, fresh_read_snapshots)
            except Exception as e:
                if self.strict:
                    raise
                logger.warning(f"CacheAligner failed: {e}")

        # Phase 2.5: Cross-turn dedup
        if self.cross_turn_dedup_enabled:
            try:
                blocks = [
                    DedupBlock(i, msg.get("content", "")) for i, msg in enumerate(current_messages)
                ]
                deduped = dedup_blocks(blocks)
                for i, block in enumerate(deduped):
                    if i < len(current_messages) and isinstance(
                        current_messages[i].get("content", ""), str
                    ):
                        current_messages[i] = {
                            **current_messages[i],
                            "content": block.content,
                        }
                if any(
                    deduped[index].content != blocks[index].content for index in range(len(blocks))
                ):
                    self._applied_transforms.append("cross_turn_dedup")
                current_messages = _restore_read_results(current_messages, fresh_read_snapshots)
            except Exception as e:
                if self.strict:
                    raise
                logger.warning(f"CrossTurnDedup failed: {e}")

        # Phase 2.7: Semantic cross-turn dedup
        if self._semantic_dedup_enabled:
            if self._semantic_dedup is None:
                self._semantic_dedup = SemanticDedup(
                    model_path=self._semantic_dedup_model_path
                    or getattr(config, "semantic_dedup_model_path", None),
                    config_path=self._semantic_dedup_config_path
                    or getattr(config, "semantic_dedup_config_path", None),
                    vocab_path=self._semantic_dedup_vocab_path
                    or getattr(config, "semantic_dedup_vocab_path", None),
                    threshold=getattr(
                        config, "semantic_dedup_threshold", self._semantic_dedup_threshold
                    ),
                    protect_recent=getattr(config, "protect_recent", 0),
                )
            try:
                dedup_result: SemanticDedupResult = self._semantic_dedup.dedup(current_messages)
                current_messages = _restore_read_results(
                    dedup_result.messages, fresh_read_snapshots
                )
                if dedup_result.dedup_count > 0:
                    self._applied_transforms.append("semantic_dedup")
                    self._warnings.append(
                        f"semantic_dedup: {dedup_result.dedup_count} semantically "
                        f"similar blocks replaced, {dedup_result.tokens_saved} tokens saved"
                    )
            except Exception as e:
                if self.strict:
                    raise
                logger.warning(f"SemanticDedup failed: {e}")
                self._warnings.append(f"Semantic dedup failed: {type(e).__name__}: {e}")

        # Phase 3: Compression
        if self.compress_enabled:
            try:
                compressed = self.compressor.apply(current_messages, config, model=model)
                if compressed != current_messages:
                    self._applied_transforms.append("compress")
                current_messages = _restore_read_results(compressed, fresh_read_snapshots)
            except Exception as e:
                if self.strict:
                    raise
                logger.warning(f"Compress phase failed: {e}")
                self._warnings.append(f"Compression failed: {type(e).__name__}: {e}")

        # Phase 4: Thinking compaction
        if self.thinking_compact_enabled:
            try:
                compacted = self.thinking_compactor.apply(current_messages)
                if compacted != current_messages:
                    self._applied_transforms.append("thinking_compactor")
                current_messages = _restore_read_results(compacted, fresh_read_snapshots)
            except Exception as e:
                if self.strict:
                    raise
                logger.warning(f"ThinkingCompactor failed: {e}")
                self._warnings.append(f"Thinking compaction failed: {type(e).__name__}: {e}")

        # Phase 5: CCR tool injection
        if self.ccr_enabled:
            try:
                injector = CCRToolInjector(provider="openai")
                injector.scan_for_markers(current_messages)
                if injector.has_compressed_content:
                    current_messages = injector.inject_system_instructions(current_messages)
                    self._applied_transforms.append("ccr_tool_injection")
            except Exception as e:
                if self.strict:
                    raise
                logger.warning(f"CCRToolInjector failed: {e}")
                self._warnings.append(f"CCR injection failed: {type(e).__name__}: {e}")

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
        if ccr_hashes:
            metadata["ccr_hashes"] = ccr_hashes

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
    cache_align_enabled: bool = False,
) -> TransformPipeline:
    """Create a default TransformPipeline with sensible settings."""
    return TransformPipeline(
        compress_enabled=compress_enabled,
        cache_align_enabled=cache_align_enabled,
        ccr_enabled=True,
        output_shaping=False,
        verbosity_level=2,
    )


# Pre-compiled regex patterns (compiled once at module load)
_UUID_PATTERN = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_ISO_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")


class CacheAligner:
    """Detects volatile content that busts KV cache alignment."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._warnings: list[str] = []
        self._metrics: dict[str, Any] = {"detected": 0, "aligned": 0}

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply cache alignment — replace volatile tokens with placeholders."""

        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Fast pre-check: skip regex if content doesn't look like
                # it contains UUIDs or ISO timestamps
                has_uuid = "-" in content and any(c in content for c in "abcdef0123456789")
                has_iso = "T" in content and "-" in content[:11]

                if has_uuid and _UUID_PATTERN.search(content):
                    content = _UUID_PATTERN.sub("[UUID_PLACEHOLDER]", content)
                    self._metrics["detected"] += 1

                if has_iso and _ISO_PATTERN.search(content):
                    content = _ISO_PATTERN.sub("[TIMESTAMP_PLACEHOLDER]", content)
                    self._metrics["detected"] += 1

                self._metrics["aligned"] += 1
            result.append({**msg, "content": content})
        return result

    def get_warnings(self) -> list[str]:
        return self._warnings

    def get_metrics(self) -> dict[str, Any]:
        return self._metrics.copy()
