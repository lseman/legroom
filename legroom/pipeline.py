"""Transform pipeline — orchestrates compression transforms.

Parallel execution:
- Within CompressPhase: each message's compression runs in parallel
- Within SemanticDedup: ONNX inference batched (already done in the
  optimizer module)
- Within CrossTurnDedup: hashing runs in parallel
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from .ccr.compression_store import CompressionStore
from .ccr.tool_injection import CCRToolInjector
from .compressors.compressor_registry import _compute_salience
from .compressors.content_detector import ContentDetector
from .compressors.content_router import ContentRouter
from .compressors.kv_cache_optimizer import KVOptimizer
from .compressors.lossless_compaction import compact_lossless
from .compressors.models import apply_profile
from .compressors.recursive_json import route_embedded_json
from .compressors.semantic_dedup import SemanticDedup, SemanticDedupResult
from .config import CompressConfig
from .cross_turn_dedup import DedupBlock, dedup_blocks
from .ir import Conversation
from .output.shaper import OutputShaper
from .phases import CallablePhase, PhaseContext, PhaseProposal, PhaseRunner
from .query_relevance import latest_query_terms
from .read_lifecycle import ReadLifecycleConfig, classify_reads
from .risk_policy import RiskAssessment, RiskPolicy
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
    """Phase that applies content compression.

    Parallel execution: compresses each message independently using a
    thread pool. This is the biggest parallelization win since compression
    is CPU-bound and each message's compression is independent.
    """

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
        max_workers: int = 4,
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
        self._max_workers = max_workers

    def _compress_single(
        self,
        msg: dict[str, Any],
        query_terms: set[str],
        model: str,
    ) -> dict[str, Any]:
        """Compress a single message. Used by both sequential and parallel paths."""
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            # Check compression cache (before lossless, on raw content).
            # Skipped under query-aware compression — see ContentRouter.compress.
            if not query_terms:
                cached = self._cache.get(content)
                if cached is not None:
                    return {**msg, "content": cached}

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
                def _dispatch(span: str) -> str | None:
                    output = self._router.compress(
                        span, source_hint="text", model=model, query_terms=query_terms
                    )
                    return output.compressed if output is not None else None

                json_result = route_embedded_json(content, _dispatch)
                if json_result is not None:
                    content = json_result

            # Cache the result keyed by the original raw content
            if content != original_content and not query_terms:
                self._cache.put(original_content, content)

        return {**msg, "content": content}

    def apply(
        self, messages: list[dict[str, Any]], config: CompressConfig, model: str = "gpt-4o"
    ) -> list[dict[str, Any]]:
        """Apply compression to messages with cache lookups.

        Parallel execution: compresses messages in batches using a thread
        pool. Each message's compression is independent, so this scales
        with CPU cores. The shared cache is thread-safe via OrderedDict's
        atomic operations.
        """
        query_terms = (
            latest_query_terms(messages) if getattr(config, "query_aware", True) else set()
        )

        # Fast path: single message or no content — no parallelism needed
        if len(messages) <= 1:
            return [
                self._compress_single(msg, query_terms, model) for msg in messages
            ]

        # Parallel compression using thread pool
        result: list[dict[str, Any] | None] = [None] * len(messages)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._compress_single, msg, query_terms, model): i
                for i, msg in enumerate(messages)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result[idx] = future.result()

        return result  # type: ignore[return-value]


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
        kv_cache_optimization_enabled: bool = False,
        kv_cache_min_prefix_bytes: int = 100,
        kv_cache_min_occurrences: int = 2,
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
        self._phase_reports: list[dict[str, Any]] = []
        self._disabled_phases: set[str] = set()

        # Semantic dedup — created lazily in apply() when config is known
        self._semantic_dedup_enabled = semantic_dedup_enabled
        self._semantic_dedup_threshold = semantic_dedup_threshold
        self._semantic_dedup_model_path = semantic_dedup_model_path
        self._semantic_dedup_config_path = semantic_dedup_config_path
        self._semantic_dedup_vocab_path = semantic_dedup_vocab_path
        self._semantic_dedup: SemanticDedup | None = None

        # KV cache optimization — created lazily in apply() when config is known
        self._kv_cache_optimization_enabled = kv_cache_optimization_enabled
        self._kv_cache_min_prefix_bytes = kv_cache_min_prefix_bytes
        self._kv_cache_min_occurrences = kv_cache_min_occurrences
        self._kv_optimizer: KVOptimizer | None = None

    def _run_phase(
        self,
        *,
        name: str,
        transform_name: str,
        messages: list[dict[str, Any]],
        transform: Callable[
            [list[dict[str, Any]]], list[dict[str, Any]] | PhaseProposal
        ],
        model: str,
        protected_spans: tuple[str, ...] = (),
        reversible: bool = False,
        confidence: float = 1.0,
        allow_inflation: bool = False,
    ) -> list[dict[str, Any]]:
        """Run one existing transform through the common phase seam."""
        if transform_name.lower() in self._disabled_phases:
            transform = lambda value: PhaseProposal(
                value, metadata={"disabled_by_calibration": True}
            )
        phase = CallablePhase(
            name,
            transform,
            reversible=reversible,
            confidence=confidence,
            allow_inflation=allow_inflation,
        )
        outcome = PhaseRunner(strict=self.strict).run(
            phase,
            messages,
            PhaseContext(model=model, protected_spans=protected_spans),
        )
        self._phase_reports.append(asdict(outcome.report))
        self._warnings.extend(outcome.warnings)
        if outcome.report.status == "applied":
            self._applied_transforms.append(transform_name)
        elif outcome.report.status == "failed":
            logger.warning("%s failed: %s", name, outcome.report.error)
        return outcome.messages

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

        # Profiles are explicit presets. Applying them implicitly made it
        # impossible to distinguish a caller's explicit value from a dataclass
        # default (notably the proxy's protect_recent=0).
        if config.use_model_profile:
            config = apply_profile(model, config)

        risk_policy = RiskPolicy()
        risk_assessment = (
            risk_policy.assess(Conversation.from_mappings(messages))
            if config.risk_policy_enabled
            else RiskAssessment((), ())
        )
        policy_original = list(messages)

        def restore_policy(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return risk_policy.restore(policy_original, value, risk_assessment)

        # Compute salience scores before compression
        if getattr(config, "track_salience", True):
            salience_scores_before = [_compute_salience(msg.get("content", "")) for msg in messages]
        else:
            salience_scores_before = None
        self._applied_transforms = []
        self._warnings = []
        self._phase_reports = []
        self._disabled_phases = {name.lower() for name in config.disabled_phases}

        current_messages = list(messages)

        # Classify Read results before any phase that may alter their bytes.
        # Fresh reads are working copies: models commonly copy an exact span
        # from them into an Edit `old_text` argument, so every subsequent
        # transform must preserve their payload verbatim.
        ccr_hashes: list[str] = []
        fresh_read_snapshots: dict[str, Any] = {}
        lifecycle_enabled = config.read_lifecycle_enabled
        lifecycle_holder: dict[str, Any] = {}

        def run_read_lifecycle(phase_messages: list[dict[str, Any]]) -> PhaseProposal:
            lifecycle_config = ReadLifecycleConfig(
                enabled=True,
                compress_stale=lifecycle_enabled and config.compress_stale,
                compress_superseded=lifecycle_enabled and config.compress_superseded,
                min_size_bytes=config.min_read_lifecycle_bytes,
                protect_recent=config.protect_recent,
            )
            lifecycle_result = classify_reads(
                phase_messages,
                lifecycle_config,
                self.compression_store if lifecycle_enabled else None,
            )
            lifecycle_holder["result"] = lifecycle_result
            warnings: tuple[str, ...] = ()
            if lifecycle_result.reads_stale or lifecycle_result.reads_superseded:
                warnings = (
                    (
                        f"read_lifecycle: {lifecycle_result.reads_stale} stale, "
                        f"{lifecycle_result.reads_superseded} superseded reads compressed"
                    ),
                )
            return PhaseProposal(
                restore_policy(lifecycle_result.messages),
                metadata={
                    "reads_stale": lifecycle_result.reads_stale,
                    "reads_superseded": lifecycle_result.reads_superseded,
                    "reads_fresh": lifecycle_result.reads_fresh,
                },
                warnings=warnings,
            )

        current_messages = self._run_phase(
            name="read_lifecycle",
            transform_name="read_lifecycle",
            messages=current_messages,
            transform=run_read_lifecycle,
            model=model,
            protected_spans=risk_assessment.protected_spans,
            reversible=self.compression_store is not None,
            confidence=1.0,
        )
        lifecycle_result = lifecycle_holder.get("result")
        if lifecycle_result is not None:
            ccr_hashes = lifecycle_result.ccr_hashes
            fresh_read_snapshots = _snapshot_read_results(
                current_messages, lifecycle_result.fresh_tool_call_ids
            )

        protected_spans = (
            *risk_assessment.protected_spans,
            *(f"tool_call:{key}" for key in fresh_read_snapshots),
        )

        def preserve_reads(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return restore_policy(_restore_read_results(output, fresh_read_snapshots))

        # Phase 1: Output shaping (pass protect_recent to avoid modifying protected messages)
        if self.output_shaping:
            self.output_shaper.protect_recent = config.protect_recent
            current_messages = self._run_phase(
                name="output_shaper",
                transform_name="output_shaper",
                messages=current_messages,
                transform=lambda value: preserve_reads(self.output_shaper.apply(value)),
                model=model,
                protected_spans=protected_spans,
                confidence=0.9,
            )

        # Phase 2: Cache alignment
        if self.cache_align_enabled:
            def align(value: list[dict[str, Any]]) -> PhaseProposal:
                output = preserve_reads(self.cache_aligner.apply(value))
                return PhaseProposal(output, warnings=tuple(self.cache_aligner.get_warnings()))

            current_messages = self._run_phase(
                name="cache_aligner",
                transform_name="cache_aligner",
                messages=current_messages,
                transform=align,
                model=model,
                protected_spans=protected_spans,
                confidence=1.0,
            )

        # Phase 2.5: Cross-turn dedup
        if self.cross_turn_dedup_enabled:
            def cross_turn(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
                output = list(value)
                blocks = [
                    DedupBlock(i, msg.get("content", "")) for i, msg in enumerate(value)
                ]
                deduped = dedup_blocks(blocks)
                for i, block in enumerate(deduped):
                    if i < len(value) and isinstance(value[i].get("content", ""), str):
                        output[i] = {**value[i], "content": block.content}
                return preserve_reads(output)

            current_messages = self._run_phase(
                name="cross_turn_dedup",
                transform_name="cross_turn_dedup",
                messages=current_messages,
                transform=cross_turn,
                model=model,
                protected_spans=protected_spans,
                reversible=True,
                confidence=1.0,
            )

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
            semantic_dedup = self._semantic_dedup
            def semantic(value: list[dict[str, Any]]) -> PhaseProposal:
                dedup_result: SemanticDedupResult = semantic_dedup.dedup(
                    value, model=model
                )
                output = preserve_reads(dedup_result.messages)
                warnings = tuple(dedup_result.warnings)
                if dedup_result.dedup_count > 0:
                    warnings += (
                        (
                            f"semantic_dedup: {dedup_result.dedup_count} semantically "
                            f"similar blocks replaced, {dedup_result.tokens_saved} tokens saved"
                        ),
                    )
                return PhaseProposal(
                    output,
                    metadata={"dedup_count": dedup_result.dedup_count},
                    warnings=warnings,
                )

            current_messages = self._run_phase(
                name="semantic_dedup",
                transform_name="semantic_dedup",
                messages=current_messages,
                transform=semantic,
                model=model,
                protected_spans=protected_spans,
                confidence=config.semantic_dedup_threshold,
            )

        # Phase 2.8: KV cache optimization (prefix dedup + token-boundary alignment)
        if self._kv_cache_optimization_enabled:
            if self._kv_optimizer is None:
                self._kv_optimizer = KVOptimizer(
                    min_prefix_bytes=self._kv_cache_min_prefix_bytes,
                    min_occurrences=self._kv_cache_min_occurrences,
                )
            kv_optimizer = self._kv_optimizer
            def optimize_kv(value: list[dict[str, Any]]) -> PhaseProposal:
                kv_result = kv_optimizer.optimize(value, model=model)
                output = preserve_reads(kv_result.messages)
                warnings: tuple[str, ...] = ()
                if kv_result.prefix_dedup_count > 0 or kv_result.token_boundary_aligned > 0:
                    warnings = (
                        (
                            f"kv_cache_optimization: {kv_result.prefix_dedup_count} prefixes "
                            f"deduped, {kv_result.token_boundary_aligned} messages aligned"
                        ),
                    )
                return PhaseProposal(output, metadata=kv_result.metadata, warnings=warnings)

            current_messages = self._run_phase(
                name="kv_cache_optimization",
                transform_name="kv_cache_optimization",
                messages=current_messages,
                transform=optimize_kv,
                model=model,
                protected_spans=protected_spans,
                confidence=0.8,
            )

        # Phase 3: Compression
        if self.compress_enabled:
            current_messages = self._run_phase(
                name="Compression",
                transform_name="compress",
                messages=current_messages,
                transform=lambda value: preserve_reads(
                    self.compressor.apply(value, config, model=model)
                ),
                model=model,
                protected_spans=protected_spans,
                confidence=0.85,
            )

        # Phase 4: Thinking compaction
        if self.thinking_compact_enabled:
            current_messages = self._run_phase(
                name="thinking_compactor",
                transform_name="thinking_compactor",
                messages=current_messages,
                transform=lambda value: preserve_reads(self.thinking_compactor.apply(value)),
                model=model,
                protected_spans=protected_spans,
                confidence=1.0,
            )

        # Phase 5: CCR tool injection
        if self.ccr_enabled:
            def inject_ccr(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
                injector = CCRToolInjector(provider="openai")
                injector.scan_for_markers(value)
                if injector.has_compressed_content:
                    return injector.inject_system_instructions(value)
                return value

            current_messages = self._run_phase(
                name="ccr_tool_injection",
                transform_name="ccr_tool_injection",
                messages=current_messages,
                transform=inject_ccr,
                model=model,
                protected_spans=protected_spans,
                reversible=True,
                confidence=1.0,
                allow_inflation=True,
            )

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
                    "phase_reports": self._phase_reports,
                    "risk_assessment": list(risk_assessment.labels),
                },
            )

        # Compute salience scores after compression
        if getattr(config, "track_salience", True):
            salience_scores_after = [
                _compute_salience(msg.get("content", "")) for msg in current_messages
            ]
        else:
            salience_scores_after = None

        metadata: dict[str, Any] = {
            "cache_metrics": self.cache_aligner.get_metrics(),
            "phase_reports": self._phase_reports,
            "risk_assessment": list(risk_assessment.labels),
        }
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
