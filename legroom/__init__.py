"""Legroom — Context compression proxy for LLMs."""

from ._version import __version__

# Analysis utilities
from .analysis.cross_turn_dedup import DedupBlock, dedup_blocks
from .analysis.query_relevance import extract_query_terms, latest_query_terms, query_relevance
from .analysis.read_lifecycle import (
    ReadLifecycleConfig,
    ReadLifecycleResult,
    ReadState,
    classify_reads,
)
from .analysis.task_replay import (
    SubprocessTaskRunner,
    TaskReplayRequest,
    TaskRunner,
    TaskRunnerEvaluator,
    TaskRunnerProtocolError,
    TaskRunResult,
)

# CCR
from .ccr.compression_store import CompressionStore
from .ccr.marker_resolution import create_resolution_prompt, parse_markers, resolve_marker
from .ccr.tool_injection import (
    CCRToolInjector,
    create_ccr_tool_definition,
    create_system_instructions,
)

# Compression
from .compressors import (
    CodeCompressor,
    CompressInput,
    CompressionCache,
    CompressOutput,
    ContentDetector,
    ContentRouter,
    JsonCanonicalizer,
    JsonCanonicalizeResult,
    LogCompressor,
    LosslessResult,
    MLTextCompressor,
    SearchCompressor,
    SemanticDedup,
    SemanticDedupResult,
    SeqNormalizeResult,
    SequentialNumberNormalizer,
    SmartCrusher,
    SmartCrusherConfig,
    TextCompressor,
    ToolSchemaCanonicalizer,
    ToolSchemaCanonicalizeResult,
    compact_lossless,
    compute_optimal_k,
    count_unique_simhash,
    route_embedded_json,
)

# Proxy and dashboard
from .proxy import LegroomProxy, ProxyState, RequestEvent, get_dashboard_html

# Pipeline
from .runtime.compress import CompressResult as CliCompressResult
from .runtime.compress import compress
from .runtime.config import CompressConfig, CompressResult
from .runtime.pipeline import TransformPipeline, TransformResult, create_default_pipeline

__all__ = [
    "CCRToolInjector",
    "CliCompressResult",
    "CodeCompressor",
    "CompressConfig",
    "CompressInput",
    "CompressOutput",
    "CompressResult",
    "CompressionCache",
    "CompressionStore",
    "ContentDetector",
    "ContentRouter",
    "DedupBlock",
    "JsonCanonicalizeResult",
    "JsonCanonicalizer",
    "LegroomProxy",
    "LogCompressor",
    "LosslessResult",
    "MLTextCompressor",
    "ProxyState",
    "ReadLifecycleConfig",
    "ReadLifecycleResult",
    "ReadState",
    "RequestEvent",
    "SearchCompressor",
    "SemanticDedup",
    "SemanticDedupResult",
    "SeqNormalizeResult",
    "SequentialNumberNormalizer",
    "SmartCrusher",
    "SmartCrusherConfig",
    "SubprocessTaskRunner",
    "TaskReplayRequest",
    "TaskRunResult",
    "TaskRunner",
    "TaskRunnerEvaluator",
    "TaskRunnerProtocolError",
    "TextCompressor",
    "ToolSchemaCanonicalizeResult",
    "ToolSchemaCanonicalizer",
    "TransformPipeline",
    "TransformResult",
    "__version__",
    "classify_reads",
    "compact_lossless",
    "compress",
    "compute_optimal_k",
    "count_unique_simhash",
    "create_ccr_tool_definition",
    "create_default_pipeline",
    "create_resolution_prompt",
    "create_system_instructions",
    "dedup_blocks",
    "extract_query_terms",
    "get_dashboard_html",
    "latest_query_terms",
    "parse_markers",
    "query_relevance",
    "resolve_marker",
    "route_embedded_json",
]
