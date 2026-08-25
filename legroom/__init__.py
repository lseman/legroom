"""Legroom — Context compression proxy for LLMs."""

from ._version import __version__

# CCR
from .ccr.compression_store import CompressionStore
from .ccr.marker_resolution import create_resolution_prompt, parse_markers, resolve_marker
from .ccr.tool_injection import (
    CCRToolInjector,
    create_ccr_tool_definition,
    create_system_instructions,
)
from .compress import CompressResult as CliCompressResult

# Compression
from .compress import compress
from .compressors import (
    CodeCompressor,
    CompressInput,
    CompressionCache,
    CompressOutput,
    ContentDetector,
    ContentRouter,
    LogCompressor,
    LosslessResult,
    MLTextCompressor,
    SearchCompressor,
    SemanticDedup,
    SemanticDedupResult,
    SmartCrusher,
    SmartCrusherConfig,
    TextCompressor,
    compact_lossless,
    compute_optimal_k,
    count_unique_simhash,
    route_embedded_json,
)
from .config import CompressConfig, CompressResult
from .cross_turn_dedup import DedupBlock, dedup_blocks

# Pipeline
from .pipeline import TransformPipeline, TransformResult, create_default_pipeline

# Proxy and dashboard
from .proxy import LegroomProxy, ProxyState, RequestEvent, get_dashboard_html
from .query_relevance import extract_query_terms, latest_query_terms, query_relevance
from .read_lifecycle import (
    ReadLifecycleConfig,
    ReadLifecycleResult,
    ReadState,
    classify_reads,
)
from .task_replay import (
    SubprocessTaskRunner,
    TaskReplayRequest,
    TaskRunner,
    TaskRunnerEvaluator,
    TaskRunnerProtocolError,
    TaskRunResult,
)

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
    "SmartCrusher",
    "SmartCrusherConfig",
    "SubprocessTaskRunner",
    "TaskReplayRequest",
    "TaskRunResult",
    "TaskRunner",
    "TaskRunnerEvaluator",
    "TaskRunnerProtocolError",
    "TextCompressor",
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
