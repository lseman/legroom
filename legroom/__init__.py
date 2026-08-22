"""Legroom — Context compression proxy for LLMs."""

from ._version import __version__

from .config import CompressConfig, CompressResult
from .compressors import (
    ContentDetector,
    ContentRouter,
    RouterCompressOutput,
    CompressionCache,
    CompressInput,
    CompressOutput,
    SmartCrusher,
    SmartCrusherConfig,
    LogCompressor,
    SearchCompressor,
    CodeCompressor,
    TextCompressor,
    MLTextCompressor,
    route_embedded_json,
    compact_lossless,
    LosslessResult,
    compute_optimal_k,
    count_unique_simhash,
)
from .cross_turn_dedup import DedupBlock, dedup_blocks
from .query_relevance import extract_query_terms, query_relevance, latest_query_terms
from .read_lifecycle import (
    ReadLifecycleConfig,
    ReadLifecycleResult,
    ReadState,
    classify_reads,
)

# Pipeline
from .pipeline import TransformPipeline, create_default_pipeline, TransformResult

# Compression
from .compress import compress, CompressResult as CliCompressResult

# CCR
from .ccr.compression_store import CompressionStore
from .ccr.marker_resolution import parse_markers, resolve_marker, create_resolution_prompt
from .ccr.tool_injection import CCRToolInjector, create_ccr_tool_definition, create_system_instructions

# Proxy and dashboard
from .proxy import LegroomProxy, ProxyState, RequestEvent, get_dashboard_html

__all__ = [
    "__version__",
    "CompressConfig",
    "CompressResult",
    "CliCompressResult",
    "ContentDetector",
    "ContentRouter",
    "RouterCompressOutput",
    "CompressionCache",
    "CompressInput",
    "CompressOutput",
    "SmartCrusher",
    "SmartCrusherConfig",
    "LogCompressor",
    "SearchCompressor",
    "CodeCompressor",
    "TextCompressor",
    "MLTextCompressor",
    "DedupBlock",
    "dedup_blocks",
    "route_embedded_json",
    "compact_lossless",
    "LosslessResult",
    "compute_optimal_k",
    "count_unique_simhash",
    "extract_query_terms",
    "query_relevance",
    "latest_query_terms",
    "ReadLifecycleConfig",
    "ReadLifecycleResult",
    "ReadState",
    "classify_reads",
    "TransformPipeline",
    "create_default_pipeline",
    "TransformResult",
    "compress",
    "CompressionStore",
    "parse_markers",
    "resolve_marker",
    "create_resolution_prompt",
    "CCRToolInjector",
    "create_ccr_tool_definition",
    "create_system_instructions",
    "LegroomProxy",
    "ProxyState",
    "RequestEvent",
    "get_dashboard_html",
]
