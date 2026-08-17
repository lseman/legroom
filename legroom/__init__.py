"""Legroom — Context compression proxy for LLMs."""

from ._version import __version__

from .config import CompressConfig, CompressResult
from .content_detector import ContentDetector
from .content_router import ContentRouter, CompressOutput as RouterCompressOutput, CompressionCache
from .compressor_registry import CompressInput, CompressOutput
from .smart_crusher import SmartCrusher, SmartCrusherConfig
from .log_compressor import LogCompressor
from .search_compressor import SearchCompressor
from .code_compressor import CodeCompressor
from .text_compressor import TextCompressor
from .ml_compressor import MLTextCompressor
from .cross_turn_dedup import DedupBlock, dedup_blocks
from .recursive_json import route_embedded_json
from .lossless_compaction import compact_lossless, LosslessResult
from .adaptive_sizer import compute_optimal_k, count_unique_simhash

# Pipeline
from .pipeline import TransformPipeline, create_default_pipeline, TransformResult

# Compression
from .compress import compress, CompressResult as CliCompressResult

# CCR
from .ccr.compression_store import CompressionStore
from .ccr.marker_resolution import parse_markers, resolve_marker, create_resolution_prompt
from .ccr.tool_injection import CCRToolInjector, create_ccr_tool_definition, create_system_instructions

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
]
