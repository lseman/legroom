"""Compressors — per-content-type compression strategies and routing."""

from .adaptive_sizer import compute_optimal_k, count_unique_simhash
from .code_compressor import CodeCompressor
from .compressor_registry import CompressInput, CompressOutput
from .content_detector import ContentDetector
from .content_router import CompressionCache, ContentRouter
from .log_compressor import LogCompressor
from .lossless_compaction import LosslessResult, compact_lossless
from .ml_compressor import MLTextCompressor
from .recursive_json import route_embedded_json
from .search_compressor import SearchCompressor
from .semantic_dedup import SemanticDedup, SemanticDedupResult
from .smart_crusher import SmartCrusher, SmartCrusherConfig
from .text_compressor import TextCompressor

__all__ = [
    "CodeCompressor",
    "CompressInput",
    "CompressOutput",
    "CompressionCache",
    "ContentDetector",
    "ContentRouter",
    "LogCompressor",
    "LosslessResult",
    "MLTextCompressor",
    "SearchCompressor",
    "SemanticDedup",
    "SemanticDedupResult",
    "SmartCrusher",
    "SmartCrusherConfig",
    "TextCompressor",
    "compact_lossless",
    "compute_optimal_k",
    "count_unique_simhash",
    "route_embedded_json",
]
