"""Compressors — per-content-type compression strategies and routing."""

from .content_detector import ContentDetector
from .content_router import ContentRouter, CompressOutput as RouterCompressOutput, CompressionCache
from .compressor_registry import CompressInput, CompressOutput, _compute_salience
from .smart_crusher import SmartCrusher, SmartCrusherConfig
from .log_compressor import LogCompressor
from .search_compressor import SearchCompressor
from .code_compressor import CodeCompressor
from .text_compressor import TextCompressor
from .ml_compressor import MLTextCompressor
from .recursive_json import route_embedded_json
from .lossless_compaction import compact_lossless, LosslessResult
from .adaptive_sizer import compute_optimal_k, count_unique_simhash

__all__ = [
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
    "route_embedded_json",
    "compact_lossless",
    "LosslessResult",
    "compute_optimal_k",
    "count_unique_simhash",
]
