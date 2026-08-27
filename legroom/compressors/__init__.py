"""Compressors — per-content-type compression strategies and routing."""

from .adaptive_sizer import compute_optimal_k, count_unique_simhash
from .code_compressor import CodeCompressor
from .compressor_registry import CompressInput, CompressOutput
from .content_detector import ContentDetector
from .content_router import CompressionCache, ContentRouter
from .json_canonicalizer import JsonCanonicalizer, JsonCanonicalizeResult
from .kv_cache_fingerprinter import KVCacheFingerprint, KVCacheFingerprinter
from .log_compressor import LogCompressor
from .lossless_compaction import LosslessResult, compact_lossless
from .ml_compressor import MLTextCompressor
from .recursive_json import route_embedded_json
from .search_compressor import SearchCompressor
from .semantic_dedup import SemanticDedup, SemanticDedupResult
from .sequential_normalizer import SeqNormalizeResult, SequentialNumberNormalizer
from .smart_crusher import SmartCrusher, SmartCrusherConfig
from .text_compressor import TextCompressor
from .token_boundary_aligner import TokenBoundaryAligner, TokenBoundaryAlignResult
from .token_normalizer import TokenNormalizationResult, TokenNormalizer
from .tool_schema_canonicalizer import ToolSchemaCanonicalizer, ToolSchemaCanonicalizeResult
from .whitespace_canonicalizer import WhitespaceCanonicalizer, WhitespaceCanonicalizeResult

__all__ = [
    "CodeCompressor",
    "CompressInput",
    "CompressOutput",
    "CompressionCache",
    "ContentDetector",
    "ContentRouter",
    "JsonCanonicalizeResult",
    "JsonCanonicalizer",
    "KVCacheFingerprint",
    "KVCacheFingerprinter",
    "LogCompressor",
    "LosslessResult",
    "MLTextCompressor",
    "SearchCompressor",
    "SemanticDedup",
    "SemanticDedupResult",
    "SeqNormalizeResult",
    "SequentialNumberNormalizer",
    "SmartCrusher",
    "SmartCrusherConfig",
    "TextCompressor",
    "TokenBoundaryAlignResult",
    "TokenBoundaryAligner",
    "TokenNormalizationResult",
    "TokenNormalizer",
    "ToolSchemaCanonicalizeResult",
    "ToolSchemaCanonicalizer",
    "WhitespaceCanonicalizeResult",
    "WhitespaceCanonicalizer",
    "compact_lossless",
    "compute_optimal_k",
    "count_unique_simhash",
    "route_embedded_json",
]
