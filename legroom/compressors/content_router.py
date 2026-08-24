"""Content router — dispatches to the right compressor."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any

from ..tokenizer import count_tokens
from .code_compressor import CodeCompressor
from .compressor_registry import CompressOutput
from .content_detector import ContentDetector
from .log_compressor import LogCompressor
from .ml_compressor import MLTextCompressor
from .search_compressor import SearchCompressor
from .smart_crusher import SmartCrusher, SmartCrusherConfig
from .text_compressor import TextCompressor

logger = logging.getLogger(__name__)


class CompressionCache:
    """LRU cache for compressed content.

    Increased to 512 for proxy mode where tool outputs repeat frequently.
    """

    def __init__(self, max_size: int = 512) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> str | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


class ContentRouter:
    """Routes content to the best compressor based on type detection."""

    def __init__(
        self,
        max_items: int = 50,
        cache_size: int = 512,
        adaptive_sizing: bool = False,
        size_bias: float = 1.0,
        ml_compress_enabled: bool = False,
        ml_model_path: str | None = None,
        ml_tokenizer_path: str | None = None,
        ml_retention_threshold: float = 0.5,
        ml_min_compression_ratio: float = 0.1,
    ) -> None:
        self._detector = ContentDetector()
        self._crusher = SmartCrusher(
            SmartCrusherConfig(
                max_items=max_items,
                adaptive_sizing=adaptive_sizing,
                size_bias=size_bias,
            )
        )
        self._log_compressor = LogCompressor()
        self._search_compressor = SearchCompressor()
        self._code_compressor = CodeCompressor()
        self._text_compressor = TextCompressor()
        self._cache = CompressionCache(max_size=cache_size)

        # Opt-in, best-effort: missing optional deps or model files just
        # mean we fall back to the lossless TextCompressor for text content.
        self._ml_compressor: MLTextCompressor | None = None
        if ml_compress_enabled:
            try:
                self._ml_compressor = MLTextCompressor(
                    model_path=ml_model_path,
                    tokenizer_path=ml_tokenizer_path,
                    retention_threshold=ml_retention_threshold,
                    min_compression_ratio=ml_min_compression_ratio,
                )
            except ImportError as e:
                logger.warning(f"ML compression requested but unavailable: {e}")

    def compress(
        self,
        content: str,
        source_hint: str = "unknown",
        model: str = "gpt-4o",
        query_terms: set[str] | None = None,
    ) -> CompressOutput | None:
        """Compress content, routing to the best compressor.

        ``query_terms`` biases JSON array compression toward items relevant
        to the current turn (see :mod:`legroom.query_relevance`). When set,
        the compression cache is bypassed for JSON content since a cached
        result from a different query wouldn't reflect the current bias.

        Cache check happens before content detection and JSON parsing to
        avoid wasted work on repeated identical content.
        """
        # Check cache — skipped for query-aware compression, since a cached
        # result may have been produced (or would be reused) under a
        # different query's relevance bias. This check happens before
        # content detection and JSON parsing to avoid wasted work.
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        if not query_terms:
            cached = self._cache.get(content_hash)
            if cached:
                return CompressOutput(
                    compressed=cached,
                    original_token_count=count_tokens(content, model),
                    compressed_token_count=count_tokens(cached, model),
                    strategy=f"cached:{source_hint}",
                )

        # Fast pre-check: if content looks like JSON, try parsing before
        # running the full detector — this avoids the detector's regex
        # overhead for the common case of repeated JSON tool outputs.
        stripped = content.strip()
        if stripped.startswith(("[", "{")):
            try:
                data = json.loads(content)
                if isinstance(data, list) and len(data) > 5:
                    output = self._crusher.compress(content, source_hint, model, query_terms)
                    if output:
                        output.content_type = "json"
                        self._cache.put(content_hash, output.compressed)
                        return output
            except (json.JSONDecodeError, ValueError):
                pass
            # Not a compressible JSON array — fall through to detector

        content_type = self._detector.detect(content)

        if content_type == "json":
            output = self._compress_json(content, source_hint, model, query_terms)
        elif content_type == "log":
            output = self._compress_log(content, source_hint, model)
        elif content_type == "search":
            output = self._compress_search(content, source_hint, model)
        elif content_type == "code":
            output = self._compress_code(content, source_hint, model)
        else:
            output = self._compress_text(content, source_hint, model)

        # Tag the output with the content type so callers can decide
        # whether the recursive JSON phase is still needed.
        if output and not query_terms:
            output.content_type = content_type
            self._cache.put(content_hash, output.compressed)
        elif output:
            output.content_type = content_type

        return output

    def _compress_json(
        self, content: str, source_hint: str, model: str, query_terms: set[str] | None = None
    ) -> CompressOutput | None:
        """Compress JSON content using SmartCrusher."""
        try:
            data = json.loads(content)
            if isinstance(data, list) and len(data) > 5:
                output = self._crusher.compress(content, source_hint, model, query_terms)
                if output:
                    return CompressOutput(
                        compressed=output.compressed,
                        original_token_count=output.original_token_count,
                        compressed_token_count=output.compressed_token_count,
                        strategy="smart_crusher",
                        content_type="json",
                    )
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _compress_log(self, content: str, source_hint: str, model: str) -> CompressOutput | None:
        """Compress log content."""
        output = self._log_compressor.compress(content, source_hint, model)
        if output:
            output.content_type = "log"
        return output

    def _compress_search(self, content: str, source_hint: str, model: str) -> CompressOutput | None:
        """Compress search results."""
        output = self._search_compressor.compress(content, source_hint, model)
        if output:
            output.content_type = "search"
        return output

    def _compress_code(self, content: str, source_hint: str, model: str) -> CompressOutput | None:
        """Compress code content."""
        output = self._code_compressor.compress(content, source_hint, model)
        if output:
            output.content_type = "code"
        return output

    def _compress_text(self, content: str, source_hint: str, model: str) -> CompressOutput | None:
        """Compress plain text: lossless whitespace normalization, then
        optional lossy ML token-retention scoring on top when enabled."""
        output = self._text_compressor.compress(content, source_hint, model)
        if output:
            output.content_type = "text"
        if self._ml_compressor is None:
            return output

        ml_output = self._ml_compressor.compress(output.compressed, source_hint, model)
        if ml_output.strategy == "ml_compressor" and ml_output.compressed_token_count < output.compressed_token_count:
            return ml_output
        return output

    def get_stats(self) -> dict[str, Any]:
        """Return compression statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_max_size": self._cache._max_size,
        }
