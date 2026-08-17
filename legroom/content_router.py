"""Content router — dispatches to the right compressor."""

from __future__ import annotations

import json
import hashlib
from typing import Any, Optional
from collections import OrderedDict

from .content_detector import ContentDetector
from .smart_crusher import SmartCrusher, SmartCrusherConfig
from .log_compressor import LogCompressor
from .search_compressor import SearchCompressor
from .code_compressor import CodeCompressor
from .text_compressor import TextCompressor


class CompressionCache:
    """LRU cache for compressed content."""

    def __init__(self, max_size: int = 100) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Optional[str]:
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


class CompressOutput:
    """Output from compression."""

    def __init__(
        self,
        compressed: str,
        original_token_count: int,
        compressed_token_count: int,
        strategy: str = "unknown",
        routing_log: list[dict] | None = None,
    ) -> None:
        self.compressed = compressed
        self.original_token_count = original_token_count
        self.compressed_token_count = compressed_token_count
        self.strategy = strategy
        self.routing_log = routing_log or []

    @property
    def tokens_saved(self) -> int:
        return self.original_token_count - self.compressed_token_count


class ContentRouter:
    """Routes content to the best compressor based on type detection."""

    def __init__(self, max_items: int = 50, cache_size: int = 100) -> None:
        self._detector = ContentDetector()
        self._crusher = SmartCrusher(SmartCrusherConfig(max_items=max_items))
        self._log_compressor = LogCompressor()
        self._search_compressor = SearchCompressor()
        self._code_compressor = CodeCompressor()
        self._text_compressor = TextCompressor()
        self._cache = CompressionCache(max_size=cache_size)

    def compress(self, content: str, source_hint: str = "unknown") -> Optional[CompressOutput]:
        """Compress content, routing to the best compressor."""
        # Check cache
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        cached = self._cache.get(content_hash)
        if cached:
            return CompressOutput(
                compressed=cached,
                original_token_count=len(content) // 4,
                compressed_token_count=len(cached) // 4,
                strategy=f"cached:{source_hint}",
            )

        content_type = self._detector.detect(content)

        if content_type == "json":
            output = self._compress_json(content, source_hint)
        elif content_type == "log":
            output = self._compress_log(content, source_hint)
        elif content_type == "search":
            output = self._compress_search(content, source_hint)
        elif content_type == "code":
            output = self._compress_code(content, source_hint)
        else:
            output = self._text_compressor.compress(content, source_hint)

        if output:
            self._cache.put(content_hash, output.compressed)

        return output

    def _compress_json(self, content: str, source_hint: str) -> Optional[CompressOutput]:
        """Compress JSON content using SmartCrusher."""
        try:
            data = json.loads(content)
            if isinstance(data, list) and len(data) > 5:
                output = self._crusher.compress(content, source_hint)
                if output:
                    return CompressOutput(
                        compressed=output.compressed,
                        original_token_count=len(content) // 4,
                        compressed_token_count=len(output.compressed) // 4,
                        strategy="smart_crusher",
                    )
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _compress_log(self, content: str, source_hint: str) -> Optional[CompressOutput]:
        """Compress log content."""
        return self._log_compressor.compress(content, source_hint)

    def _compress_search(self, content: str, source_hint: str) -> Optional[CompressOutput]:
        """Compress search results."""
        return self._search_compressor.compress(content, source_hint)

    def _compress_code(self, content: str, source_hint: str) -> Optional[CompressOutput]:
        """Compress code content."""
        return self._code_compressor.compress(content, source_hint)

    def get_stats(self) -> dict[str, Any]:
        """Return compression statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_max_size": self._cache._max_size,
        }
