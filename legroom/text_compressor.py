"""Text compressor — basic text compression utilities."""

from __future__ import annotations

import re

from .compressor_registry import CompressInput, CompressOutput


class TextCompressor:
    """Basic text compression via whitespace and repetition normalization."""

    def compress(self, content: str, source_hint: str = "text") -> CompressOutput:
        """Compress text content."""
        # Collapse multiple spaces
        collapsed = re.sub(r"[ \t]+", " ", content)

        # Collapse multiple newlines
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)

        # Remove trailing whitespace on lines
        collapsed = re.sub(r"[ \t]+\n", "\n", collapsed)

        # Strip leading/trailing whitespace
        collapsed = collapsed.strip()

        tokens_before = len(content) // 4
        tokens_after = len(collapsed) // 4

        return CompressOutput(
            compressed=collapsed,
            original_token_count=tokens_before,
            compressed_token_count=tokens_after,
            strategy="text_compressor",
        )
