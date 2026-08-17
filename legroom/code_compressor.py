"""Code compressor — basic code compression via whitespace normalization."""

from __future__ import annotations

import re

from .compressor_registry import CompressInput, CompressOutput


class CodeCompressor:
    """Compresses code content via basic normalization."""

    def compress(self, content: str, source_hint: str = "code") -> CompressOutput:
        """Compress code content."""
        # Extract fenced code blocks
        code_blocks = re.findall(r"```(\w*)\n(.*?)```", content, re.DOTALL)

        if not code_blocks:
            # Try to normalize whitespace in the entire content
            normalized = re.sub(r"[ \t]+", " ", content).strip()
            if len(normalized) < len(content):
                return CompressOutput(
                    compressed=normalized,
                    original_token_count=len(content) // 4,
                    compressed_token_count=len(normalized) // 4,
                    strategy="code_compressor",
                )
            return CompressOutput(
                compressed=content,
                original_token_count=len(content) // 4,
                compressed_token_count=len(content) // 4,
                strategy="code_compressor",
            )

        # Process code blocks
        result = []
        for lang, code in code_blocks:
            # Normalize whitespace within code blocks
            normalized = re.sub(r"[ \t]+", " ", code).strip()
            result.append(f"```{lang}\n{normalized}\n```")

        # Replace code blocks in original
        compressed = re.sub(r"```(\w*)\n.*?```", lambda m: f"```{m.group(1)}\n{re.sub(r'[ \t]+', ' ', m.group(2)).strip()}\n```", content, flags=re.DOTALL)

        tokens_before = len(content) // 4
        tokens_after = len(compressed) // 4

        return CompressOutput(
            compressed=compressed,
            original_token_count=tokens_before,
            compressed_token_count=tokens_after,
            strategy="code_compressor",
        )
