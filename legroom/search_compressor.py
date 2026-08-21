"""Search compressor — deduplicates search/grep results."""

from __future__ import annotations

import re
from collections import defaultdict

from .compressor_registry import CompressInput, CompressOutput
from .tokenizer import count_tokens


class SearchCompressor:
    """Compresses search results by grouping by file."""

    def compress(
        self, content: str, source_hint: str = "search", model: str = "gpt-4o"
    ) -> CompressOutput:
        """Compress search result content."""
        lines = content.split("\n")
        if len(lines) < 5:
            return CompressOutput(
                compressed=content,
                original_token_count=count_tokens(content, model),
                compressed_token_count=count_tokens(content, model),
                strategy="search_compressor",
            )

        # Group by file
        files = defaultdict(list)
        for line in lines:
            match = re.match(r"^(\S+):(\d+):(.*)", line)
            if match:
                files[match.group(1)].append((int(match.group(2)), match.group(3)))

        if len(files) <= 1:
            return CompressOutput(
                compressed=content,
                original_token_count=count_tokens(content, model),
                compressed_token_count=count_tokens(content, model),
                strategy="search_compressor",
            )

        # Rebuild with file headings
        result = []
        for filepath, matches in files.items():
            result.append(filepath)
            for line_num, line_content in sorted(matches):
                result.append(f"  {line_num}: {line_content}")

        compressed = "\n".join(result)
        tokens_before = count_tokens(content, model)
        tokens_after = count_tokens(compressed, model)

        return CompressOutput(
            compressed=compressed,
            original_token_count=tokens_before,
            compressed_token_count=tokens_after,
            strategy="search_compressor",
        )
