"""Search compressor — deduplicates search/grep results."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Iterable

from .compressor_registry import CompressInput, CompressOutput
from ..tokenizer import count_tokens

# Below this, factoring out a shared prefix costs more (the "# common: ..."
# heading) than it saves across the group.
MIN_PREFIX_CHARS = 8


class SearchCompressor:
    """Compresses search results by grouping by file and factoring out
    content patterns shared across a file's matches (e.g. every match
    being a call to the same function)."""

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

        # Rebuild with file headings, factoring out a common leading
        # substring per file group when most of its matches share one
        # (e.g. repeated calls to the same function/constructor).
        result = []
        for filepath, matches in files.items():
            result.append(filepath)
            ordered = sorted(matches)
            prefix = self._common_prefix(line_content for _, line_content in ordered)
            if prefix:
                result.append(f"  # common: {prefix}\N{HORIZONTAL ELLIPSIS}")
                for line_num, line_content in ordered:
                    result.append(f"  {line_num}: \N{HORIZONTAL ELLIPSIS}{line_content[len(prefix):]}")
            else:
                for line_num, line_content in ordered:
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

    def _common_prefix(self, lines: Iterable[str]) -> str:
        """Find a shared leading substring across a file's matched lines,
        cut back to a word boundary. Returns "" if there are too few lines,
        the prefix is too short, or the lines don't actually share much.
        """
        lines = list(lines)
        if len(lines) < 3:
            return ""

        prefix = os.path.commonprefix(lines)
        # Cut back to the last full "word" so we don't split an identifier
        # or a call's opening paren mid-token.
        match = re.match(r".*[\s(\[{,]", prefix)
        prefix = match.group(0) if match else ""

        if len(prefix) < MIN_PREFIX_CHARS:
            return ""
        return prefix
