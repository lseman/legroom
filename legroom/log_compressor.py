"""Log compressor — deduplicates and summarizes log entries."""

from __future__ import annotations

import re
from collections import Counter

from .compressor_registry import CompressInput, CompressOutput


class LogCompressor:
    """Compresses log content by deduplicating repeated entries."""

    def __init__(self, max_repeats: int = 5) -> None:
        self.max_repeats = max_repeats

    def compress(self, content: str, source_hint: str = "log") -> CompressOutput:
        """Compress log content."""
        lines = content.split("\n")
        if len(lines) < 10:
            return CompressOutput(
                compressed=content,
                original_token_count=len(content) // 4,
                compressed_token_count=len(content) // 4,
                strategy="log_compressor",
            )

        # Count repeated lines
        line_counts = Counter(lines)
        repeated = {line: count for line, count in line_counts.items() if count > self.max_repeats}

        if not repeated:
            return CompressOutput(
                compressed=content,
                original_token_count=len(content) // 4,
                compressed_token_count=len(content) // 4,
                strategy="log_compressor",
            )

        # Collapse repeated runs
        result_lines = []
        skip_count = 0

        for i, line in enumerate(lines):
            if skip_count > 0:
                skip_count -= 1
                continue

            if line in repeated and i + repeated[line] <= len(lines):
                # Check if this is a run of identical lines
                run_length = 0
                for j in range(i, min(i + len(repeated), len(lines))):
                    if lines[j] == line:
                        run_length += 1
                    else:
                        break

                if run_length > self.max_repeats:
                    result_lines.append(f"... (repeated {run_length} times)")
                    skip_count = run_length - 1
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)

        compressed = "\n".join(result_lines)
        tokens_before = len(content) // 4
        tokens_after = len(compressed) // 4

        return CompressOutput(
            compressed=compressed,
            original_token_count=tokens_before,
            compressed_token_count=tokens_after,
            strategy="log_compressor",
        )
