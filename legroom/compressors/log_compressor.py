"""Log compressor — deduplicates and summarizes log entries."""

from __future__ import annotations

from collections import Counter

from ..analysis.tokenizer import count_tokens
from .compressor_registry import CompressOutput


class LogCompressor:
    """Compresses log content by deduplicating repeated entries."""

    def __init__(self, max_repeats: int = 5) -> None:
        self.max_repeats = max_repeats

    def compress(
        self, content: str, source_hint: str = "log", model: str = "gpt-4o"
    ) -> CompressOutput:
        """Compress log content."""
        lines = content.split("\n")
        if len(lines) < 10:
            return CompressOutput(
                compressed=content,
                original_token_count=count_tokens(content, model),
                compressed_token_count=count_tokens(content, model),
                strategy="log_compressor",
            )

        # Count repeated lines
        line_counts = Counter(lines)
        repeated = {line: count for line, count in line_counts.items() if count > self.max_repeats}

        if not repeated:
            return CompressOutput(
                compressed=content,
                original_token_count=count_tokens(content, model),
                compressed_token_count=count_tokens(content, model),
                strategy="log_compressor",
            )

        # Collapse repeated runs
        result_lines = []
        skip_count = 0

        for i, line in enumerate(lines):
            if skip_count > 0:
                skip_count -= 1
                continue

            if line in repeated:
                # Count the full run of consecutive identical lines starting here.
                run_length = 0
                for j in range(i, len(lines)):
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
        tokens_before = count_tokens(content, model)
        tokens_after = count_tokens(compressed, model)

        return CompressOutput(
            compressed=compressed,
            original_token_count=tokens_before,
            compressed_token_count=tokens_after,
            strategy="log_compressor",
        )
