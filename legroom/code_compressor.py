"""Code compressor — code-aware compression with comment stripping, dedup, and string compression."""

from __future__ import annotations

import re
from typing import Any

from .compressor_registry import CompressInput, CompressOutput
from .tokenizer import count_tokens

# Triple-quote string delimiters defined via chr() to avoid syntax issues
TRIPLE_DQ = chr(34) * 3  # """
TRIPLE_SQ = chr(39) * 3  # '''
HASH_COMMENT = "#"


class CodeCompressor:
    """Compresses code content via normalization, comment stripping, dedup, and string compression."""

    def compress(
        self, content: str, source_hint: str = "code", model: str = "gpt-4o"
    ) -> CompressOutput:
        """Compress code content with multiple strategies."""
        code_blocks = re.findall(r"```(\w*)\n(.*?)```", content, re.DOTALL)

        if not code_blocks:
            return self._compress_plain_text(content, model)

        result = []
        parts = re.split(r"(```(\w*)\n.*?```)", content, flags=re.DOTALL)

        for part in parts:
            if not part:
                continue
            if part.startswith("```"):
                lang = re.match(r"```(\w*)", part).group(1) or ""
                code_body = re.sub(r"^```\w*\n|\n```$", "", part).strip()
                code_body = self._compress_code_block(code_body, model)
                result.append(f"```{lang}\n{code_body}\n```")
            else:
                result.append(part)

        compressed = "".join(result)
        tokens_before = count_tokens(content, model)
        tokens_after = count_tokens(compressed, model)

        return CompressOutput(
            compressed=compressed,
            original_token_count=tokens_before,
            compressed_token_count=tokens_after,
            strategy="code_compressor",
        )

    def _compress_plain_text(self, content: str, model: str = "gpt-4o") -> CompressOutput:
        """Compress non-fenced code text with comment stripping, dedup, and whitespace normalization."""
        lines = content.split("\n")
        lines = self._strip_comments(lines)
        lines = self._collapse_duplicate_lines(lines)
        normalized = "\n".join([re.sub(r"[ \t]+", " ", l).strip() for l in lines])
        if len(normalized) < len(content):
            return CompressOutput(
                compressed=normalized,
                original_token_count=count_tokens(content, model),
                compressed_token_count=count_tokens(normalized, model),
                strategy="code_compressor",
            )
        return CompressOutput(
            compressed=content,
            original_token_count=count_tokens(content, model),
            compressed_token_count=count_tokens(content, model),
            strategy="code_compressor",
        )

    def _compress_code_block(self, code: str, model: str) -> str:
        """Apply code-aware compression to a single code block."""
        lines = code.split("\n")
        lines = self._strip_comments(lines)
        lines = self._collapse_duplicate_lines(lines)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
        lines = [self._compress_string_literals(line) for line in lines]
        return "\n".join(lines)

    def _strip_comments(self, lines: list[str]) -> list[str]:
        """Strip single-line comments and docstrings."""
        result = []
        in_multiline = False
        multiline_delim = None
        for line in lines:
            stripped = line.strip()
            if in_multiline:
                if multiline_delim and multiline_delim in stripped:
                    in_multiline = False
                continue
            for delim in (TRIPLE_DQ, TRIPLE_SQ, HASH_COMMENT):
                if delim == HASH_COMMENT:
                    in_str = False
                    comment_pos = None
                    for i, c in enumerate(stripped):
                        if c in ('"', "'") and not in_str:
                            in_str = True
                        elif c == '"':
                            in_str = False
                        elif c == "#" and not in_str:
                            comment_pos = i
                            break
                    if comment_pos is not None:
                        code_part = stripped[:comment_pos].rstrip()
                        leading = line[:len(line) - len(stripped)]
                        if code_part:
                            result.append(leading + code_part)
                        break
                    continue
                elif delim in (TRIPLE_DQ, TRIPLE_SQ):
                    if stripped.startswith(delim) and stripped.endswith(delim) and len(stripped) >= 6:
                        leading = line[:len(line) - len(stripped)]
                        result.append(leading + delim + "...")
                        break
                    elif stripped.startswith(delim):
                        in_multiline = True
                        multiline_delim = delim
                        if delim in stripped[3:]:
                            in_multiline = False
            if not in_multiline and stripped:
                if stripped.startswith(HASH_COMMENT):
                    continue
                result.append(line)
        return result

    def _collapse_duplicate_lines(self, lines: list[str], max_repeats: int = 3) -> list[str]:
        """Collapse duplicate consecutive lines."""
        if not lines:
            return lines
        result = [lines[0]]
        run_length = 1
        for i in range(1, len(lines)):
            if lines[i] == lines[i - 1]:
                run_length += 1
                if run_length == max_repeats + 1:
                    result.append(f"... # {run_length} identical lines collapsed")
            else:
                result.append(lines[i])
                run_length = 1
        return result

    def _compress_string_literals(self, line: str) -> str:
        """Compress long string literals on a single line."""
        def _compress_match(m: re.Match) -> str:
            prefix = m.group(1) or ""
            inner = m.group(2)
            if len(inner) > 60:
                return prefix + chr(34) + inner[:30] + "..." + inner[-20:] + chr(34)
            return m.group(0)
        pat = r'(?<=[(,=:])\s*([fbr]?)("[^"]*?")'
        line = re.sub(pat, _compress_match, line)
        return line
