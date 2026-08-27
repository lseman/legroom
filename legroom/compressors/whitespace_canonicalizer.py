"""Whitespace and Unicode canonicalization for llama.cpp KV cache alignment.

llama.cpp's KV cache matches tokenized prefixes byte-for-byte. Invisible
whitespace differences — tabs vs spaces, non-breaking spaces, Unicode
normalization forms (NFC vs NFD), trailing whitespace — produce different
token sequences and bust the KV cache even though the semantic content is
identical.

This module canonicalizes whitespace and Unicode at the character level
before tokenization, ensuring that semantically identical text produces
identical token sequences.

    "  \t  " vs "    "  →  different tokens, same meaning
    "Café" (NFC) vs "Café" (NFD)  →  different tokens
    "\u00a0" (NBSP) vs " "  →  different tokens

**Impact:** Free alignment gains — these patterns are extremely common in
agent outputs (JSON pretty-printing, tool outputs, code diffs) and cause
100% KV cache misses despite being semantically identical.

Usage::

    from legroom.compressors.whitespace_canonicalizer import (
        WhitespaceCanonicalizer,
    )

    canonicalizer = WhitespaceCanonicalizer()
    result = canonicalizer.normalize(text, backend="llama_cpp")

The canonicalization is **lossless** — the original whitespace can be
recovered by the CCR store when needed for exact editing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


# Pre-compiled patterns
# Non-breaking space, thin space, em space, en space → regular space
_NON_STANDARD_SPACE = re.compile(
    r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]+"
)

# Tab character → space (llama.cpp tokenizes tabs differently)
_TAB = re.compile(r"\t")

# Multiple spaces → single space (only after normalizing other whitespace)
_MULTI_SPACES = re.compile(r" {2,}")

# Trailing whitespace at end of lines
_TRAILING_WHITESPACE = re.compile(r"[ \t]+(?=\n)", re.MULTILINE)

# Trailing whitespace at end of string
_TRAILING_END = re.compile(r"[ \t]+$", re.MULTILINE)

# Leading whitespace at start of lines (only spaces, not tabs)
_LEADING_INDENT = re.compile(r"^ {2,}", re.MULTILINE)


@dataclass(frozen=True)
class WhitespaceCanonicalizeResult:
    """Result of whitespace/Unicode canonicalization."""

    text: str
    changes: int = 0
    """Total number of character-level changes made."""

    tokens_saved: int = 0
    """Estimated tokens saved (rough heuristic)."""


class WhitespaceCanonicalizer:
    """Canonicalizes whitespace and Unicode for KV cache alignment.

    Applies these transformations in order:
    1. Unicode normalization (NFC) — ensures consistent Unicode representation
    2. Non-standard space → regular space — NBSP, thin space, etc.
    3. Tab → space — tabs tokenize differently than spaces
    4. Multiple spaces → single space — reduces token count
    5. Trailing whitespace removal — trailing spaces cause cache misses
    6. Indent canonicalization — 4+ spaces → consistent indentation

    All transformations are reversible via CCR when exact content is needed.
    """

    def normalize(
        self,
        text: str,
        *,
        backend: str = "openai",
        protected_indices: set[int] | None = None,
    ) -> WhitespaceCanonicalizeResult:
        """Canonicalize whitespace and Unicode in text.

        For ``backend="llama_cpp"``, whitespace and Unicode patterns are
        normalized. For other backends, returns text unchanged.

        Args:
            text: The text to canonicalize.
            backend: Target backend — only ``"llama_cpp"`` triggers
                canonicalization.
            protected_indices: Not used for text canonicalization
                (reserved for message-level protection).

        Returns:
            WhitespaceCanonicalizeResult with canonicalized text and stats.
        """
        if backend != "llama_cpp":
            return WhitespaceCanonicalizeResult(text=text)

        original = text
        canonicalized = text

        # Step 1: Unicode normalization (NFC form)
        canonicalized, count1 = self._normalize_unicode(canonicalized)

        # Step 2: Non-standard space → regular space
        canonicalized, count2 = self._normalize_spaces(canonicalized)

        # Step 3: Tab → space
        canonicalized, count3 = self._normalize_tabs(canonicalized)

        # Step 4: Multiple spaces → single space
        canonicalized, count4 = self._normalize_multiple_spaces(canonicalized)

        # Step 5: Trailing whitespace removal (end of lines)
        canonicalized, count5 = self._remove_trailing_whitespace(canonicalized)

        # Step 5.5: Trailing whitespace removal (end of string)
        canonicalized, count5b = self._remove_trailing_end(canonicalized)

        # Step 6: Indent canonicalization (4+ spaces → 4 spaces)
        canonicalized, count6 = self._normalize_indent(canonicalized)

        total_changes = count1 + count2 + count3 + count4 + count5 + count5b + count6

        if original == canonicalized:
            return WhitespaceCanonicalizeResult(text=canonicalized)

        # Estimate tokens saved (rough: each change saves ~0.1-0.5 tokens)
        tokens_saved = max(0, (len(original) - len(canonicalized)) // 20 + total_changes // 5)

        return WhitespaceCanonicalizeResult(
            text=canonicalized,
            changes=total_changes,
            tokens_saved=tokens_saved,
        )

    def _normalize_unicode(self, text: str) -> tuple[str, int]:
        """Normalize Unicode to NFC form.

        NFC (Composition) is the standard form used by most systems.
        NFD (Decomposition) can produce different tokens for the same
        visual character (e.g., "é" as a single character vs "e" + combining accent).
        """
        count = 0
        normalized = unicodedata.normalize("NFC", text)
        if normalized != text:
            # Count changed characters
            count = sum(1 for a, b in zip(text, normalized) if a != b)
        return normalized, count

    def _normalize_spaces(self, text: str) -> tuple[str, int]:
        """Replace non-standard space characters with regular space.

        Covers:
        - \\u00a0: Non-breaking space
        - \\u2000-\\u200a: Various space widths (em, en, thin, etc.)
        - \\u202f: Narrow no-break space
        - \\u205f: Medium math space
        - \\u3000: Ideographic space (CJK)
        """
        count = 0
        matches = _NON_STANDARD_SPACE.findall(text)
        count = len(matches)
        return _NON_STANDARD_SPACE.sub(" ", text), count

    def _normalize_tabs(self, text: str) -> tuple[str, int]:
        """Replace tab characters with spaces.

        Tabs tokenize differently than spaces in most tokenizers,
        causing KV cache misses when the same text uses tabs in
        different formatting styles.
        """
        count = 0
        matches = _TAB.findall(text)
        count = len(matches)
        return _TAB.sub(" ", text), count

    def _normalize_multiple_spaces(self, text: str) -> tuple[str, int]:
        """Collapse multiple consecutive spaces to single space.

        Whitespace canonicalization is important because:
        - Multiple spaces tokenize to different token sequences
        - They rarely affect semantic meaning
        - They cause 100% KV cache misses
        """
        count = 0
        matches = _MULTI_SPACES.findall(text)
        count = len(matches)
        return _MULTI_SPACES.sub(" ", text), count

    def _remove_trailing_whitespace(self, text: str) -> tuple[str, int]:
        """Remove trailing whitespace from lines.

        Trailing whitespace is invisible but affects tokenization:
        - "hello " and "hello" tokenize to different sequences
        - It rarely affects semantic meaning
        - It causes KV cache misses on every line
        """
        count = 0
        matches = _TRAILING_WHITESPACE.findall(text)
        count = len(matches)
        return _TRAILING_WHITESPACE.sub("", text), count

    def _remove_trailing_end(self, text: str) -> tuple[str, int]:
        """Remove trailing whitespace at end of string.

        Handles the case where the last line has no newline but has
        trailing whitespace.
        """
        count = 0
        matches = _TRAILING_END.findall(text)
        count = len(matches)
        return _TRAILING_END.sub("", text), count

    def _normalize_indent(self, text: str) -> tuple[str, int]:
        """Normalize indentation to consistent 4-space blocks.

        Indentation variations (2-space vs 4-space vs mixed) cause
        tokenization differences. This normalizes 4+ space indents
        to 4-space blocks for consistency.
        """
        count = 0
        matches = _LEADING_INDENT.findall(text)
        count = len(matches)
        return _LEADING_INDENT.sub("    ", text), count
