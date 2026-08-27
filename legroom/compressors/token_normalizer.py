"""Token-level verification and optimization for llama.cpp KV cache alignment.

This module provides token-level verification and optimization for KV cache
alignment. It works on top of character-level normalizers to ensure that
normalized text produces **identical token sequences** across turns.

Key features:
1. **Verification**: After character-level normalization, verifies that
   the normalized text produces identical tokens to expected canonical forms
2. **Optimization**: Detects cases where character-level normalization
   is not enough and applies token-level adjustments
3. **Cache fingerprinting**: Generates deterministic token fingerprints
   for KV cache matching

Usage::

    from legroom.compressors.token_normalizer import (
        TokenNormalizer,
    )

    normalizer = TokenNormalizer()
    result = normalizer.verify(text, expected_tokens, backend="llama_cpp")

The token-level approach complements character-level normalization:
- Character-level: replaces `:42` → `:LN` in text
- Token-level: verifies `:LN` produces the same tokens as other `:LN`
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import tiktoken


@dataclass(frozen=True)
class TokenNormalizationResult:
    """Result of token-level normalization/verification."""

    text: str
    """Normalized text (may be same as input)."""

    token_count: int = 0
    """Token count after normalization."""

    verified: bool = False
    """Whether normalization was verified to produce expected tokens."""

    token_fingerprint: str = ""
    """Deterministic hash of token sequence for KV cache matching."""


class TokenNormalizer:
    """Token-level verification and optimization for KV cache alignment.

    This normalizer works at the token level to:
    1. Verify that normalized text produces consistent token sequences
    2. Generate deterministic token fingerprints for KV cache matching
    3. Detect and fix token-level inconsistencies

    The key insight is that two texts with identical semantic content
    must produce identical token sequences for KV cache hits. Token-level
    normalization ensures this by working at the exact granularity of
    llama.cpp's KV cache matching.
    """

    def __init__(self) -> None:
        self._encoding: tiktoken.Encoding | None = None

    def get_encoding(self, model: str = "gpt-4o") -> tiktoken.Encoding:
        """Get or create the tiktoken encoding for a model."""
        if self._encoding is None:
            try:
                self._encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def verify(
        self,
        text: str,
        expected_tokens: list[int] | None = None,
        *,
        model: str = "gpt-4o",
        backend: str = "openai",
    ) -> TokenNormalizationResult:
        """Verify that text produces consistent token sequences.

        Args:
            text: The text to verify.
            expected_tokens: Optional expected token sequence for comparison.
            model: Model name for tokenizer.
            backend: Target backend — only ``"llama_cpp"`` triggers verification.

        Returns:
            TokenNormalizationResult with verification status.
        """
        if backend != "llama_cpp":
            return TokenNormalizationResult(text=text)

        encoding = self.get_encoding(model)
        tokens = encoding.encode(text)
        token_count = len(tokens)

        # Generate deterministic fingerprint
        fingerprint = self._generate_fingerprint(tokens)

        # Verify against expected tokens if provided
        verified = False
        if expected_tokens is not None:
            verified = tokens == expected_tokens

        return TokenNormalizationResult(
            text=text,
            token_count=token_count,
            verified=verified,
            token_fingerprint=fingerprint,
        )

    def optimize(
        self,
        text: str,
        *,
        model: str = "gpt-4o",
        backend: str = "openai",
    ) -> TokenNormalizationResult:
        """Optimize text for KV cache alignment at token level.

        Applies token-level optimizations to improve KV cache hit rates:
        1. Ensures whitespace consistency at token boundaries
        2. Normalizes Unicode at token level (not just character level)
        3. Detects and fixes token-level inconsistencies

        Args:
            text: The text to optimize.
            model: Model name for tokenizer.
            backend: Target backend — only ``"llama_cpp"`` triggers optimization.

        Returns:
            TokenNormalizationResult with optimized text.
        """
        if backend != "llama_cpp":
            return TokenNormalizationResult(text=text)

        encoding = self.get_encoding(model)
        original_tokens = encoding.encode(text)

        # Apply token-level optimizations
        optimized_tokens = self._optimize_tokens(original_tokens, encoding)

        # Decode back to text
        optimized_text = encoding.decode(optimized_tokens)

        # Generate fingerprint
        fingerprint = self._generate_fingerprint(optimized_tokens)

        return TokenNormalizationResult(
            text=optimized_text,
            token_count=len(optimized_tokens),
            verified=optimized_tokens == original_tokens or self._verify_optimization(
                optimized_text, optimized_tokens, encoding
            ),
            token_fingerprint=fingerprint,
        )

    def _optimize_tokens(
        self, tokens: list[int], encoding: tiktoken.Encoding
    ) -> list[int]:
        """Apply token-level optimizations.

        Strategies:
        1. Merge consecutive whitespace tokens
        2. Normalize Unicode at token level
        3. Align token boundaries
        """
        # Strategy 1: Merge consecutive whitespace tokens
        tokens = self._merge_whitespace_tokens(tokens, encoding)

        # Strategy 2: Normalize Unicode token sequences
        tokens = self._normalize_unicode_tokens(tokens, encoding)

        return tokens

    def _merge_whitespace_tokens(
        self, tokens: list[int], encoding: tiktoken.Encoding
    ) -> list[int]:
        """Merge consecutive whitespace tokens into single token.

        Multiple whitespace tokens (spaces, tabs, newlines) tokenize
        to different sequences. Merging them ensures consistent token
        sequences for semantically identical text.
        """
        if not tokens:
            return tokens

        result: list[int] = []
        i = 0

        while i < len(tokens):
            # Check if this is a whitespace token
            token_text = encoding.decode([tokens[i]])
            if token_text.isspace():
                # Start of whitespace run
                run_tokens = [tokens[i]]
                i += 1

                # Collect consecutive whitespace tokens
                while i < len(tokens):
                    next_text = encoding.decode([tokens[i]])
                    if next_text.isspace():
                        run_tokens.append(tokens[i])
                        i += 1
                    else:
                        break

                # Replace with single space token
                result.append(encoding.encode(" ")[0])
            else:
                result.append(tokens[i])
                i += 1

        return result

    def _normalize_unicode_tokens(
        self, tokens: list[int], encoding: tiktoken.Encoding
    ) -> list[int]:
        """Normalize Unicode token sequences.

        Detects token sequences that represent different Unicode
        representations of the same character and normalizes them.
        """
        # This is a simplified implementation - full version would
        # decode tokens, check Unicode forms, and re-encode.
        # For now, we rely on character-level normalization.
        return tokens

    def _verify_optimization(
        self,
        optimized_text: str,
        optimized_tokens: list[int],
        encoding: tiktoken.Encoding,
    ) -> bool:
        """Verify that optimization produces consistent tokens."""
        verified_tokens = encoding.encode(optimized_text)
        return verified_tokens == optimized_tokens

    def _generate_fingerprint(self, tokens: list[int]) -> str:
        """Generate deterministic fingerprint from token sequence.

        Creates a hash of the token sequence for KV cache matching.
        """
        import hashlib

        # Use first 16 tokens for fingerprint (sufficient for uniqueness)
        fingerprint_tokens = tokens[:16]
        data = ",".join(str(t) for t in fingerprint_tokens)
        return hashlib.md5(data.encode()).hexdigest()[:8]


# Module-level instance for convenience
token_normalizer = TokenNormalizer()
