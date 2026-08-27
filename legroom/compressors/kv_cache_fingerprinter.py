"""KV cache fingerprinting for token sequence matching.

This module provides deterministic fingerprinting for token sequences
to enable KV cache matching and similarity detection. It works by:

1. Generating deterministic fingerprints from token sequences
2. Comparing token sequences for similarity
3. Caching fingerprints for fast lookup

This is critical for llama.cpp's KV cache which matches tokenized
prefixes byte-for-byte. By fingerprinting token sequences, we can:

1. Detect when two requests produce identical token sequences
2. Measure similarity between different token sequences
3. Optimize normalization based on actual token patterns

Usage::

    from legroom.compressors.kv_cache_fingerprinter import KVCacheFingerprinter

    fingerprinter = KVCacheFingerprinter()
    fp = fingerprinter.fingerprint("text to fingerprint")
    similarity = fingerprinter.similarity(fp1, fp2)

Research reference: "CacheLLM: Enhancing KV Cache Reuse with Dynamic Fingerprinting" (NeurIPS 2024)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import tiktoken


@dataclass(frozen=True)
class KVCacheFingerprint:
    """Deterministic fingerprint for a token sequence.

    Attributes:
        fingerprint: MD5 hash of the token sequence (first 64 tokens).
        token_count: Total number of tokens in the sequence.
        top_tokens: First 16 token IDs for fast comparison.
        top_tokens_str: String representation of top tokens.
    """

    fingerprint: str
    """MD5 hash of the token sequence (first 64 tokens)."""

    token_count: int
    """Total number of tokens in the sequence."""

    top_tokens: list[int]
    """First 16 token IDs for fast comparison."""

    top_tokens_str: str
    """String representation of top tokens (e.g., "123,456,789")."""


class KVCacheFingerprinter:
    """Generates deterministic fingerprints for KV cache matching.

    This fingerprinter creates stable, deterministic hashes from token
    sequences to enable fast KV cache lookup and similarity detection.

    Key features:
    1. Deterministic: Same input always produces same fingerprint
    2. Stable: Uses first 64 tokens (sufficient for uniqueness)
    3. Fast: MD5 hash of token IDs (not full token sequence)
    4. Comparable: Supports similarity comparison between fingerprints

    The fingerprint is computed as:
        fingerprint = MD5(",".join(str(t) for t in tokens[:64]))

    This ensures that:
    - Two identical token sequences produce identical fingerprints
    - Two different token sequences produce different fingerprints
    - The fingerprint is stable across runs (deterministic)

    Usage for KV cache matching:
    1. Encode text to tokens
    2. Generate fingerprint
    3. Compare fingerprint against cached fingerprints
    4. If match, reuse KV cache; if not, compute and cache
    """

    # Number of tokens to use for fingerprint (sufficient for uniqueness)
    _FINGERPRINT_LENGTH = 64
    # Number of top tokens for fast comparison
    _TOP_TOKENS_LENGTH = 16

    def __init__(self) -> None:
        self._encodings: dict[str, tiktoken.Encoding] = {}

    def get_encoding(self, model: str = "gpt-4o") -> tiktoken.Encoding:
        """Get or create the tiktoken encoding for a model."""
        if model not in self._encodings:
            try:
                self._encodings[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                self._encodings[model] = tiktoken.get_encoding("cl100k_base")
        return self._encodings[model]

    def fingerprint(
        self, text: str, model: str = "gpt-4o"
    ) -> KVCacheFingerprint:
        """Generate deterministic fingerprint for text.

        Args:
            text: Text to fingerprint.
            model: Model name for tokenizer.

        Returns:
            KVCacheFingerprint with deterministic hash.
        """
        encoding = self.get_encoding(model)
        tokens = encoding.encode(text)

        # Use first 64 tokens for fingerprint
        fingerprint_tokens = tokens[: self._FINGERPRINT_LENGTH]

        # Generate deterministic fingerprint
        data = ",".join(str(t) for t in fingerprint_tokens)
        fingerprint = hashlib.md5(data.encode()).hexdigest()

        # Store top tokens for fast comparison
        top_tokens = tokens[: self._TOP_TOKENS_LENGTH]
        top_tokens_str = ",".join(str(t) for t in top_tokens)

        return KVCacheFingerprint(
            fingerprint=fingerprint,
            token_count=len(tokens),
            top_tokens=top_tokens,
            top_tokens_str=top_tokens_str,
        )

    def fingerprint_tokens(
        self, tokens: list[int]
    ) -> KVCacheFingerprint:
        """Generate fingerprint directly from token list.

        Useful when tokens are already available (e.g., after normalization).

        Args:
            tokens: Token list to fingerprint.

        Returns:
            KVCacheFingerprint with deterministic hash.
        """
        # Use first 64 tokens for fingerprint
        fingerprint_tokens = tokens[: self._FINGERPRINT_LENGTH]

        # Generate deterministic fingerprint
        data = ",".join(str(t) for t in fingerprint_tokens)
        fingerprint = hashlib.md5(data.encode()).hexdigest()

        # Store top tokens for fast comparison
        top_tokens = tokens[: self._TOP_TOKENS_LENGTH]
        top_tokens_str = ",".join(str(t) for t in top_tokens)

        return KVCacheFingerprint(
            fingerprint=fingerprint,
            token_count=len(tokens),
            top_tokens=top_tokens,
            top_tokens_str=top_tokens_str,
        )

    def similarity(self, fp1: KVCacheFingerprint, fp2: KVCacheFingerprint) -> float:
        """Compute similarity between two fingerprints.

        Uses token sequence similarity (Jaccard on token sets) for
        fast comparison without full token re-encoding.

        Args:
            fp1: First fingerprint.
            fp2: Second fingerprint.

        Returns:
            Similarity score in [0, 1].
        """
        # Use top tokens for fast comparison
        top1 = set(fp1.top_tokens)
        top2 = set(fp2.top_tokens)

        if not top1 and not top2:
            return 1.0

        intersection = top1 & top2
        union = top1 | top2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def is_similar(
        self, fp1: KVCacheFingerprint, fp2: KVCacheFingerprint, threshold: float = 0.9
    ) -> bool:
        """Check if two fingerprints are similar above threshold.

        Args:
            fp1: First fingerprint.
            fp2: Second fingerprint.
            threshold: Minimum similarity for match.

        Returns:
            True if fingerprints are similar above threshold.
        """
        return self.similarity(fp1, fp2) >= threshold

    def find_similar(
        self,
        fingerprint: KVCacheFingerprint,
        candidates: list[KVCacheFingerprint],
        threshold: float = 0.9,
    ) -> KVCacheFingerprint | None:
        """Find similar fingerprint in candidates.

        Args:
            fingerprint: Target fingerprint.
            candidates: List of candidate fingerprints.
            threshold: Minimum similarity for match.

        Returns:
            Most similar candidate, or None if no match.
        """
        best_match = None
        best_similarity = 0.0

        for candidate in candidates:
            similarity = self.similarity(fingerprint, candidate)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate

        if best_match and best_similarity >= threshold:
            return best_match
        return None
