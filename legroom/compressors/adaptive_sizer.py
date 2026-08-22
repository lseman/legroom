"""Adaptive sizer — auto-determines optimal K for JSON compression."""

from __future__ import annotations

import json
import hashlib
from collections import Counter


def compute_optimal_k(
    items: list[str],
    bias: float = 1.0,
    min_k: int = 1,
) -> int:
    """Compute optimal K for JSON array compression using Kneedle-style analysis.

    Args:
        items: List of JSON strings to analyze
        bias: Compression aggressiveness (lower = more aggressive)
        min_k: Minimum items to keep

    Returns:
        Optimal number of items to keep
    """
    if len(items) <= min_k:
        return len(items)

    # Fast path: small inputs
    if len(items) <= 8:
        return len(items)

    # Tier 1: Simhash near-duplicate check
    unique_count = count_unique_simhash(items)
    if unique_count <= min_k * 2:
        # High redundancy — keep fewer items
        return max(min_k, int(unique_count * bias))

    # Tier 2: Kneedle on bigram coverage
    k_values = _compute_bigram_coverage(items)

    # Find the knee point
    k = _find_knee(k_values)

    # Apply bias
    k = int(k * bias)
    return max(min_k, min(k, len(items)))


_SIMHASH_BITS = 64


def _simhash(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint over word shingles.

    Each shingle is hashed to 64 bits; the fingerprint bit at position i
    is set if more shingles have bit i = 1 than = 0 (weighted majority vote).
    Near-duplicate texts produce fingerprints with small Hamming distance.
    """
    words = text.lower().split()
    shingles = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)] or words
    if not shingles:
        return 0

    bit_votes = [0] * _SIMHASH_BITS
    for shingle in shingles:
        h = int.from_bytes(hashlib.sha256(shingle.encode()).digest()[:8], "big")
        for bit in range(_SIMHASH_BITS):
            if h & (1 << bit):
                bit_votes[bit] += 1
            else:
                bit_votes[bit] -= 1

    fingerprint = 0
    for bit, vote in enumerate(bit_votes):
        if vote > 0:
            fingerprint |= 1 << bit
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def count_unique_simhash(items: list[str], threshold: int = 10) -> int:
    """Count unique items using simhash-based near-duplicate detection.

    Items with simhash Hamming distance <= threshold are considered duplicates.
    """
    if not items:
        return 0

    fingerprints: list[int] = []
    unique = 0

    for item in items:
        fp = _simhash(item)
        if not any(_hamming_distance(fp, seen_fp) <= threshold for seen_fp in fingerprints):
            fingerprints.append(fp)
            unique += 1

    return unique


def _compute_bigram_coverage(items: list[str]) -> list[tuple[int, float]]:
    """Compute bigram coverage curve for Kneedle analysis."""
    all_bigrams: Counter = Counter()
    for item in items:
        words = item.lower().split()
        for i in range(len(words) - 1):
            all_bigrams[(words[i], words[i + 1])] += 1

    total_bigrams = len(all_bigrams)
    if total_bigrams == 0:
        return [(i, 1.0) for i in range(1, len(items) + 1)]

    # Sort bigrams by frequency
    sorted_bigrams = sorted(all_bigrams.items(), key=lambda x: x[1], reverse=True)

    coverage = []
    cumulative = set()
    for i in range(1, len(items) + 1):
        # Take top i bigrams
        for bg, _ in sorted_bigrams[:i]:
            cumulative.add(bg)
        coverage.append((i, len(cumulative) / total_bigrams))

    return coverage


def _find_knee(coverage: list[tuple[int, float]]) -> int:
    """Find the knee point in a coverage curve.

    When the curve never bows above the diagonal — i.e. items are diverse
    enough that there's no natural elbow — that itself means "no good
    compression point," so the fallback is to keep everything rather than
    collapsing to the initial candidate of 1.
    """
    if not coverage:
        return 1

    # Calculate the elbow using the maximum distance from the diagonal
    max_dist = 0
    knee_k = None

    for i, (k, coverage_pct) in enumerate(coverage):
        # Distance from the ideal diagonal (45 degree line)
        ideal_coverage = (i + 1) / len(coverage)
        dist = coverage_pct - ideal_coverage

        if dist > max_dist:
            max_dist = dist
            knee_k = k

    if knee_k is None:
        return len(coverage)
    return max(1, knee_k)
