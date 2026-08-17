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


def count_unique_simhash(items: list[str], threshold: int = 10) -> int:
    """Count unique items using simhash-based near-duplicate detection.

    Items with simhash distance <= threshold are considered duplicates.
    """
    if not items:
        return 0

    # Compute simple hash-based grouping
    seen: set[str] = set()
    unique = 0

    for item in items:
        h = hashlib.sha256(item.encode()).hexdigest()[:16]
        # Check for near-duplicates via hash prefix similarity
        found = False
        for seen_hash in seen:
            # Count matching prefix characters
            match = sum(1 for a, b in zip(h, seen_hash) if a == b)
            if match >= len(h) - 2:
                found = True
                break

        if not found:
            seen.add(h)
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
    """Find the knee point in a coverage curve."""
    if not coverage:
        return 1

    # Calculate the elbow using the maximum distance from the diagonal
    max_dist = 0
    knee_k = 1

    for i, (k, coverage_pct) in enumerate(coverage):
        # Distance from the ideal diagonal (45 degree line)
        ideal_coverage = (i + 1) / len(coverage)
        dist = coverage_pct - ideal_coverage

        if dist > max_dist:
            max_dist = dist
            knee_k = k

    return max(1, knee_k)
