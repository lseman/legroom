"""Compressor registry and input/output types."""

from __future__ import annotations

import hashlib
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# LRU cache for _compute_salience (avoids recomputing unchanged content).
# Increased to 1024 for better hit rates in proxy mode where tool outputs repeat.
_salience_cache: dict[str, float] = {}
_SALIENCE_CACHE_MAX = 1024

# Stopwords set — created once at module load.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "or", "but",
    "if", "then", "else", "when", "that", "this", "these", "those",
    "it", "its", "i", "you", "he", "she", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "our", "their", "which",
    "what", "where", "who", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "about",
})


def _compute_salience(content: str) -> float:
    """Compute a salience score for content (0.0 = low importance, 1.0 = high importance).

    Scoring factors:
    - TF-style term weighting: rare words within the document get higher weight.
    - Position decay: tokens near the start of the document contribute more.
    - Information density: moderate-length documents score highest (very short
      ones lack context, very long ones dilute per-token importance).

    Returns a float in [0, 1] where higher means more semantically important.
    Non-string content (e.g. Anthropic content blocks) returns 0.0.
    """
    if not isinstance(content, str) or not content.strip():
        return 0.0

    # Check cache by content hash (fast path for unchanged messages)
    cache_key = hashlib.md5(content.encode()).hexdigest()[:16]
    cached = _salience_cache.get(cache_key)
    if cached is not None:
        return cached

    # Tokenize — strip common punctuation for comparison
    words = content.split()
    if not words:
        return 0.0

    n = len(words)

    # Clean words for comparison (strip surrounding punctuation)
    clean_words = [w.lower().strip(string.punctuation) for w in words if isinstance(w, str)]

    # --- TF-style term weighting ---
    # Count frequency of each unique word.  Rare-but-present words get high
    # weight (like IDF); very common stop-words get zero.
    freq = Counter(clean_words)
    tf_scores = []
    for w in clean_words:
        if w in _STOPWORDS or len(w) < 2:
            tf_scores.append(0.0)
            continue
        # Inverse frequency: rarer words score higher
        count = freq[w]
        tf_score = 1.0 / max(count, 1)
        # Bonus for longer (more specific) words
        tf_score *= min(len(w) / 4.0, 2.0)
        tf_scores.append(tf_score)

    # Normalize TF scores to [0, 1]
    if tf_scores:
        max_tf = max(tf_scores)
        tf_normalized = [s / max_tf for s in tf_scores] if max_tf > 0 else [0.0] * len(tf_scores)
    else:
        return 0.0

    # --- Position decay ---
    # Exponential decay: position p gets weight decay^p (decayed over 1/4 of doc length).
    # First ~25% of the document gets substantially more weight.
    decay_per_pos = 0.7 ** max(1, n // 4)
    pos_scores = [decay_per_pos ** (i / max(n // 4, 1)) for i in range(n)]

    # --- Document-length density bonus ---
    # Very short docs (< 50 words) lack context; very long ones (> 2000)
    # dilute per-token importance.  Peak at ~300-800.
    if n < 30:
        density_score = 0.4
    elif n < 100:
        density_score = 0.7 + 0.3 * (n - 30) / 70.0
    elif n <= 2000:
        density_score = 1.0 - 0.2 * min((n - 100) / 1900.0, 1.0)
    else:
        density_score = 0.8 * (1.0 + 100 / max(n, 1))

    # --- Combine per-token scores, then average ---
    weighted_scores = [
        tf_normalized[i] * pos_scores[i]
        for i in range(n)
    ]
    avg_score = sum(weighted_scores) / n if n > 0 else 0.0

    # Blend with document-level density score to avoid ultra-short docs getting 0
    salience = 0.8 * avg_score + 0.2 * min(density_score, 1.0)
    result = round(min(max(salience, 0.0), 1.0), 4)

    # Cache result (LRU — evict oldest when full)
    if len(_salience_cache) >= _SALIENCE_CACHE_MAX:
        # Evict the first key inserted
        _salience_cache.pop(next(iter(_salience_cache)))
    _salience_cache[cache_key] = result

    return result


@dataclass
class CompressInput:
    """Input to a compressor."""

    content: str
    content_type: str = "text"
    source_hint: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressOutput:
    """Output from a compressor."""

    compressed: str
    original_token_count: int = 0
    compressed_token_count: int = 0
    strategy: str = "unknown"
    routing_log: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_type: str | None = None

    @property
    def tokens_saved(self) -> int:
        return self.original_token_count - self.compressed_token_count

    @property
    def compression_ratio(self) -> float:
        if self.original_token_count == 0:
            return 0.0
        return self.compressed_token_count / self.original_token_count
