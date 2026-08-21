"""Compressor registry and input/output types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _compute_salience(content: str) -> float:
    """Compute a salience score for content (0.0 = low importance, 1.0 = high importance).

    Scoring factors:
    - Word frequency (uncommon words = more salient)
    - Position (first 25% of content gets a boost)
    - Role weighting (applied at pipeline level)

    Returns a float in [0, 1] where higher means more semantically important.
    """
    if not content or not content.strip():
        return 0.0

    words = content.split()
    if not words:
        return 0.0

    n = len(words)

    # Raw word count (higher = more information density)
    word_score = min(n / 500.0, 1.0)  # Cap at 500 words

    # Uncommon word ratio (technical terms, proper nouns are more salient)
    # Common English words tend to be 4-6 chars; uncommon are longer or have special chars
    common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                    "have", "has", "had", "do", "does", "did", "will", "would", "could",
                    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
                    "on", "with", "at", "by", "from", "as", "into", "through", "during",
                    "before", "after", "above", "below", "between", "and", "or", "but",
                    "if", "then", "else", "when", "that", "this", "these", "those",
                    "it", "its", "i", "you", "he", "she", "we", "they", "me", "him",
                    "her", "us", "them", "my", "your", "his", "our", "their", "which",
                    "what", "where", "who", "how", "all", "each", "every", "both",
                    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                    "only", "own", "same", "so", "than", "too", "very", "just", "about"
                    }
    uncommon_ratio = sum(1 for w in words if w.lower().strip(".,;:!?\"'()[]{}") not in common_words) / max(n, 1)

    # Position bonus (first portion of content gets higher weight)
    head_ratio = min(n / (n * 0.25), 1.0) if n > 0 else 0
    position_score = 0.6 * min(1.0, 1.0 / (n / 100.0)) if n < 100 else 0.3

    # Combined salience
    salience = 0.4 * word_score + 0.35 * uncommon_ratio + 0.25 * position_score
    return min(max(salience, 0.0), 1.0)


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

    @property
    def tokens_saved(self) -> int:
        return self.original_token_count - self.compressed_token_count

    @property
    def compression_ratio(self) -> float:
        if self.original_token_count == 0:
            return 0.0
        return self.compressed_token_count / self.original_token_count
