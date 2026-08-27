"""Query relevance scoring — biases compression toward what the current turn needs.

Mirrors the shape of ``compressor_registry._compute_salience`` (same style of
plain lexical scoring, no external deps) but scores content against the
*current query* instead of scoring content in isolation. Compressors that are
query-aware can use this to keep items relevant to the live question intact
even when they'd otherwise be summarized away as redundant.
"""

from __future__ import annotations

import re
from typing import Any

_STOPWORDS = {
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
}

_WORD_RE = re.compile(r"[a-z0-9_]+")


def extract_query_terms(query: str) -> set[str]:
    """Extract lowercased, non-stopword terms from a query string.

    Terms shorter than 3 characters are dropped — they're mostly noise
    (articles, prepositions the stopword list missed) and rarely carry
    the kind of specific signal (an id, a name, a keyword) this is for.
    """
    if not query:
        return set()
    words = _WORD_RE.findall(query.lower())
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def query_relevance(content: str, query_terms: set[str]) -> float:
    """Score how relevant ``content`` is to a set of query terms, in [0, 1].

    Plain term-overlap scoring: the fraction of query terms that appear
    in the content, weighted slightly by how densely they appear. This is
    deliberately simple (no embeddings, no external calls) so it's cheap
    enough to run per-item in a compression hot path.
    """
    if not query_terms or not content:
        return 0.0

    content_words = _WORD_RE.findall(content.lower())
    if not content_words:
        return 0.0

    content_word_set = set(content_words)
    matched = query_terms & content_word_set
    if not matched:
        return 0.0

    coverage = len(matched) / len(query_terms)

    match_count = sum(1 for w in content_words if w in matched)
    density = min(match_count / len(content_words) * 10, 1.0)

    return min(0.75 * coverage + 0.25 * density, 1.0)


def latest_query_terms(messages: list[dict[str, Any]]) -> set[str]:
    """Extract query terms from the most recent user message, if any.

    Looks at the last message with role "user" walking backward, since
    that's the question the current compression pass should stay useful
    for. Returns an empty set (never scores anything as relevant) if
    there's no user message — callers should treat that as "no query
    signal available" and fall back to non-query-aware behavior.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return extract_query_terms(content)
            if isinstance(content, list):
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return extract_query_terms(" ".join(text_parts))
    return set()
