"""Text compressor — sentence-level compression with TF-IDF ranking.

Optimized with:
  1. Pre-computed IDF scores (cached per unique vocabulary set)
  2. Sentence splitting cache to avoid re-splitting identical content
  3. Combined whitespace normalization where possible
  4. Fast dict lookups for IDF scoring

This is the single biggest win for plain-text tool outputs (grep results,
log dumps, file reads) that previously got near-zero compression.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, OrderedDict

from ..analysis.tokenizer import count_tokens
from .compressor_registry import CompressOutput

# ---------------------------------------------------------------------------
# Sentence splitting — handles common English abbreviations and edge cases
# ---------------------------------------------------------------------------

# Common abbreviations that should NOT trigger a sentence break.
_ABBREVS = frozenset(
    [
        "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr",
        "vs", "etc",
        "e.g", "i.e",
        "Inc", "Ltd", "Corp", "Co", "St", "Ave", "Blvd",
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
        "US", "UK", "UN", "NATO", "WHO", "USA",
        "src", "lib", "bin", "etc",
    ]
)

# Regex: split on sentence-ending punctuation followed by whitespace and a
# capital letter or start-of-string.  We use a lookahead so the delimiter
# stays attached to the preceding sentence.
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])"
    r"\s+"
    r"(?=[A-Z\"\'\(\[])"  # next sentence starts with upper-case, quote, or paren
    r"|"
    r"\n+"  # newlines are also sentence boundaries
)

# Pre-compiled whitespace normalization patterns for text_compressor.compress
# Note: 3 separate re.sub calls are faster than a Python character loop
# because the C-optimized regex engine beats pure Python loops.
_WS_NORM = re.compile(r"[ \t]+")
_NL_TRIPLE = re.compile(r"\n{3,}")
_WS_NL = re.compile(r"[ \t]+\n")

_COMPRESS_SUMMARY = "[~{n} omitted sentences of technical details]"

# Sentence splitting cache (avoids re-splitting identical content)
_sentence_split_cache: OrderedDict[str, list[str]] = OrderedDict()
_SENTENCE_SPLIT_CACHE_MAX = 512


def _split_sentences_cached(text: str) -> list[str]:
    """Split text into sentences with LRU caching."""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    cached = _sentence_split_cache.get(cache_key)
    if cached is not None:
        _sentence_split_cache.move_to_end(cache_key)
        return cached

    result = _split_sentences(text)

    # Cache result (LRU — evict oldest when full)
    if len(_sentence_split_cache) >= _SENTENCE_SPLIT_CACHE_MAX:
        _sentence_split_cache.pop(next(iter(_sentence_split_cache)))
    _sentence_split_cache[cache_key] = result

    return result


_NL_TO_SPACE = re.compile(r"\n+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations."""
    if not text or not text.strip():
        return []

    # First, handle newlines as sentence boundaries (replace with single space)
    text = _NL_TO_SPACE.sub(" ", text)

    # Split on sentence-ending punctuation followed by whitespace + uppercase.
    # The lookbehind keeps the period attached to the preceding sentence.
    parts = _SENTENCE_SPLIT.split(text)

    # Clean up: strip whitespace, drop empty strings, and normalise trailing
    # punctuation so we don't get double periods when joining.
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Remove trailing periods/exclamation/question marks that the regex
        # kept attached — we'll add a single period back when joining.
        p = p.rstrip(".!? ")
        if p:
            sentences.append(p)

    # Merge very short fragments (< 3 words) into the previous sentence
    merged: list[str] = []
    for s in sentences:
        if len(s.split()) < 3 and merged:
            merged[-1] += " " + s
        else:
            merged.append(s)

    # Add trailing punctuation back (last sentence gets a period)
    for i in range(len(merged) - 1):
        merged[i] += "."
    if merged:
        merged[-1] += "."

    return merged


# ---------------------------------------------------------------------------
# TF-IDF-style scoring
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
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
)

_WORD_RE = re.compile(r"[a-z0-9_]+")

# IDF score cache keyed by frozenset of unique words (avoids recomputing IDF)
_idf_cache: dict[frozenset[str], dict[str, float]] = {}
_IDF_CACHE_MAX = 128


def _tokenize(sentence: str) -> list[str]:
    """Extract lowercased words from a sentence."""
    return _WORD_RE.findall(sentence.lower())


def _compute_idf_fast(
    n_sentences: int,
    all_tokenized: list[list[str]],
) -> dict[str, float]:
    """Compute IDF scores using cached vocabulary.

    Pre-computes IDF for all words in the document to avoid repeated
    lookups during scoring. Uses a frozenset key for caching.
    """
    # Create cache key from unique words
    unique_words: frozenset[str] = frozenset()
    for tokens in all_tokenized:
        unique_words = unique_words | set(tokens)

    cached = _idf_cache.get(unique_words)
    if cached is not None:
        return cached

    # Compute document frequency
    doc_freq: Counter = Counter()
    for tokens in all_tokenized:
        unique = set(tokens)
        for w in unique:
            doc_freq[w] += 1

    # Compute IDF for each word
    idf: dict[str, float] = {}
    for w, df in doc_freq.items():
        if w in _STOPWORDS or len(w) < 2:
            idf[w] = 0.0
        else:
            idf[w] = math.log((n_sentences + 1) / (df + 1)) + 1  # smoothing

    # Cache result (LRU — evict oldest when full)
    if len(_idf_cache) >= _IDF_CACHE_MAX:
        _idf_cache.pop(next(iter(_idf_cache)))
    _idf_cache[unique_words] = idf

    return idf


def _score_sentences(
    sentences: list[str],
) -> list[tuple[float, int]]:
    """Score each sentence using pre-computed IDF for speed.

    TF-IDF-style: rare words within the document get higher weight.
    First ~25% of sentences get a position bonus.
    """
    n = len(sentences)
    if n == 0:
        return []

    # Tokenize all sentences
    tokenized = [_tokenize(s) for s in sentences]

    # Compute IDF once for the entire document (cached)
    idf = _compute_idf_fast(n, tokenized)

    # Pre-compute position bonus factor (same for all sentences in first 25%)
    top_quarter = n * 0.25
    top_quarter_max = max(top_quarter, 1)

    # Score each sentence using fast dict lookups
    scores: list[float] = []
    for i, tokens in enumerate(tokenized):
        if not tokens:
            scores.append(0.0)
            continue

        # TF-IDF sum — direct dict lookup (fast in Python)
        tfidf_sum = 0.0
        for w in tokens:
            tfidf_sum += idf.get(w, 0.0)

        # Normalize by sentence length (avoid bias toward long sentences)
        tfidf_avg = tfidf_sum / len(tokens)

        # Position bonus: first 25% of sentences get a boost
        pos_bonus = 1.0
        if i < top_quarter:
            pos_bonus = 1.0 + 0.5 * (1.0 - i / top_quarter_max)

        # Length bonus: very short sentences (< 5 words) get a small penalty
        length_bonus = 1.0
        if len(tokens) < 5:
            length_bonus = 0.5 + 0.1 * len(tokens)

        score = tfidf_avg * pos_bonus * length_bonus
        scores.append(score)

    return [(s, i) for i, s in enumerate(scores)]


class TextCompressor:
    """Sentence-level text compression with TF-IDF ranking.

    For short content (< 5 sentences) falls back to lossless whitespace
    normalisation only — summarising tiny documents throws away too much.
    """

    def __init__(
        self,
        keep_ratio: float = 0.5,
        min_sentences: int = 3,
        max_sentences: int = 50,
    ) -> None:
        self.keep_ratio = keep_ratio
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def compress(
        self,
        content: str,
        source_hint: str = "text",
        model: str = "gpt-4o",
    ) -> CompressOutput:
        """Compress text via sentence-level TF-IDF ranking.

        Returns the original text unchanged if there are fewer than
        ``min_sentences`` sentences (nothing to summarise).
        """
        # Lossless whitespace normalisation (3 fast C-optimized passes)
        collapsed = _WS_NORM.sub(" ", content)
        collapsed = _NL_TRIPLE.sub("\n\n", collapsed)
        collapsed = _WS_NL.sub("\n", collapsed)
        collapsed = collapsed.strip()

        # Sentence-level compression (only if enough sentences exist)
        sentences = _split_sentences_cached(collapsed)
        n = len(sentences)

        if n < self.min_sentences:
            # Not enough sentences to summarise — return whitespace-normalised
            tokens_before = count_tokens(content, model)
            tokens_after = count_tokens(collapsed, model)
            return CompressOutput(
                compressed=collapsed,
                original_token_count=tokens_before,
                compressed_token_count=tokens_after,
                strategy="text_compressor",
            )

        # Score and rank sentences
        scored = _score_sentences(sentences)
        scored.sort(key=lambda x: x[0], reverse=True)

        # How many to keep
        k = max(self.min_sentences, min(int(n * self.keep_ratio), self.max_sentences))

        # Top-K indices
        keep_indices = {idx for _, idx in scored[:k]}

        # Build output: keep top sentences in original order, add summary for omitted
        output_parts: list[str] = []
        omitted_count = 0
        for i, s in enumerate(sentences):
            if i in keep_indices:
                output_parts.append(s)
            else:
                omitted_count += 1

        if omitted_count > 0:
            output_parts.append(_COMPRESS_SUMMARY.format(n=omitted_count))

        compressed = " ".join(output_parts)

        tokens_before = count_tokens(content, model)
        tokens_after = count_tokens(compressed, model)

        return CompressOutput(
            compressed=compressed,
            original_token_count=tokens_before,
            compressed_token_count=tokens_after,
            strategy="text_compressor",
        )
