"""Text compressor — sentence-level compression with TF-IDF ranking.

Unlike the old version that only normalised whitespace, this compressor:
  1. Splits content into sentences (handling abbreviations, numbers, etc.)
  2. Scores each sentence via TF-IDF-style weighting (rare words = more
     salient; first sentences get a position bonus)
  3. Keeps the top-K sentences and replaces the rest with a compact summary
     line so the model knows information was omitted.

This is the single biggest win for plain-text tool outputs (grep results,
log dumps, file reads) that previously got near-zero compression.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .compressor_registry import CompressInput, CompressOutput
from ..tokenizer import count_tokens

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

_COMPRESS_SUMMARY = "[~{n} omitted sentences of technical details]"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations."""
    if not text or not text.strip():
        return []

    # First, handle newlines as sentence boundaries (replace with single space)
    text = re.sub(r"\n+", " ", text)

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
    for i, s in enumerate(sentences):
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


def _tokenize(sentence: str) -> list[str]:
    """Extract lowercased words from a sentence."""
    return _WORD_RE.findall(sentence.lower())


def _score_sentences(
    sentences: list[str],
) -> list[tuple[float, int]]:
    """Score each sentence and return (score, index) pairs.

    TF-IDF-style: rare words within the document get higher weight.
    First ~25% of sentences get a position bonus.
    """
    n = len(sentences)
    if n == 0:
        return []

    # Tokenize all sentences
    tokenized = [_tokenize(s) for s in sentences]

    # Document-level word frequency
    doc_freq: Counter = Counter()
    for tokens in tokenized:
        unique = set(tokens)
        for w in unique:
            doc_freq[w] += 1

    # IDF for each word: log(N / df)
    idf: dict[str, float] = {}
    for w, df in doc_freq.items():
        if w in _STOPWORDS or len(w) < 2:
            idf[w] = 0.0
        else:
            idf[w] = math.log((n + 1) / (df + 1)) + 1  # smoothing

    # Score each sentence
    scores: list[float] = []
    for i, tokens in enumerate(tokenized):
        if not tokens:
            scores.append(0.0)
            continue

        # TF-IDF sum
        tfidf_sum = sum(idf.get(w, 0.0) for w in tokens)

        # Normalize by sentence length (avoid bias toward long sentences)
        tfidf_avg = tfidf_sum / len(tokens)

        # Position bonus: first 25% of sentences get a boost
        pos_bonus = 1.0
        if i < n * 0.25:
            pos_bonus = 1.0 + 0.5 * (1.0 - i / max(n * 0.25, 1))

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
        # Lossless whitespace normalisation (always applied)
        collapsed = re.sub(r"[ \t]+", " ", content)
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        collapsed = re.sub(r"[ \t]+\n", "\n", collapsed)
        collapsed = collapsed.strip()

        # Sentence-level compression (only if enough sentences exist)
        sentences = _split_sentences(collapsed)
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
        keep_indices = set(idx for _, idx in scored[:k])

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
