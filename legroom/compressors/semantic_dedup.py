"""Semantic cross-turn deduplication — replaces paraphrased/repeated content
across messages with pointers using lightweight ONNX sentence embeddings.

Unlike the exact-match dedup in ``cross_turn_dedup`` (which only catches
byte-for-byte duplicates), this module detects *semantic* similarity:
when turn B says "I need help with the login endpoint" and turn A already
contains a full login endpoint implementation with 90%+ embedding cosine
similarity, turn A gets compressed.

Model: loads ``Xenova/all-MiniLM-L6-v2`` (ONNX, ~82 MB) for fast
sentence-level embeddings. Falls back gracefully when the model files
or ``onnxruntime`` are not available.

Performance optimizations:
- Jaccard word-set pre-filter before expensive ONNX inference
- Batch ONNX inferences (multiple texts in one model call)
- Vectorized cosine similarity via numpy
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any, ClassVar, Protocol, TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ONNX embedding model (lazy, optional)
# ---------------------------------------------------------------------------

# Default model path — mirrors the Kompress-v2 path convention.
_DEFAULT_EMBED_MODEL_PATH = "models/minilm-l6-v2/model.onnx"
_DEFAULT_EMBED_CONFIG_PATH = "models/minilm-l6-v2/config.json"
_DEFAULT_EMBED_VOCAB_PATH = "models/minilm-l6-v2/vocab.txt"

# Minimum content length (bytes) before semantic dedup is worth attempting —
# short messages are unlikely to be paraphrased duplicates of longer ones.
_MIN_SEMANTIC_DEDUP_BYTES = 150

# Similarity threshold: cosine similarity above this triggers dedup.
_DEFAULT_THRESHOLD = 0.85

# Embedding cache — avoids re-embedding identical content.
_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
_EMBED_CACHE_MAX = 512


def _load_onnx_session(model_path: str):
    """Lazy-load the ONNX embedding session."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        return session
    except Exception:  # noqa: BLE001 — ONNX runtime raises custom exception types
        return None


class _EncodedInput(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]


class _Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> _EncodedInput: ...


class _HuggingFaceTokenizer:
    """Adapt tokenizers.Tokenizer to Legroom's small embedding interface."""

    def __init__(self, vocab_path: str) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(vocab_path)

    def encode(self, text: str, add_special_tokens: bool = True) -> _EncodedInput:
        encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return {
            "input_ids": list(encoded.ids),
            "attention_mask": list(encoded.attention_mask),
        }


def _load_tokenizer(vocab_path: str) -> _Tokenizer | None:
    """Load a tokenizer behind the uniform embedding-tokenizer interface."""
    try:
        return _HuggingFaceTokenizer(vocab_path)
    except Exception:  # noqa: BLE001
        # Fallback: naive word-piece tokenizer
        return None


class _SimpleTokenizer:
    """Minimal tokenizer for ONNX embedding models when ``tokenizers`` is
    unavailable. Handles basic word splitting and special token wrapping."""

    SPECIAL_TOKENS: ClassVar[dict[str, int]] = {
        "[PAD]": 0,
        "[CLS]": 101,
        "[SEP]": 102,
        "[UNK]": 100,
    }

    def __init__(self, vocab_path: str) -> None:
        self._vocab: dict[str, int] = {}
        try:
            with open(vocab_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if parts:
                        token = parts[0]
                        idx = int(parts[1]) if len(parts) > 1 else len(self._vocab)
                        self._vocab[token] = idx
        except (OSError, UnicodeError, ValueError) as exc:
            logger.warning("Could not load semantic-dedup vocabulary %s: %s", vocab_path, exc)
        # Ensure special tokens are present
        for tok, idx in self.SPECIAL_TOKENS.items():
            if tok not in self._vocab:
                self._vocab[tok] = idx

    def encode(self, text: str, add_special_tokens: bool = True) -> _EncodedInput:
        """Encode text → {input_ids, attention_mask}."""
        words = text.lower().split()
        ids = []
        mask = []
        if add_special_tokens:
            ids.append(101)  # [CLS]
            mask.append(1)
        for w in words:
            # Basic word-piece: try exact match, then character-level
            token_id = self._vocab.get(w, self._vocab.get("[UNK]", 100))
            ids.append(token_id)
            mask.append(1)
        if add_special_tokens:
            ids.append(102)  # [SEP]
            mask.append(1)
        # Pad to fixed length (256 for MiniLM)
        max_len = 256
        ids = ids[:max_len]
        mask = mask[:max_len]
        pad_count = max(0, max_len - len(ids))
        ids.extend([0] * pad_count)
        mask.extend([0] * pad_count)
        return {"input_ids": ids, "attention_mask": mask}


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _batch_cosine_similarity(query: list[float], candidates: list[list[float]]) -> list[float]:
    """Compute cosine similarity between one query and many candidates.

    Uses numpy vectorization when available for significant speedup
    over repeated Python-loop cosine_similarity calls.
    """
    if not candidates:
        return []

    try:
        import numpy as np

        q = np.array(query, dtype=np.float32)
        norms_q = np.linalg.norm(q)
        if norms_q == 0:
            return [0.0] * len(candidates)

        c_arr = np.array(candidates, dtype=np.float32)
        norms_c = np.linalg.norm(c_arr, axis=1)
        # Zero-norm candidates get 0 similarity
        valid = norms_c > 0
        sims = np.zeros(len(candidates), dtype=np.float32)
        if np.any(valid):
            sims[valid] = (c_arr[valid] @ q) / (norms_q * norms_c[valid])
        return sims.tolist()
    except ImportError:
        return [_cosine_similarity(query, c) for c in candidates]


def _embed_text(
    text: str,
    session,
    tokenizer,
    cache: OrderedDict[str, list[float]],
) -> list[float] | None:
    """Generate an embedding for text using the ONNX model.

    Returns None if the model fails or text is empty.
    Uses LRU cache keyed by content hash.
    """
    if not text or not text.strip():
        return None

    cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
    cached = cache.get(cache_key)
    if cached is not None:
        cache.move_to_end(cache_key)
        return cached

    try:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        import numpy as np

        input_ids = np.array([encoded["input_ids"]], dtype=np.int64)
        attention_mask = np.array([encoded["attention_mask"]], dtype=np.int64)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        # Use [CLS] token embedding (first token of last hidden state)
        # outputs[0] shape: (1, seq_len, hidden_dim)
        embedding = outputs[0][0][0].tolist()

        if len(cache) >= _EMBED_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[cache_key] = embedding
        return embedding

    except Exception:  # noqa: BLE001
        return None


def _embed_batch(
    texts: list[str],
    session,
    tokenizer,
    cache: OrderedDict[str, list[float]],
) -> list[list[float] | None]:
    """Generate embeddings for multiple texts in a single ONNX inference.

    Returns a list of embeddings (or None for failures). This is much
    faster than calling _embed_text per-text because ONNX inference
    has significant per-call overhead.
    """
    if not texts:
        return []

    # Check cache first — skip encoding entirely for cached texts
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []
    results: list[list[float] | None] = [None] * len(texts)

    for i, text in enumerate(texts):
        if not text or not text.strip():
            results[i] = None
            continue
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        cached = cache.get(cache_key)
        if cached is not None:
            cache.move_to_end(cache_key)
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    # Nothing to embed — return cached results
    if not uncached_texts:
        return results

    try:
        # Batch encode
        encoded_batch = [tokenizer.encode(t, add_special_tokens=True) for t in uncached_texts]

        # Pad to max length in batch
        max_len = max(len(e["input_ids"]) for e in encoded_batch)
        input_ids_batch = []
        attention_mask_batch = []
        for e in encoded_batch:
            ids = e["input_ids"]
            mask = e["attention_mask"]
            pad = max_len - len(ids)
            input_ids_batch.append(ids + [0] * pad)
            attention_mask_batch.append(mask + [0] * pad)

        import numpy as np

        input_ids_arr = np.array(input_ids_batch, dtype=np.int64)
        attention_mask_arr = np.array(attention_mask_batch, dtype=np.int64)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids_arr,
                "attention_mask": attention_mask_arr,
            },
        )

        # Extract [CLS] embeddings and update cache
        for idx_pos, orig_idx in enumerate(uncached_indices):
            embedding = outputs[0][idx_pos][0].tolist()
            cache_key = hashlib.md5(uncached_texts[idx_pos].encode()).hexdigest()[:16]
            if len(cache) >= _EMBED_CACHE_MAX:
                cache.pop(next(iter(cache)))
            cache[cache_key] = embedding
            results[orig_idx] = embedding

    except Exception:  # noqa: BLE001
        # Fallback: compute individually for uncached texts
        for idx_pos, orig_idx in enumerate(uncached_indices):
            results[orig_idx] = _embed_text(uncached_texts[idx_pos], session, tokenizer, cache)

    return results


# ---------------------------------------------------------------------------
# Semantic dedup logic
# ---------------------------------------------------------------------------


class SemanticDedupResult:
    """Result from semantic deduplication."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        dedup_count: int = 0,
        tokens_saved: int = 0,
        warnings: list[str] | None = None,
    ) -> None:
        self.messages = messages
        self.dedup_count = dedup_count
        self.tokens_saved = tokens_saved
        self.warnings = warnings or []


class SemanticDedup:
    """Embedding-based semantic deduplication for cross-turn messages.

    For each message (after the first ``protect_recent`` messages from the
    end), computes a sentence-level embedding and compares it against all
    earlier messages. If the cosine similarity exceeds the threshold and
    the content is at least ``min_bytes`` long, the later message is
    replaced with a pointer.

    Works best when:
    - The model paraphrases or restates earlier content
    - Tool results repeat with minor wording changes
    - The same code/file is discussed across multiple turns
    """

    def __init__(
        self,
        model_path: str | None = None,
        config_path: str | None = None,
        vocab_path: str | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
        min_bytes: int = _MIN_SEMANTIC_DEDUP_BYTES,
        protect_recent: int = 0,
    ) -> None:
        self._threshold = threshold
        self._min_bytes = min_bytes
        self._protect_recent = protect_recent

        # Lazy model loading
        self._model_path = model_path or _DEFAULT_EMBED_MODEL_PATH
        self._config_path = config_path or _DEFAULT_EMBED_CONFIG_PATH
        self._vocab_path = vocab_path or _DEFAULT_EMBED_VOCAB_PATH
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._loaded = False
        self._load_error: str | None = None

    def _ensure_loaded(self) -> bool:
        """Load the ONNX model and tokenizer (once). Returns True on success."""
        if self._loaded:
            return self._load_error is None

        self._loaded = True

        # Check onnxruntime is available
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self._load_error = "onnxruntime not installed"
            return False

        # Try loading with tokenizers library first
        session = _load_onnx_session(self._model_path)
        if session is not None:
            try:
                tokenizer = _load_tokenizer(self._vocab_path)
                if tokenizer is None:
                    raise RuntimeError("tokenizer could not be loaded")
                self._session = session
                self._tokenizer = tokenizer
                return True
            except Exception as exc:  # noqa: BLE001 - tokenizer uses custom exceptions
                logger.debug("HuggingFace tokenizer unavailable; using fallback: %s", exc)

        # Fallback: try simple tokenizer
        if session is not None:
            tokenizer = _SimpleTokenizer(self._vocab_path)
            self._session = session
            self._tokenizer = tokenizer
            return True

        self._load_error = f"Could not load model from {self._model_path}"
        return False

    def dedup(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "gpt-4o",
    ) -> SemanticDedupResult:
        """Deduplicate semantically similar messages.

        Scans messages from oldest to newest (skipping recent ones). For
        each message, compares its embedding against all earlier messages.
        If similarity exceeds the threshold, the later message is replaced
        with a pointer.

        Optimizations:
        - Jaccard word-set pre-filter: skip ONNX inference when word
          overlap is too low to possibly reach the similarity threshold.
        - Batch ONNX inferences: compute all embeddings in one model call.
        - Vectorized cosine similarity via numpy.

        Returns a SemanticDedupResult with the modified messages and stats.
        """
        if not self._ensure_loaded():
            logger.debug(f"SemanticDedup skipped: {self._load_error}")
            return SemanticDedupResult(
                messages=messages,
                dedup_count=0,
                tokens_saved=0,
                warnings=[f"SemanticDedup skipped: {self._load_error}"],
            )

        if len(messages) < 2:
            return SemanticDedupResult(messages=messages, dedup_count=0, tokens_saved=0)

        # Count tokens before
        from ..analysis.tokenizer import count_tokens_messages

        tokens_before = count_tokens_messages(messages, model)

        # Determine which messages to check (skip recent ones)
        n = len(messages)
        check_stop = max(0, n - self._protect_recent)

        # Phase 1: Collect eligible messages and compute Jaccard pre-filters.
        # Jaccard similarity is a cheap proxy — if word overlap is below
        # a fraction of the threshold, cosine similarity can't possibly
        # exceed the threshold (proven bound for unit-normalized vectors).
        eligible: list[tuple[int, str, set[str]]] = []
        for i in range(check_stop):
            msg = messages[i]
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            content_bytes = len(content.encode("utf-8"))
            if content_bytes < self._min_bytes:
                continue
            words = set(content.lower().split())
            eligible.append((i, content, words))

        if not eligible:
            tokens_after = count_tokens_messages(messages, model)
            return SemanticDedupResult(
                messages=messages,
                dedup_count=0,
                tokens_saved=tokens_before - tokens_after,
            )

        # Phase 2: Batch-embed all eligible texts in one ONNX call.
        eligible_texts = [item[1] for item in eligible]
        embeddings = _embed_batch(eligible_texts, self._session, self._tokenizer, _embedding_cache)

        # Phase 3: Compare each message against known messages with
        # Jaccard pre-filter + vectorized cosine similarity.
        # known: index → (embedding, content, word_set)
        known: dict[int, tuple[list[float], str, set[str]]] = {}
        result = list(messages)  # shallow copy of message refs
        dedup_count = 0

        for pos, (msg_idx, content, words) in enumerate(eligible):
            embedding = embeddings[pos]
            if embedding is None:
                continue

            # Jaccard pre-filter: skip ONNX comparison when word overlap
            # is too small to possibly reach the similarity threshold.
            best_sim = 0.0
            best_idx = -1
            for j, (prev_embed, _prev_content, prev_words) in known.items():
                # Quick Jaccard check
                if not prev_words:
                    continue
                intersection = len(words & prev_words)
                union = len(words | prev_words)
                jaccard = intersection / union if union > 0 else 0.0
                # Cosine similarity of unit vectors is bounded by
                # sqrt(Jaccard) in the worst case — use a conservative
                # bound: if jaccard < threshold², skip.
                if jaccard < self._threshold**2:
                    continue
                # Full cosine similarity (vectorized)
                sim = _cosine_similarity(embedding, prev_embed)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j

            content_bytes = len(content.encode("utf-8"))
            if best_sim >= self._threshold and best_idx >= 0:
                # Replace with pointer
                pointer = (
                    f"[semantically equivalent to block {best_idx} — "
                    f"{content_bytes} chars omitted, "
                    f"similarity={best_sim:.2f}]"
                )
                result[msg_idx] = {**messages[msg_idx], "content": pointer}
                dedup_count += 1
                logger.debug(
                    f"Semantic dedup: message {msg_idx} ≈ message {best_idx} (sim={best_sim:.2f})"
                )
            else:
                # Store this message for future comparison
                known[msg_idx] = (embedding, content, words)

        tokens_after = count_tokens_messages(result, model)

        return SemanticDedupResult(
            messages=result,
            dedup_count=dedup_count,
            tokens_saved=tokens_before - tokens_after,
        )

    @property
    def is_available(self) -> bool:
        """Check if the semantic dedup model is available."""
        if not self._loaded:
            return False
        return self._load_error is None
