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
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any, ClassVar

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


def _load_tokenizer(vocab_path: str):
    """Simple tokenizer compatible with Xenova/all-MiniLM-L6-v2 ONNX models."""
    try:
        from tokenizers import Tokenizer as HfTokenizer

        return HfTokenizer.from_file(vocab_path)
    except Exception:  # noqa: BLE001
        # Fallback: naive word-piece tokenizer
        return None


class _SimpleTokenizer:
    """Minimal tokenizer for ONNX embedding models when ``tokenizers`` is
    unavailable. Handles basic word splitting and special token wrapping."""

    SPECIAL_TOKENS: ClassVar[dict[str, int]] = {
        "[PAD]": 0, "[CLS]": 101, "[SEP]": 102, "[UNK]": 100
    }

    def __init__(self, vocab_path: str) -> None:
        self._vocab: dict[str, int] = {}
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
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

    def encode(self, text: str, add_special_tokens: bool = True) -> dict:
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
        pad_count = max(0, max_len - len(ids))
        ids.extend([0] * pad_count)
        mask.extend([0] * pad_count)
        return {"input_ids": [ids], "attention_mask": [mask]}


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_text(
    text: str,
    session,
    tokenizer,
    cache: dict[str, list[float]],
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

        input_ids = np.array(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.array(encoded["attention_mask"], dtype=np.int64)

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
                from tokenizers import Tokenizer as HfTokenizer

                tokenizer = HfTokenizer.from_file(self._vocab_path)
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
    ) -> SemanticDedupResult:
        """Deduplicate semantically similar messages.

        Scans messages from oldest to newest (skipping recent ones). For
        each message, compares its embedding against all earlier messages.
        If similarity exceeds the threshold, the later message is replaced
        with a pointer.

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
        from ..tokenizer import count_tokens_messages

        tokens_before = count_tokens_messages(messages, "gpt-4o")

        # Determine which messages to check (skip recent ones)
        n = len(messages)
        check_start = max(0, n - self._protect_recent - 1)

        # Store embeddings and content for each message
        # Key: message index, Value: (embedding, original_content)
        known: dict[int, tuple[list[float], str]] = {}
        result = list(messages)  # shallow copy of message refs
        dedup_count = 0

        for i in range(check_start, n):
            msg = messages[i]
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue

            content_bytes = len(content.encode("utf-8"))
            if content_bytes < self._min_bytes:
                continue

            # Compute embedding
            embedding = _embed_text(content, self._session, self._tokenizer, _embedding_cache)
            if embedding is None:
                continue

            # Compare against all earlier known messages
            best_sim = 0.0
            best_idx = -1
            for j, (prev_embed, prev_content) in known.items():
                sim = _cosine_similarity(embedding, prev_embed)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j

            if best_sim >= self._threshold and best_idx >= 0:
                # Replace with pointer
                pointer = (
                    f"[semantically equivalent to block {best_idx} — "
                    f"{content_bytes} chars omitted, "
                    f"similarity={best_sim:.2f}]"
                )
                result[i] = {**msg, "content": pointer}
                dedup_count += 1
                logger.debug(
                    f"Semantic dedup: message {i} ≈ message {best_idx} "
                    f"(sim={best_sim:.2f})"
                )
            else:
                # Store this message for future comparison
                known[i] = (embedding, content)

        tokens_after = count_tokens_messages(result, "gpt-4o")

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
