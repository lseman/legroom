"""ML-based token-level retention compression using Kompress-v2-base."""

from __future__ import annotations

import json
import numpy as np
from typing import Any, Optional

from .compressor_registry import CompressInput, CompressOutput

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer as HfTokenizer
    _HAS_ML = True
except ImportError:
    _HAS_ML = False


class MLTextCompressor:
    """Compresses text using ML token-level retention scoring.

    Uses the Kompress-v2-base model (ONNX format) to score tokens
    by importance and drops low-score tokens, reconstructing text.
    """

    def __init__(self, model_path: str | None = None, tokenizer_path: str | None = None) -> None:
        if not _HAS_ML:
            raise ImportError(
                "ML features require onnxruntime, numpy, and tokenizers. "
                "Install with: pip install legroom[ml]"
            )

        self._model_path = model_path or "models/kompress-v2-base/onnx/kompress-fp32.onnx"
        self._tokenizer_path = tokenizer_path or "models/kompress-v2-base/tokenizer.json"

        self._session: Optional[ort.InferenceSession] = None
        self._tokenizer: Optional[HfTokenizer] = None
        self._retention_threshold: float = 0.5
        self._min_compression_ratio: float = 0.1

        self._load_model()

    def _load_model(self) -> None:
        """Load the ONNX model and tokenizer."""
        try:
            self._session = ort.InferenceSession(
                self._model_path,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = HfTokenizer.from_file(self._tokenizer_path)
        except Exception:
            self._session = None
            self._tokenizer = None

    def compress(self, content: str, source_hint: str = "ml") -> CompressOutput:
        """Compress text using ML retention scoring."""
        if not _HAS_ML or self._session is None or self._tokenizer is None:
            # Fallback to basic compression
            return CompressOutput(
                compressed=content,
                original_token_count=len(content) // 4,
                compressed_token_count=len(content) // 4,
                strategy="ml_compressor_fallback",
            )

        # Tokenize
        encoding = self._tokenizer.encode(content, add_special_tokens=True)
        tokens = encoding.ids
        attention_mask = [1] * len(tokens)

        if not tokens:
            return CompressOutput(
                compressed=content,
                original_token_count=len(content) // 4,
                compressed_token_count=len(content) // 4,
                strategy="ml_compressor_empty",
            )

        # Run inference
        try:
            input_ids = np.array([tokens], dtype=np.int64)
            attention_mask_arr = np.array([attention_mask], dtype=np.int64)

            outputs = self._session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask_arr},
            )

            # Get retention scores
            scores = outputs[0][0]  # shape: (1, seq_len)

            # Drop low-score tokens (keep special tokens)
            if len(scores) > 2:
                non_special_scores = scores[1:-1]
                threshold = self._retention_threshold
                keep_mask = non_special_scores >= threshold

                # Calculate compression ratio
                ratio = keep_mask.sum() / len(keep_mask)

                # Apply minimum compression ratio guard
                if ratio < (1.0 - self._min_compression_ratio):
                    # Don't over-compress — adjust threshold
                    threshold = np.percentile(non_special_scores, self._min_compression_ratio * 100)
                    keep_mask = non_special_scores >= threshold

                # Reconstruct tokens
                kept_tokens = [tokens[0]] + [
                    t for i, t in enumerate(tokens[1:-1]) if keep_mask[i]
                ] + [tokens[-1]]

                # Decode
                compressed = self._tokenizer.decode(kept_tokens, skip_special_tokens=True)
                tokens_before = len(content) // 4
                tokens_after = len(compressed) // 4

                return CompressOutput(
                    compressed=compressed,
                    original_token_count=tokens_before,
                    compressed_token_count=tokens_after,
                    strategy="ml_compressor",
                )

        except Exception:
            pass

        # Fallback
        return CompressOutput(
            compressed=content,
            original_token_count=len(content) // 4,
            compressed_token_count=len(content) // 4,
            strategy="ml_compressor_fallback",
        )
