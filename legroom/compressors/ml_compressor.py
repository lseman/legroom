"""ML-based token-level retention compression using Kompress-v2-base.

Unlike naive token-dropping (which produces garbled fragments the model
cannot parse), this compressor keeps high-score tokens fully intact and
replaces low-score *runs* of tokens with compact summary-into-placeholders.
The result is readable at a glance — LLMs know something was compressed
behind a marker without being presented with broken text.
"""

from __future__ import annotations

from ..tokenizer import count_tokens
from .compressor_registry import CompressOutput

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer as HfTokenizer
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

# Placeholder format: [COMPRESSED:N chars, ~M tokens] — replaces low-score
# runs.  The model can still read the surrounding intact text.
_COMPRESS_MARKER = "[COMPRESSED:~{chars} chars of technical details replaced]"


class MLTextCompressor:
    """ML token-retention compressor using a placeholder strategy.

    Scoring pipeline (per-token from Kompress-v2-base):
      1. Run ONNX model → retention score per non-special token
      2. Merge low-score runs into contiguous spans
      3. Replace each span with [COMPRESSED:N chars ...]

    This avoids the "Th capial of Frnce is Par." problem entirely: all
    retained text is decoded exactly as-is, so the model reads clean prose.
    """

    def __init__(
        self,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        retention_threshold: float = 0.5,
        min_compression_ratio: float = 0.1,
        min_run_length: int = 4,   # minimum contiguous low-score tokens to merge
    ) -> None:
        if not _HAS_ML:
            raise ImportError(
                "ML features require onnxruntime, numpy, and tokenizers. "
                "Install with: pip install legroom[ml]"
            )

        self._model_path = model_path or "models/kompress-v2-base/onnx/kompress-fp32.onnx"
        self._tokenizer_path = tokenizer_path or "models/kompress-v2-base/tokenizer.json"

        self._session: ort.InferenceSession | None = None
        self._tokenizer: HfTokenizer | None = None
        self._retention_threshold = retention_threshold
        self._min_compression_ratio = min_compression_ratio
        self._min_run_length = min_run_length

        self._load_model()

    def _load_model(self) -> None:
        """Load the ONNX model and tokenizer."""
        try:
            self._session = ort.InferenceSession(
                self._model_path,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = HfTokenizer.from_file(self._tokenizer_path)
        except Exception:  # noqa: BLE001 - ONNX/tokenizer backends use custom exceptions
            self._session = None
            self._tokenizer = None

    def compress(
        self, content: str, source_hint: str = "ml", model: str = "gpt-4o"
    ) -> CompressOutput:
        """Compress text using ML retention scoring with placeholder replacement.

        If the model cannot run or the optional deps are missing, falls back
        to lossless whitespace normalisation (the TextCompressor).
        """
        if not _HAS_ML or self._session is None or self._tokenizer is None:
            return CompressOutput(
                compressed=content,
                original_token_count=count_tokens(content, model),
                compressed_token_count=count_tokens(content, model),
                strategy="ml_compressor_fallback",
            )

        encoding = self._tokenizer.encode(content, add_special_tokens=True)
        tokens = encoding.ids

        if not tokens or len(tokens) <= 2:
            return CompressOutput(
                compressed=content,
                original_token_count=count_tokens(content, model),
                compressed_token_count=count_tokens(content, model),
                strategy="ml_compressor_empty",
            )

        try:
            import numpy as np

            input_ids = np.array([tokens], dtype=np.int64)
            attention_mask_arr = np.array([[1] * len(tokens)], dtype=np.int64)

            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask_arr,
                },
            )

            scores = outputs[0][0]  # shape: (1, seq_len)

        except Exception:  # noqa: BLE001 - inference backends use custom exceptions
            return CompressOutput(
                compressed=content,
                original_token_count=count_tokens(content, model),
                compressed_token_count=count_tokens(content, model),
                strategy="ml_compressor_fallback",
            )

        # ---- Step 1: classify each non-special token as keep / drop ----------
        if len(scores) <= 2:
            return CompressOutput(
                compressed=content,
                original_token_count=count_tokens(content, model),
                compressed_token_count=count_tokens(content, model),
                strategy="ml_compressor_fallback",
            )

        non_special_scores = scores[1:-1]  # strip [CLS], [SEP]
        n_ns = len(non_special_scores)

        # Auto-adjust threshold if raw threshold compresses too aggressively
        threshold = self._retention_threshold
        keep_ratio = float((non_special_scores >= threshold).mean())
        min_keep = 1.0 - self._min_compression_ratio

        if keep_ratio < min_keep and n_ns > 1:
            # Lower threshold until we keep at least min_keep fraction
            percentile_val = np.percentile(non_special_scores, min_keep * 100)
            threshold = float(percentile_val)

        keep_mask = non_special_scores >= threshold

        already_kept_all = bool(keep_mask.all())

        # ---- Step 2: merge low-score runs that are too short ------------------
        # Very short drop-runs (< min_run_length) would produce many tiny
        # placeholders that *inflate* the output.  Promote them back to keep.
        if not already_kept_all:
            run_starts = []
            in_run = False
            for i in range(n_ns):
                if not keep_mask[i] and not in_run:
                    run_starts.append(i)
                    in_run = True
                elif keep_mask[i] and in_run:
                    run_len = i - (run_starts[-1] if run_starts else i)
                    if run_len < self._min_run_length:
                        for j in range(run_starts[-1], i):
                            keep_mask[j] = True
                    in_run = False
            # Handle trailing run
            if in_run and run_starts:
                run_len = n_ns - run_starts[-1]
                if run_len < self._min_run_length:
                    for j in range(run_starts[-1], n_ns):
                        keep_mask[j] = True

        already_kept_all = bool(keep_mask.all())

        # ---- Step 3: build output with placeholders --------------------------
        if already_kept_all:
            return CompressOutput(
                compressed=content,
                original_token_count=count_tokens(content, model),
                compressed_token_count=count_tokens(content, model),
                strategy="ml_compressor_noop",
            )

        # Tokenise again to get per-token strings for byte estimation
        sub_word_encoding = self._tokenizer.encode(content, add_special_tokens=True)
        all_ids = sub_word_encoding.ids

        # Map back: tokens[0] = cls, then non-special tokens at indices 1..n_ns,
        # then sep.
        kept_text_parts: list[str] = []
        total_chars_compressed = 0

        i = 0
        while i < n_ns:
            if keep_mask[i]:
                # Collect consecutive kept tokens and decode them
                start = i
                while i < n_ns and keep_mask[i]:
                    i += 1
                # Decode this run of tokens (map back to full token indices)
                chunk_ids = all_ids[start + 1 : start + 1 + (i - start)]
                kept_text_parts.append(
                    self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
                )
            else:
                # Low-score run — replace with placeholder
                start = i
                while i < n_ns and not keep_mask[i]:
                    i += 1
                run_len = i - start
                # Estimate chars in this span: decode and measure
                chunk_ids = all_ids[start + 1 : start + 1 + run_len]
                try:
                    span_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
                    char_est = len(span_text)
                except Exception:  # noqa: BLE001 - tokenizer backends use custom exceptions
                    char_est = run_len * 4
                total_chars_compressed += char_est
                kept_text_parts.append(_COMPRESS_MARKER.format(chars=char_est))

        compressed = "".join(kept_text_parts)
        tokens_before = count_tokens(content, model)
        tokens_after = count_tokens(compressed, model)

        return CompressOutput(
            compressed=compressed,
            original_token_count=tokens_before,
            compressed_token_count=tokens_after,
            strategy="ml_compressor",
        )
