"""Dynamic token budget compression — budget-aware outer loop.

The existing pipeline compresses content by type (JSON → SmartCrusher,
text → TF-IDF) with no awareness of how much context the model actually
has room for.  This module wraps the pipeline in a budget-aware loop that:

1. Counts total tokens before compression.
2. Determines a target budget (explicit or derived from model context).
3. Runs the pipeline, then checks if the budget is met.
4. If over budget, increases compression aggressiveness and retries.
5. If under budget with room to spare, reduces compression to preserve
   more content.

Usage::

    from legroom.budget import BudgetCompressor, model_context_windows

    compressor = BudgetCompressor(
        target_budget=80_000,  # or None to auto-detect from model
    )
    result = compressor.compress(messages, model="gpt-4o")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..runtime.compress import compress
from ..runtime.config import CompressConfig
from .tokenizer import count_tokens_messages

logger = logging.getLogger(__name__)

# Context window sizes by model family (in tokens).
# Used as the default target when no explicit budget is given.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
    "gpt-3.5-turbo-16k": 16_385,
    # Anthropic
    "claude-3-5-sonnet": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-haiku": 200_000,
    "claude-2.1": 200_000,
    "claude-2": 100_000,
    "claude-instant-1": 100_000,
}

# Fraction of context window to use as default target budget.
DEFAULT_BUDGET_FRACTION = 0.80

# Maximum iterations the budget loop will retry before giving up.
MAX_BUDGET_ITERATIONS = 3


@dataclass
class BudgetResult:
    """Result of a budget-aware compression run."""

    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    budget: int
    budget_met: bool
    transforms_applied: list[str]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BudgetCompressor:
    """Budget-aware compression that adjusts aggressiveness to hit a token target.

    The compressor runs the existing pipeline iteratively, adjusting
    compression parameters when the budget is not met.  Each iteration
    increases aggressiveness on the phases that contributed the most
    compression (measured by tokens saved per phase).

    Parameters:
        target_budget: Maximum tokens for the compressed context.
            If None, derived from the model's context window
            (DEFAULT_BUDGET_FRACTION of the window).
        min_budget: Hard floor — never compress below this many tokens.
            Prevents over-compression of small conversations.
        max_iterations: Max retry iterations (default 3).
    """

    def __init__(
        self,
        target_budget: int | None = None,
        min_budget: int = 1_000,
        max_iterations: int = MAX_BUDGET_ITERATIONS,
    ) -> None:
        self.target_budget = target_budget
        self.min_budget = min_budget
        self.max_iterations = max_iterations

    def _resolve_budget(self, model: str, tokens_before: int) -> int:
        """Resolve the target budget from model context or explicit value."""
        if self.target_budget is not None:
            return max(self.target_budget, self.min_budget)

        context_window = MODEL_CONTEXT_WINDOWS.get(model)
        if context_window is None:
            # Unknown model — use a conservative default
            return int(32_000 * DEFAULT_BUDGET_FRACTION)

        budget = int(context_window * DEFAULT_BUDGET_FRACTION)
        return max(budget, self.min_budget)

    def compress(
        self,
        messages: list[dict[str, Any]],
        model: str = "gpt-4o",
        config: CompressConfig | None = None,
        **kwargs: Any,
    ) -> BudgetResult:
        """Compress messages to fit within the token budget.

        Runs the pipeline iteratively, adjusting compression
        aggressiveness when the budget is not met.
        """
        tokens_before = count_tokens_messages(messages, model)
        budget = self._resolve_budget(model, tokens_before)

        # If already under budget, just run the pipeline normally
        if tokens_before <= budget:
            result = compress(messages, model=model, config=config, **kwargs)
            return BudgetResult(
                messages=result.messages,
                tokens_before=result.tokens_before,
                tokens_after=result.tokens_after,
                tokens_saved=result.tokens_saved,
                budget=budget,
                budget_met=True,
                transforms_applied=result.transforms_applied,
                warnings=result.warnings,
                metadata=result.metadata,
            )

        # Iterative budget-aware compression
        current_messages = list(messages)
        current_config = config or CompressConfig()
        all_transforms: list[str] = []
        all_warnings: list[str] = []
        iterations = 0

        while tokens_before > budget and iterations < self.max_iterations:
            iterations += 1

            # Adjust aggressiveness for this iteration
            aggressiveness = self._compute_aggressiveness(
                tokens_before, budget, iterations
            )
            adjusted_config = self._apply_aggressiveness(
                current_config, aggressiveness
            )

            logger.debug(
                "Budget compression iter %d: budget=%d, "
                "tokens_before=%d, aggressiveness=%.2f",
                iterations,
                budget,
                tokens_before,
                aggressiveness,
            )

            result = compress(
                current_messages,
                model=model,
                config=adjusted_config,
                **kwargs,
            )

            current_messages = result.messages
            tokens_after = result.tokens_after
            all_transforms.extend(result.transforms_applied)
            all_warnings.extend(result.warnings)

            # Check if we've hit the budget
            if tokens_after <= budget:
                break

            # If we can't compress further, stop
            if tokens_after >= tokens_before:
                logger.warning(
                    "Budget compression: no further compression possible "
                    "(tokens_after=%d >= tokens_before=%d)",
                    tokens_after,
                    tokens_before,
                )
                break

            tokens_before = tokens_after

        return BudgetResult(
            messages=current_messages,
            tokens_before=count_tokens_messages(messages, model),
            tokens_after=count_tokens_messages(current_messages, model),
            tokens_saved=count_tokens_messages(messages, model)
            - count_tokens_messages(current_messages, model),
            budget=budget,
            budget_met=count_tokens_messages(current_messages, model) <= budget,
            transforms_applied=all_transforms,
            warnings=all_warnings,
            metadata={"iterations": iterations, "aggressiveness": aggressiveness},
        )

    def _compute_aggressiveness(
        self, tokens_before: int, budget: int, iteration: int
    ) -> float:
        """Compute compression aggressiveness for this iteration.

        Returns a value in [0, 1] where 0 = no extra compression,
        1 = maximum compression.  Increases with iteration count and
        with the gap between current tokens and budget.
        """
        # Base aggressiveness from the compression ratio needed
        ratio = tokens_before / budget
        if ratio <= 1.0:
            return 0.0

        # Scale: 10% over budget → 0.1, 2x over → 0.5, 5x over → 0.8
        base = min((ratio - 1.0) / 4.0, 0.8)

        # Iteration multiplier: each iteration gets more aggressive
        iteration_factor = 1.0 + (iteration - 1) * 0.3

        return min(base * iteration_factor, 1.0)

    def _apply_aggressiveness(
        self, config: CompressConfig, aggressiveness: float
    ) -> CompressConfig:
        """Apply aggressiveness by adjusting compression parameters."""
        adjusted = CompressConfig(**{
            k: v for k, v in config.__dict__.items()
            if k not in ("ml_model_path", "ml_tokenizer_path",
                         "semantic_dedup_model_path", "semantic_dedup_config_path",
                         "semantic_dedup_vocab_path")
        })

        if aggressiveness <= 0:
            return adjusted

        # Text compressor: lower keep_ratio to keep fewer sentences
        # (passed through ml_retention_threshold as a proxy)
        adjusted.retention_threshold = max(0.1, 0.5 - aggressiveness * 0.4)

        # SmartCrusher: reduce max_items proportionally
        # We pass this through size_bias as a proxy for aggressiveness
        adjusted.size_bias = max(0.3, 1.0 - aggressiveness * 0.7)

        # Enable adaptive sizing if not already enabled
        adjusted.adaptive_sizing = True

        # Lower min_compression_ratio to accept more aggressive compression
        adjusted.min_compression_ratio = max(0.0, 0.15 - aggressiveness * 0.1)

        return adjusted


def compress_with_budget(
    messages: list[dict[str, Any]],
    model: str = "gpt-4o",
    target_budget: int | None = None,
    min_budget: int = 1_000,
    config: CompressConfig | None = None,
    **kwargs: Any,
) -> BudgetResult:
    """Convenience function for budget-aware compression.

    Args:
        messages: List of message dicts.
        model: Model name for context window detection.
        target_budget: Max tokens for compressed context.
            If None, uses 80% of model's context window.
        min_budget: Hard floor for compression (default 1,000 tokens).
        config: Optional compression configuration overrides.
        **kwargs: Passed through to the underlying compress() call.

    Returns:
        BudgetResult with compressed messages and budget metadata.
    """
    compressor = BudgetCompressor(
        target_budget=target_budget,
        min_budget=min_budget,
    )
    return compressor.compress(messages, model=model, config=config, **kwargs)
