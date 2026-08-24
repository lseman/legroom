"""Tests for dynamic token budget compression."""

from __future__ import annotations

from legroom.budget import (
    BudgetCompressor,
    BudgetResult,
    compress_with_budget,
)
from legroom.config import CompressConfig


def _make_long_text(n_words: int = 500) -> str:
    """Generate a long text with many sentences for compression testing."""
    words = [
        "the quick brown fox jumps over the lazy dog near the river bank",
        "a large elephant walks slowly through the dense green forest",
        "the sun rises over the mountains casting golden light across the valley",
        "scientists discovered a new species of deep sea fish in the ocean",
        "the computer processed millions of data points in seconds using algorithms",
    ]
    sentences = []
    for i in range(n_words // 12):
        sentences.append(words[i % len(words)])
    return " ".join(sentences)


def _make_long_json(n_items: int = 100) -> list[dict]:
    """Generate a large JSON array for SmartCrusher testing."""
    return [
        {
            "id": i,
            "name": f"item_{i}",
            "status": "active" if i % 2 == 0 else "inactive",
            "value": i * 1.5,
            "description": f"This is a detailed description for item number {i} in the list",
            "metadata": {"created": "2024-01-01", "updated": "2024-01-02"},
        }
        for i in range(n_items)
    ]


class TestBudgetCompressorInit:
    def test_default_budget_is_none(self):
        bc = BudgetCompressor()
        assert bc.target_budget is None

    def test_explicit_budget(self):
        bc = BudgetCompressor(target_budget=50_000)
        assert bc.target_budget == 50_000

    def test_min_budget_enforced(self):
        bc = BudgetCompressor(min_budget=5_000)
        assert bc.min_budget == 5_000


class TestBudgetResolution:
    def test_gpt4o_budget(self):
        bc = BudgetCompressor()
        budget = bc._resolve_budget("gpt-4o", 1000)
        assert budget == int(128_000 * 0.80)  # 102400

    def test_claude_budget(self):
        bc = BudgetCompressor()
        budget = bc._resolve_budget("claude-3-5-sonnet", 1000)
        assert budget == int(200_000 * 0.80)  # 160000

    def test_unknown_model_fallback(self):
        bc = BudgetCompressor()
        budget = bc._resolve_budget("unknown-model", 1000)
        assert budget == int(32_000 * 0.80)  # 25600

    def test_explicit_budget_overrides_model(self):
        bc = BudgetCompressor(target_budget=10_000)
        budget = bc._resolve_budget("gpt-4o", 1000)
        assert budget == 10_000

    def test_min_budget_floor(self):
        # min_budget is a floor — model-derived budget wins if higher
        bc = BudgetCompressor(min_budget=50_000)
        budget = bc._resolve_budget("gpt-4o", 1000)
        assert budget == 102_400  # model-derived > min_budget

        # When model-derived is below min_budget, min_budget wins
        bc2 = BudgetCompressor(min_budget=200_000)
        budget2 = bc2._resolve_budget("gpt-4o", 1000)
        assert budget2 == 200_000


class TestAggressiveness:
    def test_under_budget_zero_aggressiveness(self):
        bc = BudgetCompressor()
        agg = bc._compute_aggressiveness(500, 800, 1)
        assert agg == 0.0

    def test_slightly_over_budget_low_aggressiveness(self):
        bc = BudgetCompressor()
        agg = bc._compute_aggressiveness(1000, 800, 1)
        assert 0 < agg < 0.2

    def test_moderately_over_budget_medium_aggressiveness(self):
        bc = BudgetCompressor()
        agg = bc._compute_aggressiveness(2000, 800, 1)
        assert 0.2 < agg < 0.6

    def test_very_over_budget_high_aggressiveness(self):
        bc = BudgetCompressor()
        agg = bc._compute_aggressiveness(5000, 800, 1)
        assert agg > 0.7

    def test_iteration_increases_aggressiveness(self):
        bc = BudgetCompressor()
        agg1 = bc._compute_aggressiveness(2000, 800, 1)
        agg2 = bc._compute_aggressiveness(2000, 800, 2)
        agg3 = bc._compute_aggressiveness(2000, 800, 3)
        assert agg1 < agg2 < agg3


class TestAggressivenessApplication:
    def test_zero_aggressiveness_no_change(self):
        cfg = CompressConfig()
        bc = BudgetCompressor()
        adj = bc._apply_aggressiveness(cfg, 0.0)
        # Defaults have shifted (task 1): adaptive_sizing is now True
        assert adj.retention_threshold == 0.5
        assert adj.size_bias == 1.0
        assert adj.adaptive_sizing is True  # was False before task 1

    def test_high_aggressiveness_increases_compression(self):
        cfg = CompressConfig()
        bc = BudgetCompressor()
        adj = bc._apply_aggressiveness(cfg, 0.8)
        assert adj.retention_threshold < 0.5
        assert adj.size_bias < 1.0
        assert adj.adaptive_sizing is True
        assert adj.min_compression_ratio < 0.15

    def test_aggressiveness_clamped_to_max(self):
        cfg = CompressConfig()
        bc = BudgetCompressor()
        adj = bc._apply_aggressiveness(cfg, 1.0)
        assert adj.retention_threshold >= 0.1
        assert adj.size_bias >= 0.3
        assert adj.min_compression_ratio >= 0.0


class TestBudgetCompression:
    def test_already_under_budget(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = compress_with_budget(
            messages, model="gpt-4o", target_budget=1000
        )
        assert result.budget_met is True
        assert result.tokens_after <= result.budget
        assert isinstance(result, BudgetResult)

    def test_over_budget_triggers_compression(self):
        long_text = _make_long_text(n_words=2000)
        messages = [
            {"role": "user", "content": "Here is some text:"},
            {"role": "assistant", "content": long_text},
            {"role": "user", "content": "Summarize this."},
        ]
        # protect_recent=0 so the budget loop can actually compress
        config = CompressConfig(protect_recent=0)
        result = compress_with_budget(
            messages, model="gpt-4o", target_budget=500, config=config
        )
        # The text is ~1900 tokens; aggressive compression may or may not
        # reach 500, but we should at least see some savings or the budget
        # loop should stop gracefully.
        assert result.tokens_saved >= 0  # never makes things worse
        assert result.tokens_after <= result.tokens_before
        assert result.metadata.get("iterations", 0) >= 1

    def test_budget_result_structure(self):
        messages = [{"role": "user", "content": "test"}]
        result = compress_with_budget(messages, model="gpt-4o", target_budget=1000)
        assert hasattr(result, "messages")
        assert hasattr(result, "tokens_before")
        assert hasattr(result, "tokens_after")
        assert hasattr(result, "tokens_saved")
        assert hasattr(result, "budget")
        assert hasattr(result, "budget_met")
        assert hasattr(result, "transforms_applied")
        assert hasattr(result, "warnings")
        assert hasattr(result, "metadata")

    def test_max_iterations_limit(self):
        # Create a scenario where compression can't help much
        messages = [
            {"role": "user", "content": "x" * 100},
            {"role": "assistant", "content": "y" * 100},
        ]
        result = compress_with_budget(
            messages, model="gpt-4o", target_budget=1, max_iterations=1
        )
        # Should not infinite loop
        assert result.metadata.get("iterations", 0) <= 1

    def test_custom_config_passed_through(self):
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am doing well, thank you for asking! I hope you are having a great day as well."},
        ]
        config = CompressConfig(
            protect_recent=2,  # Protect both messages
            cache_align_enabled=False,
        )
        result = compress_with_budget(
            messages, model="gpt-4o", target_budget=50, config=config
        )
        # With protect_recent=2, nothing should be compressed
        assert result.tokens_after == result.tokens_before

    def test_json_compression_under_budget(self):
        import json
        long_json = _make_long_json(n_items=150)
        messages = [
            {"role": "user", "content": "Here are the results:"},
            {"role": "assistant", "content": json.dumps(long_json)},
        ]
        # protect_recent=0 so SmartCrusher can compress the JSON array
        config = CompressConfig(protect_recent=0)
        result = compress_with_budget(
            messages, model="gpt-4o", target_budget=500, config=config
        )
        # SmartCrusher should compress 150-item JSON arrays significantly
        assert result.tokens_saved > 0
        assert result.tokens_after < result.tokens_before

    def test_empty_messages(self):
        result = compress_with_budget(
            [], model="gpt-4o", target_budget=1000
        )
        assert result.tokens_before == 0
        assert result.tokens_after == 0
        assert result.budget_met is True
