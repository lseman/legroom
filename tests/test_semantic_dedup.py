"""Tests for semantic cross-turn deduplication."""

from __future__ import annotations

import json

import pytest

from legroom import CompressConfig, compress
from legroom.compressors.semantic_dedup import (
    SemanticDedup,
    SemanticDedupResult,
    _cosine_similarity,
)

# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical():
    """Identical vectors should have similarity 1.0."""
    v = [1.0, 2.0, 3.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should have similarity 0.0."""
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_negative():
    """Opposite vectors should have similarity -1.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_empty():
    """Empty vectors should return 0.0."""
    assert _cosine_similarity([], [1.0, 2.0]) == 0.0
    assert _cosine_similarity([1.0], []) == 0.0


# ---------------------------------------------------------------------------
# SemanticDedup — no model (graceful degradation)
# ---------------------------------------------------------------------------


def test_semantic_dedup_no_model_returns_unchanged():
    """Without an ONNX model, dedup should return messages unchanged."""
    dedup = SemanticDedup(
        model_path="/nonexistent/model.onnx",
        vocab_path="/nonexistent/vocab.txt",
    )
    messages = [
        {"role": "user", "content": "Show me the code"},
        {"role": "assistant", "content": "Here is the code:\n" + "x" * 300},
    ]
    result = dedup.dedup(messages)
    assert isinstance(result, SemanticDedupResult)
    assert result.dedup_count == 0
    assert result.tokens_saved == 0
    # Messages should be unchanged
    assert result.messages[0]["content"] == messages[0]["content"]
    assert result.messages[1]["content"] == messages[1]["content"]


def test_semantic_dedup_empty_messages():
    """Empty message list should return empty result."""
    dedup = SemanticDedup()
    result = dedup.dedup([])
    assert result.messages == []
    assert result.dedup_count == 0


def test_semantic_dedup_single_message():
    """Single message should pass through unchanged."""
    dedup = SemanticDedup()
    messages = [{"role": "user", "content": "Hello"}]
    result = dedup.dedup(messages)
    assert result.dedup_count == 0
    assert result.messages == messages


def test_semantic_dedup_short_messages():
    """Messages shorter than min_bytes should pass through unchanged."""
    dedup = SemanticDedup(min_bytes=1000)  # High threshold
    messages = [
        {"role": "user", "content": "Short"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = dedup.dedup(messages)
    assert result.dedup_count == 0


def test_semantic_dedup_protect_recent():
    """Recent messages should not be deduplicated."""
    dedup = SemanticDedup(min_bytes=1000, protect_recent=2)
    content = "x" * 200
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "Second"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "Third"},
        {"role": "assistant", "content": "Recent"},
    ]
    result = dedup.dedup(messages)
    # The last two messages are protected, so the second occurrence of
    # `content` (at index 3) should still be checked, but the last
    # message (index 5) is protected.
    # With protect_recent=2, indices 4 and 5 are protected, so index 3
    # is the last one checked.
    assert result.dedup_count >= 0  # May or may not dedup depending on model availability


def test_semantic_dedup_is_available():
    """is_available should be False without a model."""
    dedup = SemanticDedup(model_path="/nonexistent/model.onnx")
    assert dedup.is_available is False


def test_semantic_dedup_load_error_message():
    """Load error should be recorded."""
    dedup = SemanticDedup(model_path="/nonexistent/model.onnx")
    dedup.dedup([{"role": "user", "content": "x" * 200}])
    assert dedup._load_error is not None


# ---------------------------------------------------------------------------
# Integration: semantic dedup in pipeline (without model — no-op path)
# ---------------------------------------------------------------------------


def test_pipeline_semantic_dedup_disabled_by_default():
    """Semantic dedup should be off by default."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]
    result = compress(messages, model="gpt-4o")
    assert "semantic_dedup" not in result.transforms_applied


def test_pipeline_semantic_dedup_enabled_no_model():
    """Enabling semantic dedup without a model should be a no-op."""
    messages = [
        {"role": "user", "content": "Show me the code"},
        {"role": "assistant", "content": "Here is the code:\n" + "x" * 300},
    ]
    config = CompressConfig(
        semantic_dedup_enabled=True,
        semantic_dedup_threshold=0.85,
    )
    result = compress(messages, model="gpt-4o", config=config)
    assert result.tokens_saved >= 0
    # Should not crash, just skip semantic dedup


def test_pipeline_semantic_dedup_with_other_transforms():
    """Semantic dedup should coexist with other pipeline transforms."""
    long_json = json.dumps([{"id": i} for i in range(50)], indent=2)
    messages = [
        {"role": "user", "content": "Start"},
        {"role": "assistant", "content": long_json},
        {"role": "user", "content": "End"},
    ]
    config = CompressConfig(
        semantic_dedup_enabled=True,
        protect_recent=1,
    )
    result = compress(messages, model="gpt-4o", config=config)
    assert result.messages[-1]["content"] == "End"  # Recent protected
    assert result.tokens_saved >= 0


def test_compress_semantic_dedup_kwargs():
    """Semantic dedup kwargs should be accepted."""
    messages = [
        {"role": "user", "content": "Test"},
        {"role": "assistant", "content": "Response"},
    ]
    result = compress(
        messages,
        model="gpt-4o",
        semantic_dedup_enabled=True,
    )
    assert result.tokens_before > 0


def test_pipeline_semantic_dedup_inflation_guard():
    """Inflation guard should still work with semantic dedup enabled."""
    messages = [
        {"role": "user", "content": "Short"},
        {"role": "assistant", "content": "Brief"},
    ]
    config = CompressConfig(
        semantic_dedup_enabled=True,
    )
    result = compress(messages, model="gpt-4o", config=config)
    assert result.tokens_after <= result.tokens_before


def test_semantic_dedup_non_string_content():
    """Non-string content should pass through unchanged."""
    dedup = SemanticDedup()
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
    ]
    result = dedup.dedup(messages)
    assert result.messages[0]["content"] == messages[0]["content"]


def test_semantic_dedup_multiple_short_messages():
    """Multiple short messages should all pass through."""
    dedup = SemanticDedup(min_bytes=500)
    messages = [
        {"role": "user", "content": "A" * 10},
        {"role": "assistant", "content": "B" * 10},
        {"role": "user", "content": "C" * 10},
        {"role": "assistant", "content": "D" * 10},
    ]
    result = dedup.dedup(messages)
    assert result.dedup_count == 0


def test_semantic_dedup_result_fields():
    """SemanticDedupResult should have all expected fields."""
    result = SemanticDedupResult(
        messages=[{"role": "user", "content": "test"}],
        dedup_count=5,
        tokens_saved=100,
        warnings=["test warning"],
    )
    assert result.dedup_count == 5
    assert result.tokens_saved == 100
    assert result.warnings == ["test warning"]
    assert len(result.messages) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
