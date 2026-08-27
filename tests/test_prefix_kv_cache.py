"""Tests for PrefixKVCache and delta encoding.

This module tests the prefix-only KV cache with delta encoding feature
from "Prefix-Only KV Cache with Delta Encoding" (P4). It verifies:

1. Prefix caching works correctly (cache hit/miss)
2. Delta encoding correctly identifies changes between turns
3. KV cache matching works correctly
4. Integration with the compression pipeline
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from legroom.compressors.kv_cache_fingerprinter import KVCacheFingerprinter
from legroom.runtime.prefix_kv_cache import (
    PrefixDelta,
    PrefixKVCache,
    PrefixKVResult,
)
from legroom.runtime.stable_prefix import StablePrefixCache


class TestPrefixDelta:
    """Tests for PrefixDelta."""

    def test_empty_delta(self):
        """An empty delta should have no changes."""
        delta = PrefixDelta()
        assert delta.is_empty is True
        assert delta.total_changes == 0
        assert delta.token_count == 0
        assert delta.insertions == []
        assert delta.deletions == []
        assert delta.replacements == []
        assert delta.unchanged == []

    def test_insertion_only_delta(self):
        """A delta with only insertions should have total_changes=1."""
        delta = PrefixDelta(
            insertions=[{"role": "user", "content": "new message"}],
            deletions=[],
            replacements=[],
            unchanged=[],
        )
        assert delta.is_empty is False
        assert delta.total_changes == 1

    def test_deletion_only_delta(self):
        """A delta with only deletions should have total_changes=1."""
        delta = PrefixDelta(
            insertions=[],
            deletions=[{"role": "user", "content": "old message"}],
            replacements=[],
            unchanged=[],
        )
        assert delta.total_changes == 1

    def test_replacement_delta(self):
        """A delta with only replacements should have total_changes=1."""
        delta = PrefixDelta(
            insertions=[],
            deletions=[],
            replacements=[
                (
                    {"role": "user", "content": "old"},
                    {"role": "user", "content": "new"},
                )
            ],
            unchanged=[],
        )
        assert delta.total_changes == 1

    def test_mixed_delta(self):
        """A mixed delta should count all changes."""
        delta = PrefixDelta(
            insertions=[{"role": "user", "content": "added"}],
            deletions=[{"role": "user", "content": "removed"}],
            replacements=[
                (
                    {"role": "user", "content": "old"},
                    {"role": "user", "content": "new"},
                )
            ],
            unchanged=[{"role": "user", "content": "same"}],
        )
        assert delta.total_changes == 3
        assert len(delta.unchanged) == 1


class TestKVCacheFingerprinter:
    """Tests for KVCacheFingerprinter."""

    def test_identical_fingerprints(self):
        """Identical text should produce identical fingerprints."""
        fp = KVCacheFingerprinter()
        fp1 = fp.fingerprint("hello world")
        fp2 = fp.fingerprint("hello world")
        assert fp1.fingerprint == fp2.fingerprint
        assert fp1.fingerprint != ""

    def test_different_fingerprints(self):
        """Different text should produce different fingerprints."""
        fp = KVCacheFingerprinter()
        fp1 = fp.fingerprint("hello world")
        fp2 = fp.fingerprint("goodbye world")
        assert fp1.fingerprint != fp2.fingerprint

    def test_similarity_identical(self):
        """Identical fingerprints should have similarity=1.0."""
        fp = KVCacheFingerprinter()
        fp1 = fp.fingerprint("hello world")
        fp2 = fp.fingerprint("hello world")
        assert fp.similarity(fp1, fp2) == pytest.approx(1.0)

    def test_similarity_different(self):
        """Different fingerprints should have similarity < 1.0."""
        fp = KVCacheFingerprinter()
        fp1 = fp.fingerprint("hello world")
        fp2 = fp.fingerprint("goodbye world")
        similarity = fp.similarity(fp1, fp2)
        assert similarity < 1.0
        assert similarity >= 0.0

    def test_is_similar(self):
        """is_similar should return True for similar fingerprints."""
        fp = KVCacheFingerprinter()
        fp1 = fp.fingerprint("hello world")
        fp2 = fp.fingerprint("hello world")
        assert fp.is_similar(fp1, fp2) is True

    def test_is_not_similar(self):
        """is_similar should return False for different fingerprints."""
        fp = KVCacheFingerprinter()
        fp1 = fp.fingerprint("hello world")
        fp2 = fp.fingerprint("completely different text here")
        assert fp.is_similar(fp1, fp2, threshold=0.9) is False

    def test_fingerprint_from_tokens(self):
        """fingerprint_tokens should work with token lists."""
        fp = KVCacheFingerprinter()
        fp1 = fp.fingerprint_tokens([1, 2, 3])
        fp2 = fp.fingerprint_tokens([1, 2, 3])
        fp3 = fp.fingerprint_tokens([4, 5, 6])
        assert fp1.fingerprint == fp2.fingerprint
        assert fp1.fingerprint != fp3.fingerprint


class TestPrefixKVCache:
    """Tests for PrefixKVCache."""

    def test_get_delta_no_previous_tail(self):
        """get_delta should return None when no previous tail is available."""
        cache = PrefixKVCache()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = cache.get_delta(messages, "gpt-4o")
        assert result is None
        # Previous tail should now be set
        assert len(cache._previous_tail) == 1

    def test_get_delta_with_previous_tail(self):
        """get_delta should compute delta when previous tail is available."""
        cache = PrefixKVCache()
        # First call sets the previous tail
        messages1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result1 = cache.get_delta(messages1, "gpt-4o")
        assert result1 is None

        # Second call with same tail
        messages2 = list(messages1)
        result2 = cache.get_delta(messages2, "gpt-4o")
        assert result2 is not None
        assert result2.delta.is_empty is True
        assert result2.kv_cache_match is True

    def test_get_delta_with_insertion(self):
        """get_delta should detect insertions in the tail."""
        cache = PrefixKVCache()
        # First call
        messages1 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        cache.get_delta(messages1, "gpt-4o")

        # Second call with insertion
        messages2 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = cache.get_delta(messages2, "gpt-4o")
        assert result is not None
        assert result.delta.total_changes > 0

    def test_prefix_key_consistency(self):
        """get_prefix_key should return consistent keys for same messages."""
        cache = PrefixKVCache()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        key1 = cache.get_prefix_key(messages, "gpt-4o")
        key2 = cache.get_prefix_key(messages, "gpt-4o")
        assert key1 == key2
        assert key1.startswith("spc:")

    def test_prefix_key_different_model(self):
        """get_prefix_key should return different keys for different models."""
        cache = PrefixKVCache()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        key1 = cache.get_prefix_key(messages, "gpt-4o")
        key2 = cache.get_prefix_key(messages, "gpt-3.5-turbo")
        assert key1 != key2

    def test_get_or_compute_prefix(self):
        """get_or_compute_prefix should cache and return prefix messages."""
        cache = PrefixKVCache()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        # First call (cache miss)
        compress_fn = MagicMock(return_value=messages[:1])
        prefix1, tokens1 = cache.get_or_compute_prefix(
            messages, "gpt-4o", compress_fn
        )
        assert prefix1 == messages[:1]
        assert tokens1 > 0

        # Second call (cache hit)
        compress_fn.reset_mock()
        prefix2, tokens2 = cache.get_or_compute_prefix(
            messages, "gpt-4o", compress_fn
        )
        assert prefix2 == prefix1
        assert compress_fn.call_count == 0  # Should not re-compress


class TestPrefixKVCacheIntegration:
    """Integration tests for PrefixKVCache with StablePrefixCache."""

    def test_full_get_or_compute(self):
        """Test full get_or_compute flow with prefix cache."""
        stable_cache = StablePrefixCache()
        cache = PrefixKVCache(stable_prefix_cache=stable_cache)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        # Mock compress_fn
        compress_fn = MagicMock(return_value=[{"role": "system", "content": "You are helpful."}])

        # First call (cache miss)
        result = cache.get_or_compute(messages, "gpt-4o", compress_fn)
        assert result is not None
        assert result.kv_cache_match is False
        assert compress_fn.call_count == 1

        # Second call (cache hit)
        compress_fn.reset_mock()
        result = cache.get_or_compute(messages, "gpt-4o", compress_fn)
        assert result is not None
        assert result.kv_cache_match is True
        assert compress_fn.call_count == 0  # Should not re-compress

    def test_metadata_integration(self):
        """Test that metadata includes delta information."""
        stable_cache = StablePrefixCache()
        cache = PrefixKVCache(stable_prefix_cache=stable_cache)

        # First call to set previous tail
        messages1 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        cache.get_delta(messages1, "gpt-4o")

        # Second call with delta
        messages2 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = cache.get_delta(messages2, "gpt-4o")
        assert result is not None
        assert result.delta.total_changes > 0

    def test_prefix_token_count(self):
        """Test that prefix token count is computed correctly."""
        cache = PrefixKVCache()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = cache.get_delta(messages, "gpt-4o")
        # First call returns None, so we test on second call
        assert result is None

        result2 = cache.get_delta(messages, "gpt-4o")
        assert result2 is not None
        assert result2.prefix_tokens > 0


class TestDeltaEncoding:
    """Tests for delta encoding between turns."""

    def test_no_change_delta(self):
        """When tail hasn't changed, delta should be empty."""
        cache = PrefixKVCache()
        previous = [
            {"role": "user", "content": "Hello"},
        ]
        current = [
            {"role": "user", "content": "Hello"},
        ]
        delta = cache._compute_delta(previous, current)
        assert delta.is_empty is True

    def test_new_message_delta(self):
        """When a new message is added, delta should have insertion."""
        cache = PrefixKVCache()
        previous = [
            {"role": "user", "content": "Hello"},
        ]
        current = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        delta = cache._compute_delta(previous, current)
        assert delta.total_changes > 0
        assert len(delta.insertions) > 0

    def test_replacement_delta(self):
        """When a message content changes, delta should have replacement."""
        cache = PrefixKVCache()
        previous = [
            {"role": "user", "content": "Hello"},
        ]
        current = [
            {"role": "user", "content": "Goodbye"},
        ]
        delta = cache._compute_delta(previous, current)
        assert delta.total_changes > 0
        assert len(delta.replacements) > 0

    def test_deleted_message_delta(self):
        """When a message is removed, delta should have deletion."""
        cache = PrefixKVCache()
        previous = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        current = [
            {"role": "user", "content": "Hello"},
        ]
        delta = cache._compute_delta(previous, current)
        assert delta.total_changes > 0
        assert len(delta.deletions) > 0

    def test_tool_calls_change_detection(self):
        """Delta should detect changes in tool calls."""
        cache = PrefixKVCache()
        previous = [
            {"role": "assistant", "content": "Let me help."},
        ]
        current = [
            {"role": "assistant", "content": "Let me help.", "tool_calls": [{"id": "1", "function": {"name": "test"}}]},
        ]
        delta = cache._compute_delta(previous, current)
        assert delta.total_changes > 0


class TestKVCacheFingerprinting:
    """Tests for KV cache fingerprinting."""

    def test_fingerprint_deterministic(self):
        """Fingerprints should be deterministic."""
        fp = KVCacheFingerprinter()
        for _ in range(10):
            result = fp.fingerprint("test message")
            assert result.fingerprint == fp.fingerprint("test message").fingerprint

    def test_fingerprint_stable_for_short_messages(self):
        """Fingerprints should be stable for short messages."""
        fp = KVCacheFingerprinter()
        fp1 = fp.fingerprint("short")
        fp2 = fp.fingerprint("short")
        assert fp1.fingerprint == fp2.fingerprint

    def test_fingerprint_stable_for_long_messages(self):
        """Fingerprints should be stable for long messages (uses first 64 tokens)."""
        fp = KVCacheFingerprinter()
        long_text = "word " * 100
        fp1 = fp.fingerprint(long_text)
        fp2 = fp.fingerprint(long_text)
        assert fp1.fingerprint == fp2.fingerprint

    def test_find_similar(self):
        """find_similar should return the most similar fingerprint."""
        fp = KVCacheFingerprinter()
        target = fp.fingerprint("hello world")
        candidates = [
            fp.fingerprint("hello world"),  # Exact match
            fp.fingerprint("goodbye world"),  # Partial match
            fp.fingerprint("completely different"),  # No match
        ]
        result = fp.find_similar(target, candidates, threshold=0.5)
        assert result is not None
        assert result.fingerprint == target.fingerprint

    def test_find_similar_no_match(self):
        """find_similar should return None when no match above threshold."""
        fp = KVCacheFingerprinter()
        target = fp.fingerprint("hello world")
        candidates = [
            fp.fingerprint("completely different text"),
            fp.fingerprint("totally unrelated message"),
        ]
        result = fp.find_similar(target, candidates, threshold=0.9)
        assert result is None
