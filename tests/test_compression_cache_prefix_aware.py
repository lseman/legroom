"""Tests for prefix-aware compression cache key (llama.cpp optimization)."""

from __future__ import annotations

import pytest

from legroom.proxy.compression_cache import (
    CachedCompression,
    CompressionResultCache,
)
from legroom.runtime.stable_prefix import StablePrefixCache, _prefix_key


# ---------------------------------------------------------------------------
# CompressionResultCache — key() baseline
# ---------------------------------------------------------------------------


def test_key_consistency():
    """The standard key() method should produce consistent results."""
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]
    key1 = CompressionResultCache.key(
        protocol="openai_chat",
        model="gpt-4o",
        mode="token",
        messages=messages,
        policy="v2:ccr=True",
    )
    key2 = CompressionResultCache.key(
        protocol="openai_chat",
        model="gpt-4o",
        mode="token",
        messages=messages,
        policy="v2:ccr=True",
    )
    assert key1 == key2
    assert len(key1) == 64  # SHA-256 hex digest


def test_key_different_messages_different_result():
    """Different messages should produce different keys."""
    messages_a = [{"role": "user", "content": "hello"}]
    messages_b = [{"role": "user", "content": "world"}]
    key_a = CompressionResultCache.key(
        protocol="openai_chat", model="gpt-4o", mode="token",
        messages=messages_a, policy="v2:ccr=True",
    )
    key_b = CompressionResultCache.key(
        protocol="openai_chat", model="gpt-4o", mode="token",
        messages=messages_b, policy="v2:ccr=True",
    )
    assert key_a != key_b


# ---------------------------------------------------------------------------
# CompressionResultCache — tail_key() prefix-aware key
# ---------------------------------------------------------------------------


def test_tail_key_consistency():
    """The tail_key() should produce consistent results."""
    tail = [{"role": "user", "content": "list files"}, {"role": "assistant", "content": "done"}]
    key1 = CompressionResultCache.tail_key(model="llama-3", tail_messages=tail)
    key2 = CompressionResultCache.tail_key(model="llama-3", tail_messages=tail)
    assert key1 == key2


def test_tail_key_same_tail_different_prefixes():
    """Two requests with different prefixes but the same tail get the same key.

    This is the core optimization: repeated conversation patterns across
    different turns share a cache entry even though the full message lists
    (including system prompts / tool definitions) are completely different.
    """
    tail = [{"role": "user", "content": "list files"}, {"role": "assistant", "content": "done"}]
    key = CompressionResultCache.tail_key(model="gpt-4o", tail_messages=tail)

    # Two entirely different full message lists but same tail
    prefix_a = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "done"},
    ]
    prefix_b = [
        {"role": "system", "content": "You are a data scientist."},
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "done"},
    ]
    # Both tails are the same → same key
    tail_a = [msg for msg in prefix_a if msg["role"] not in ("system", "tool", "function")]
    tail_b = [msg for msg in prefix_b if msg["role"] not in ("system", "tool", "function")]
    key_a = CompressionResultCache.tail_key(model="gpt-4o", tail_messages=tail_a)
    key_b = CompressionResultCache.tail_key(model="gpt-4o", tail_messages=tail_b)
    assert key_a == key_b  # Same tail → same key


def test_tail_key_different_model_different_key():
    """Different models should produce different tail keys."""
    tail = [{"role": "user", "content": "hello"}]
    key_a = CompressionResultCache.tail_key(model="gpt-4o", tail_messages=tail)
    key_b = CompressionResultCache.tail_key(model="claude-3", tail_messages=tail)
    assert key_a != key_b


def test_tail_key_different_tail_different_key():
    """Different tails should produce different keys."""
    tail_a = [{"role": "user", "content": "hello"}]
    tail_b = [{"role": "user", "content": "world"}]
    key_a = CompressionResultCache.tail_key(model="gpt-4o", tail_messages=tail_a)
    key_b = CompressionResultCache.tail_key(model="gpt-4o", tail_messages=tail_b)
    assert key_a != key_b


# ---------------------------------------------------------------------------
# CompressionResultCache — get_or_compute(prefix-aware)
# ---------------------------------------------------------------------------


def test_get_or_compute_cache_hit():
    """get_or_compute should return cached result on hit."""
    cache = CompressionResultCache(maxsize=10)
    tail = [{"role": "user", "content": "list files"}]

    result1 = cache.get_or_compute(
        model="gpt-4o",
        tail_messages=tail,
        compute=lambda: CachedCompression(
            messages=[{"role": "assistant", "content": "here are the files"}],
            tokens_before=100,
            tokens_after=50,
            transforms=["smart_crusher"],
        ),
    )
    assert result1.tokens_after == 50
    assert cache.hits == 0  # First call is a put

    # Same inputs → cache hit
    result2 = cache.get_or_compute(
        model="gpt-4o",
        tail_messages=tail,
        compute=lambda: CachedCompression(
            messages=[{"role": "assistant", "content": "DIFFERENT"}],
            tokens_before=200,
            tokens_after=100,
            transforms=["other"],
        ),
    )
    assert result2.tokens_after == 50  # Same cached result
    assert cache.hits == 1  # Second call is a hit


def test_get_or_compute_cache_miss():
    """get_or_compute should call compute on cache miss."""
    cache = CompressionResultCache(maxsize=10)
    tail = [{"role": "user", "content": "hello"}]
    compute_count = [0]

    def counting_compute():
        compute_count[0] += 1
        return CachedCompression(
            messages=[{"role": "assistant", "content": "hi"}],
            tokens_before=50,
            tokens_after=30,
            transforms=["text_compressor"],
        )

    result1 = cache.get_or_compute(
        model="gpt-4o",
        tail_messages=tail,
        compute=counting_compute,
    )
    assert result1.tokens_after == 30
    assert compute_count[0] == 1  # compute() called once

    # Same inputs → cache hit, compute not called
    result2 = cache.get_or_compute(
        model="gpt-4o",
        tail_messages=tail,
        compute=counting_compute,
    )
    assert compute_count[0] == 1  # Still 1 — not called again


def test_get_or_compute_same_tail_same_key():
    """Same tail should produce the same cache key regardless of prefix.

    This is the core llama.cpp optimization: the prefix is handled by
    StablePrefixCache separately, and the compression cache only needs
    the tail to determine a cache hit.
    """
    cache = CompressionResultCache(maxsize=10)
    tail = [{"role": "user", "content": "list files"}]

    result_a = cache.get_or_compute(
        model="gpt-4o",
        tail_messages=tail,
        compute=lambda: CachedCompression(
            messages=[{"content": "A result"}],
            tokens_before=100,
            tokens_after=50,
            transforms=["a_transform"],
        ),
    )
    assert result_a.transforms == ["a_transform"]

    # Same tail, different "prefix" — should be a cache hit
    result_b = cache.get_or_compute(
        model="gpt-4o",
        tail_messages=tail,
        compute=lambda: CachedCompression(
            messages=[{"content": "B result"}],
            tokens_before=100,
            tokens_after=40,
            transforms=["b_transform"],
        ),
    )
    assert result_b.transforms == ["a_transform"]  # Same cached result


# ---------------------------------------------------------------------------
# StablePrefixCache — key_for_messages
# ---------------------------------------------------------------------------


def test_stable_prefix_key_for_messages():
    """key_for_messages should produce the same key as _prefix_key directly."""
    cache = StablePrefixCache()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    model = "gpt-4o"

    key_public = cache.key_for_messages(messages, model)
    key_direct = _prefix_key(messages, model)
    assert key_public == key_direct


def test_stable_prefix_key_includes_model():
    """Different models should produce different keys for the same messages."""
    cache = StablePrefixCache()
    messages = [{"role": "system", "content": "test"}]
    key_a = cache.key_for_messages(messages, "gpt-4o")
    key_b = cache.key_for_messages(messages, "claude-3")
    assert key_a != key_b


# ---------------------------------------------------------------------------
# Integration: prefix-aware cache flow (simulates proxy behavior)
# ---------------------------------------------------------------------------


def test_prefix_aware_cache_flow_after_prefix_cold_start():
    """After the StablePrefixCache is populated (post cold start),
    repeated tails get compression cache hits via tail keys.

    This is the real-world llama.cpp scenario:
    - First request with a new prefix: StablePrefixCache miss, full-message key
    - After prefix is cached: all subsequent requests with same prefix
      use tail keys, so repeated conversation patterns get cache hits.
    """
    cache = CompressionResultCache(maxsize=10)
    sp_cache = StablePrefixCache()

    # Shared system prompt (simulates a long-running session)
    prefix_a = [{"role": "system", "content": "You are a coding assistant."}]

    # Simulate that StablePrefixCache is already populated
    # (this happens after a few turns in a real session)
    sp_key = sp_cache.key_for_messages(prefix_a, "llama-3")
    sp_cache.put(sp_key, [{"content": "compressed system"}], [])

    # Now simulate 5 turns with the same prefix but alternating tails
    # Tail A: "list files" → Tail B: "show error" → Tail A → Tail B → Tail A
    tails = [
        [{"role": "user", "content": "list files"}],
        [{"role": "user", "content": "show error"}],
        [{"role": "user", "content": "list files"}],  # Repeat of tail 1
        [{"role": "user", "content": "show error"}],  # Repeat of tail 2
        [{"role": "user", "content": "list files"}],  # Repeat of tail 1
    ]

    # Turn 1: Tail A (first time) — miss
    tk_1 = cache.tail_key(model="llama-3", tail_messages=tails[0])
    assert cache.get(tk_1) is None
    cache.put(tk_1, CachedCompression(
        messages=[{"content": "result for list files"}],
        tokens_before=100,
        tokens_after=50,
        transforms=["compress"],
    ))

    # Turn 2: Tail B (first time) — miss
    tk_2 = cache.tail_key(model="llama-3", tail_messages=tails[1])
    assert cache.get(tk_2) is None
    cache.put(tk_2, CachedCompression(
        messages=[{"content": "result for show error"}],
        tokens_before=100,
        tokens_after=60,
        transforms=["compress"],
    ))

    # Turn 3: Tail A again — HIT!
    tk_3 = cache.tail_key(model="llama-3", tail_messages=tails[2])
    result_3 = cache.get(tk_3)
    assert result_3 is not None
    assert result_3.tokens_after == 50

    # Turn 4: Tail B again — HIT!
    tk_4 = cache.tail_key(model="llama-3", tail_messages=tails[3])
    result_4 = cache.get(tk_4)
    assert result_4 is not None
    assert result_4.tokens_after == 60

    # Turn 5: Tail A again — HIT!
    tk_5 = cache.tail_key(model="llama-3", tail_messages=tails[4])
    result_5 = cache.get(tk_5)
    assert result_5 is not None
    assert result_5.tokens_after == 50

    # With prefix-aware caching, after the cold start, all repeated
    # tails get cache hits — dramatically improving throughput.
    assert cache.hits == 3  # Turns 3, 4, 5 were cache hits


def test_prefix_aware_cache_with_stable_prefix_hit():
    """When StablePrefixCache hits, the tail key gives cache hits for repeated tails.

    This is the main use case: same system prompt across many turns, and
    the same conversation patterns repeat. Without prefix-aware caching,
    every unique conversation tail would miss the compression cache.
    With it, repeated patterns get cache hits.
    """
    cache = CompressionResultCache(maxsize=10)
    sp_cache = StablePrefixCache()

    # Common system prompt
    prefix = [{"role": "system", "content": "You are a coding assistant."}]

    # Pre-populate StablePrefixCache (simulates after a few turns)
    sp_key = sp_cache.key_for_messages(prefix, "llama-3")
    sp_cache.put(sp_key, [{"role": "assistant", "content": "compressed system"}], [])

    # Simulate 3 turns with the same prefix but different tails
    # Turn 1: tail "list files" — miss (first time)
    tail_1 = [{"role": "user", "content": "list files"}]
    key_1 = cache.tail_key(model="llama-3", tail_messages=tail_1)
    assert cache.get(key_1) is None
    cache.put(key_1, CachedCompression(
        messages=[{"content": "File1, File2"}],
        tokens_before=100,
        tokens_after=50,
        transforms=["compress"],
    ))

    # Turn 2: tail "show error" — miss (first time)
    tail_2 = [{"role": "user", "content": "show error"}]
    key_2 = cache.tail_key(model="llama-3", tail_messages=tail_2)
    assert cache.get(key_2) is None
    cache.put(key_2, CachedCompression(
        messages=[{"content": "Error: null pointer"}],
        tokens_before=100,
        tokens_after=60,
        transforms=["compress"],
    ))

    # Turn 3: tail "list files" again — HIT!
    key_3 = cache.tail_key(model="llama-3", tail_messages=tail_1)  # Same as tail_1
    result_3 = cache.get(key_3)
    assert result_3 is not None  # Cache hit!
    assert result_3.tokens_after == 50

    # Turn 4: tail "show error" again — HIT!
    key_4 = cache.tail_key(model="llama-3", tail_messages=tail_2)
    result_4 = cache.get(key_4)
    assert result_4 is not None  # Cache hit!
    assert result_4.tokens_after == 60

    # Verify hit rate
    assert cache.ratio == pytest.approx(0.5, abs=0.01)  # 2 hits / 4 lookups
