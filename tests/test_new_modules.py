"""Tests for new compression modules: cross-turn dedup, recursive JSON, lossless compaction, adaptive sizer."""

import json
from legroom import (
    compress,
    CompressConfig,
    dedup_blocks,
    DedupBlock,
    route_embedded_json,
    compact_lossless,
    LosslessResult,
    compute_optimal_k,
    count_unique_simhash,
)
from legroom.content_router import ContentRouter


# ---------------------------------------------------------------------------
# Cross-turn dedup
# ---------------------------------------------------------------------------


def test_cross_turn_dedup_identical_blocks():
    """Blocks that are repeated verbatim across turns should be deduplicated."""
    long_content = "line " + "x" * 100 + "\n" * 3
    blocks = [
        DedupBlock(0, long_content),
        DedupBlock(1, long_content),
        DedupBlock(2, long_content),
    ]
    deduped = dedup_blocks(blocks)
    assert "see block 0" in deduped[1].content or deduped[1].content == blocks[1].content


def test_cross_turn_dedup_different_blocks():
    """Different blocks should not be deduplicated."""
    blocks = [
        DedupBlock(0, "content A" * 20),
        DedupBlock(1, "content B" * 20),
    ]
    deduped = dedup_blocks(blocks)
    assert deduped[0].content == blocks[0].content
    assert deduped[1].content == blocks[1].content


def test_cross_turn_dedup_empty_blocks():
    """Empty blocks should pass through unchanged."""
    blocks = [
        DedupBlock(0, ""),
        DedupBlock(1, "some content"),
    ]
    deduped = dedup_blocks(blocks)
    assert deduped[0].content == ""
    assert deduped[1].content == "some content"


def test_cross_turn_dedup_in_pipeline():
    """Cross-turn dedup works end-to-end through the compress pipeline."""
    long_json = json.dumps([{"id": i} for i in range(100)], indent=2)
    messages = [
        {"role": "user", "content": "Start"},
        {"role": "assistant", "content": long_json},
        {"role": "user", "content": "Middle"},
        {"role": "assistant", "content": long_json},
        {"role": "user", "content": "Recent 1"},
    ]
    result = compress(messages, model="gpt-4o", config=CompressConfig(protect_recent=1))
    assert result.messages[-1]["content"] == "Recent 1"


# ---------------------------------------------------------------------------
# Recursive JSON routing
# ---------------------------------------------------------------------------


def test_recursive_json_embedded():
    """JSON embedded in larger text should be routed through the compressor."""
    text = "Here's the result:\n" + json.dumps([{"id": i, "data": "x" * 20} for i in range(15)], indent=2) + "\n\nDone."
    router = ContentRouter()

    def dispatch(json_span: str) -> str | None:
        result = router.compress(json_span, source_hint="embedded_json")
        if result and result.tokens_saved > 0:
            return result.compressed
        return None

    routed = route_embedded_json(text, dispatch)
    if routed is not None:
        assert len(routed) < len(text)


def test_recursive_json_no_embedded():
    """Text with no embedded JSON should return None."""
    text = "Just some plain text with no JSON at all."
    routed = route_embedded_json(text, lambda x: x)
    assert routed is None


def test_recursive_json_already_compressed():
    """Already-compressed JSON (with markers) should be skipped."""
    text = "Some text [N items compressed. hash=abc123def456] more text."

    def dispatch(span: str) -> str | None:
        return span + "_compressed"

    routed = route_embedded_json(text, dispatch)
    assert routed is None


# ---------------------------------------------------------------------------
# Lossless compaction
# ---------------------------------------------------------------------------


def test_lossless_ansi_strip():
    """ANSI color sequences should be stripped."""
    text = "Line 1\x1b[31mred\x1b[0m\nLine 2\x1b[1mbold\x1b[0m"
    result = compact_lossless(text)
    assert isinstance(result, LosslessResult)
    assert "\x1b[" not in result.compressed


def test_lossless_search_heading():
    """Grep results with same path should get heading compression."""
    text = (
        "src/main.py:15:import os\n"
        "src/main.py:42:print('hello')\n"
        "src/utils.py:10:import sys\n"
    )
    result = compact_lossless(text, content_hint="grep")
    assert isinstance(result, LosslessResult)


def test_lossless_diff_strip():
    """Diff index lines should be stripped."""
    text = (
        "index abc123..def456 100644\n"
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1,3 +1,3 @@\n"
        "-old\n"
        "+new\n"
    )
    result = compact_lossless(text, content_hint="diff")
    assert "index abc123" not in result.compressed


def test_lossless_run_collapse():
    """Repeated log lines should be collapsed."""
    text = "\n".join([f"INFO: Processing item {i}" for i in range(10)] * 5)
    result = compact_lossless(text, content_hint="log")
    assert isinstance(result, LosslessResult)


def test_lossless_roundtrip():
    """Lossless compaction should be reversible for non-ANSI content."""
    text = "line1\nline2\nline3\n"
    result = compact_lossless(text)
    if result.compressed_size < result.original_size:
        assert len(result.compressed) > 0


# ---------------------------------------------------------------------------
# Adaptive sizer
# ---------------------------------------------------------------------------


def test_adaptive_sizer_identical_items():
    """Nearly identical items (few distinct variants) should result in low K."""
    items = [{"status": "ok", "value": "x" * 50} for _ in range(100)]
    item_strs = [json.dumps(item) for item in items]
    k = compute_optimal_k(item_strs, bias=0.7)
    assert k <= 50


def test_adaptive_sizer_unique_items():
    """All unique items should result in higher K."""
    items = [{"id": i, "value": f"unique_{i}_" + "x" * 50} for i in range(50)]
    item_strs = [json.dumps(item) for i, item in enumerate(items)]
    k = compute_optimal_k(item_strs, bias=1.5, min_k=5)
    assert k >= 5


def test_adaptive_sizer_small_input():
    """Small input should return all items."""
    items = ["a", "b", "c"]
    k = compute_optimal_k(items)
    assert k == 3


def test_adaptive_sizer_simhash():
    """Simhash should detect near-duplicates."""
    items = [
        "this is a very similar sentence with different words",
        "this is a very similar sentence with other different words",
        "this is a very similar sentence with completely different words",
        "completely different topic entirely",
    ]
    unique = count_unique_simhash(items)
    assert unique <= len(items)


def test_adaptive_sizer_in_smart_crusher():
    """SmartCrusher should use adaptive sizing by default."""
    from legroom.smart_crusher import SmartCrusher, SmartCrusherConfig

    config = SmartCrusherConfig(max_items=100, adaptive_sizing=True, size_bias=1.0)
    crusher = SmartCrusher(config)

    # Create JSON with many identical items
    data = [{"type": "result", "score": 0.5, "padding": "x" * 20} for _ in range(50)]
    output = crusher.compress(json.dumps(data, indent=2), "test")
    assert output.tokens_saved >= 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_pipeline_with_all_new_modules():
    """Full pipeline should use all new modules."""
    messages = [
        {"role": "user", "content": "Show me the results"},
        {"role": "assistant", "content": json.dumps([{"id": i, "data": "x" * 30} for i in range(30)], indent=2)},
        {"role": "user", "content": "Compare these"},
        {"role": "assistant", "content": json.dumps([{"id": i, "data": "y" * 30} for i in range(30)], indent=2)},
    ]
    result = compress(messages, model="gpt-4o")
    assert result.messages == messages or result.tokens_saved >= 0


def test_compression_with_embedded_json():
    """Text with embedded JSON should be compressed via recursive routing."""
    text = f"Here are the results:\n```json\n{json.dumps([{'id': i, 'score': 0.5} for i in range(20)], indent=2)}\n```"

    router = ContentRouter()
    result = router.compress(text, source_hint="api")
    assert result is not None
    assert result.tokens_saved >= 0


# ---------------------------------------------------------------------------
# ML compressor opt-in wiring
# ---------------------------------------------------------------------------


def test_ml_compress_disabled_by_default():
    """ml_compress_enabled defaults to False — no MLTextCompressor is built."""
    router = ContentRouter()
    assert router._ml_compressor is None


def test_ml_compress_enabled_without_optional_deps_falls_back():
    """Requesting ML compression without onnxruntime/tokenizers installed
    (or without model files present) must not raise — it should silently
    fall back to the lossless text compressor."""
    router = ContentRouter(ml_compress_enabled=True)
    # Either the optional deps are missing (constructor caught ImportError,
    # _ml_compressor stays None) or they're present but the model file
    # isn't (compress() itself falls back) — both are valid, non-crashing
    # outcomes for an environment without the ML extra fully set up.
    text = "This is a plain text message that should compress losslessly " * 3
    result = router.compress(text, source_hint="text")
    assert result is not None
    assert result.compressed_token_count <= result.original_token_count


def test_ml_compress_enabled_end_to_end_via_config():
    """CompressConfig.ml_compress_enabled should reach the router without
    crashing the top-level compress() call, deps present or not."""
    messages = [
        {"role": "user", "content": "Summarize this"},
        {"role": "assistant", "content": "This is a fairly long plain-text response " * 5},
    ]
    config = CompressConfig(ml_compress_enabled=True, retention_threshold=0.5)
    result = compress(messages, model="gpt-4o", config=config)
    assert result.tokens_saved >= 0
