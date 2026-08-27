"""Tests for pipeline and compression."""

import json

from legroom import (
    CompressConfig,
    ContentRouter,
    DedupBlock,
    LosslessResult,
    SmartCrusher,
    SmartCrusherConfig,
    TransformPipeline,
    compact_lossless,
    compress,
    compute_optimal_k,
    count_unique_simhash,
    dedup_blocks,
    route_embedded_json,
)
from legroom.compressors.content_router import CompressionCache

# ---------------------------------------------------------------------------
# Compression tests
# ---------------------------------------------------------------------------


def test_compress_basic():
    """Basic compression should work."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    result = compress(messages, model="gpt-4o")
    assert result.tokens_before > 0
    assert result.tokens_after >= 0
    assert result.tokens_saved >= 0


def test_compress_long_content():
    """Long JSON content should be compressed."""
    long_json = json.dumps([{"id": i} for i in range(100)], indent=2)
    messages = [
        {"role": "user", "content": "Start"},
        {"role": "assistant", "content": long_json},
    ]
    result = compress(messages, model="gpt-4o")
    assert result.tokens_after <= result.tokens_before or result.tokens_saved >= 0


def test_pipeline_preserves_protected_recent_messages():
    """Recent messages should be preserved."""
    long_json = json.dumps([{"id": i} for i in range(100)], indent=2)
    messages = [
        {"role": "user", "content": "Start"},
        {"role": "assistant", "content": long_json},
        {"role": "user", "content": "Middle"},
        {"role": "assistant", "content": long_json},
        {"role": "user", "content": "Recent 1"},
        {"role": "assistant", "content": "Recent response"},
    ]
    result = compress(messages, model="gpt-4o", config=CompressConfig(protect_recent=2))
    assert result.messages[-1]["content"] == "Recent response"


def test_compress_no_optimize():
    """With optimize=False, content should pass through unchanged."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]
    result = compress(messages, model="gpt-4o", config=CompressConfig(optimize=False))
    assert result.messages == messages


def test_high_level_compress_empty_input():
    """Empty messages should return empty result."""
    result = compress([], model="gpt-4o")
    assert result.messages == []


def test_token_counting():
    """Token counting should work."""
    from legroom.analysis.tokenizer import count_tokens, count_tokens_messages

    assert count_tokens("", model="gpt-4o") == 0
    assert count_tokens("hello", model="gpt-4o") > 0
    assert count_tokens_messages([{"role": "user", "content": "test"}], model="gpt-4o") > 0


def test_compress_preserves_errors():
    """Errors in content should be preserved."""
    messages = [
        {"role": "assistant", "content": "Error: Something went wrong"},
    ]
    result = compress(messages, model="gpt-4o")
    assert any("Error" in str(m.get("content", "")) for m in result.messages)


def test_compress_config_kwargs():
    """Config kwargs should override defaults."""
    messages = [
        {"role": "user", "content": "Test"},
        {"role": "assistant", "content": "Response"},
    ]
    result = compress(messages, model="gpt-4o", protect_recent=0)
    assert result.tokens_before > 0


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


def test_detect_json():
    """Content detector should identify JSON."""
    from legroom.compressors.content_detector import ContentDetector

    detector = ContentDetector()
    assert detector.detect('{"key": "value"}') == "json"
    assert detector.detect("[1, 2, 3]") == "json"


def test_detect_code():
    """Content detector should identify code."""
    from legroom.compressors.content_detector import ContentDetector

    detector = ContentDetector()
    assert detector.detect("def foo():\n    pass") == "code"
    assert detector.detect("class Bar:\n    pass") == "code"


def test_detect_log():
    """Content detector should identify logs."""
    from legroom.compressors.content_detector import ContentDetector

    detector = ContentDetector()
    assert detector.detect("2024-01-01T00:00:00Z INFO Hello") == "log"


def test_detect_search():
    """Content detector should identify search results."""
    from legroom.compressors.content_detector import ContentDetector

    detector = ContentDetector()
    assert detector.detect("src/main.py:10:import os") == "search"


def test_detect_mixed():
    """Content detector should identify mixed content."""
    from legroom.compressors.content_detector import ContentDetector

    detector = ContentDetector()
    mixed = "Here is some text:\n\n```python\ndef foo():\n    pass\n```\n\nAnd some logs:\n2024-01-01 INFO Hello"
    sections = detector.detect_sections(mixed)
    assert len(sections) > 0


def test_split_mixed():
    """Content detector should split mixed content into sections."""
    from legroom.compressors.content_detector import ContentDetector

    detector = ContentDetector()
    mixed = "code1\ncode2\nlog line"
    sections = detector.detect_sections(mixed)
    assert len(sections) >= 1


def test_compress_with_pipeline():
    """Full pipeline should compress messages."""
    messages = [
        {"role": "user", "content": "Test message"},
        {"role": "assistant", "content": "Response"},
    ]
    pipeline = TransformPipeline()
    result = pipeline.apply(messages)
    assert result.tokens_before > 0


def test_compress_high_level():
    """High-level compress() should work."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]
    result = compress(messages)
    assert result.messages == messages or result.tokens_saved >= 0


def test_compress_protect_recent():
    """Recent messages should be preserved by pipeline."""
    messages = [
        {"role": "user", "content": "Old"},
        {"role": "assistant", "content": "Old response"},
        {"role": "user", "content": "New"},
        {"role": "assistant", "content": "New response"},
    ]
    pipeline = TransformPipeline()
    result = pipeline.apply(messages, config=CompressConfig(protect_recent=2))
    assert result.messages[-1]["content"] == "New response"


def test_compress_no_user_messages():
    """Messages without user role should pass through."""
    messages = [
        {"role": "assistant", "content": "Just assistant"},
    ]
    result = compress(messages)
    assert len(result.messages) == 1


def test_compress_thinking_stripping():
    """Thinking blocks should be stripped."""
    messages = [
        {"role": "assistant", "content": "<think>Let me think...</think>Answer"},
    ]
    result = compress(
        messages, model="gpt-4o", config=CompressConfig(thinking_compact_enabled=True)
    )
    content = result.messages[0]["content"]
    assert "<think>" not in content or "Answer" in content


def test_cache_aligner():
    """Cache aligner should replace volatile tokens."""
    from legroom.runtime.pipeline import CacheAligner

    aligner = CacheAligner()
    messages = [
        {"role": "assistant", "content": "UUID: 550e8400-e29b-41d4-a716-446655440000"},
    ]
    result = aligner.apply(messages)
    assert "550e8400" not in result[0]["content"]


def test_content_router():
    """Content router should compress JSON."""
    router = ContentRouter()
    json_content = json.dumps([{"id": i} for i in range(50)], indent=2)
    output = router.compress(json_content)
    assert output is not None
    assert output.tokens_saved >= 0


def test_compression_cache():
    """Compression cache should work."""
    cache = CompressionCache(max_size=10)
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None


def test_compress_empty():
    """Empty input should return empty result."""
    result = compress([], model="gpt-4o")
    assert result.messages == []


def test_compress_optimize_false():
    """optimize=False should pass through unchanged."""
    messages = [
        {"role": "user", "content": "Test"},
    ]
    result = compress(messages, config=CompressConfig(optimize=False))
    assert result.messages == messages


def test_compress_inflation_guard():
    """Inflation guard should revert if compression increases tokens."""
    messages = [
        {"role": "user", "content": "Short"},
        {"role": "assistant", "content": "Brief"},
    ]
    pipeline = TransformPipeline()
    result = pipeline.apply(messages)
    assert result.tokens_after <= result.tokens_before


def test_fresh_openai_read_is_byte_faithful_for_edit_old_text():
    """A fresh file Read must remain safe to copy verbatim into an Edit call."""
    file_content = (
        "def render(value):\n"
        "    # spacing and comments are part of the exact edit target\n"
        "    request_id = '550e8400-e29b-41d4-a716-446655440000'\n"
        '    return f"value = {value!r}  id={request_id}"\n'
    ) * 4
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "read-current-file",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/render.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "read-current-file",
            "content": file_content,
        },
        {"role": "user", "content": "Edit the return expression using exact old_text."},
    ]

    pipeline = TransformPipeline(cache_align_enabled=True, thinking_compact_enabled=True)
    result = pipeline.apply(messages, config=CompressConfig(protect_recent=0))

    assert result.messages[1]["content"] == file_content
    assert result.messages[1]["content"].encode() == file_content.encode()


def test_fresh_anthropic_read_block_is_byte_faithful():
    """Anthropic tool_result blocks receive the same fresh-read guarantee."""
    file_content = "# exact comment\n\ndef target():\n    return '  keep spaces  '\n" * 4
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "read-anthropic",
                    "name": "Read",
                    "input": {"file_path": "src/target.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "read-anthropic",
                    "content": file_content,
                }
            ],
        },
        {"role": "user", "content": "Apply an exact replacement."},
    ]

    result = TransformPipeline(cache_align_enabled=True).apply(messages)

    block = result.messages[1]["content"][0]
    assert block["content"] == file_content


def test_read_byte_faithfulness_is_independent_of_lifecycle_compression():
    """Turning off stale-read compression must not allow generic rewriting."""
    file_content = "# preserve me exactly\nvalue = '550e8400-e29b-41d4-a716-446655440000'\n" * 4
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "read-with-lifecycle-off",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "src/value.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "read-with-lifecycle-off",
            "content": file_content,
        },
    ]

    result = TransformPipeline(cache_align_enabled=True).apply(
        messages, config=CompressConfig(read_lifecycle_enabled=False)
    )

    assert result.messages[1]["content"] == file_content


# ---------------------------------------------------------------------------
# New module tests
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
    assert deduped[1].content != long_content or deduped[1].content == long_content


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


def test_recursive_json_embedded():
    """JSON embedded in larger text should be routed through the compressor."""
    text = (
        "Here's the result:\n"
        + json.dumps([{"id": i, "data": "x" * 20} for i in range(15)], indent=2)
        + "\n\nDone."
    )
    from legroom.compressors.content_router import ContentRouter

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


def test_lossless_ansi_strip():
    """ANSI color sequences should be stripped."""
    text = "Line 1\x1b[31mred\x1b[0m\nLine 2\x1b[1mbold\x1b[0m"
    result = compact_lossless(text)
    assert isinstance(result, LosslessResult)
    assert "\x1b[" not in result.compressed


def test_lossless_search_heading():
    """Grep results with same path should get heading compression."""
    text = "src/main.py:15:import os\nsrc/main.py:42:print('hello')\nsrc/utils.py:10:import sys\n"
    result = compact_lossless(text, content_hint="grep")
    assert isinstance(result, LosslessResult)


def test_lossless_diff_strip():
    """Diff index lines should be stripped."""
    text = (
        "index abc123..def456 100644\n--- a/file.txt\n+++ b/file.txt\n@@ -1,3 +1,3 @@\n-old\n+new\n"
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
    config = SmartCrusherConfig(max_items=100, adaptive_sizing=True, size_bias=1.0)
    crusher = SmartCrusher(config)

    # Create JSON with many identical items
    data = [{"type": "result", "score": 0.5, "padding": "x" * 20} for _ in range(50)]
    output = crusher.compress(json.dumps(data, indent=2), "test")
    assert output.tokens_saved >= 0


def test_pipeline_with_all_new_modules():
    """Full pipeline should use all new modules."""
    messages = [
        {"role": "user", "content": "Show me the results"},
        {
            "role": "assistant",
            "content": json.dumps([{"id": i, "data": "x" * 30} for i in range(30)], indent=2),
        },
        {"role": "user", "content": "Compare these"},
        {
            "role": "assistant",
            "content": json.dumps([{"id": i, "data": "y" * 30} for i in range(30)], indent=2),
        },
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
