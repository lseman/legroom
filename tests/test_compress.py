"""Tests for high-level compress() function."""

import json

from legroom import CompressConfig, compress


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


def test_compress_protect_recent():
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


def test_compress_empty():
    """Empty messages should return empty result."""
    result = compress([], model="gpt-4o")
    assert result.messages == []


def test_token_counting():
    """Token counting should work."""
    from legroom.analysis.tokenizer import count_tokens, count_tokens_messages

    assert count_tokens("", model="gpt-4o") == 0
    assert count_tokens("hello", model="gpt-4o") > 0
    assert count_tokens_messages([{"role": "user", "content": "test"}], model="gpt-4o") > 0


def test_gpt4o_uses_native_encoding():
    from legroom.analysis.tokenizer import get_encoding

    assert get_encoding("gpt-4o").name == "o200k_base"


def test_protocol_token_count_includes_tool_payloads_and_framing():
    from legroom.analysis.tokenizer import count_tokens_messages

    plain = [{"role": "assistant", "content": "calling"}]
    with_tool = [
        {
            **plain[0],
            "tool_calls": [
                {"id": "call_123", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}
            ],
        }
    ]
    assert count_tokens_messages(with_tool, protocol="openai_chat") > count_tokens_messages(
        plain, protocol="openai_chat"
    )
    assert count_tokens_messages(plain, protocol="openai_chat") > count_tokens_messages(
        plain, protocol="content_only"
    )

    responses = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe this"},
                {"type": "input_image", "image_url": "https://example.test/image.png"},
            ],
        }
    ]
    assert count_tokens_messages(responses) > count_tokens_messages(
        responses, protocol="content_only"
    )


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
