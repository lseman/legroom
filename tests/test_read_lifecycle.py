"""Tests for Read Lifecycle — stale/superseded Read compression."""

import json
from legroom import (
    compress,
    CompressConfig,
    ReadLifecycleConfig,
    classify_reads,
    ReadLifecycleResult,
)


# ---------------------------------------------------------------------------
# OpenAI format tests
# ---------------------------------------------------------------------------


def test_read_lifecycle_stale_read():
    """A Read followed by an Edit should be classified as STALE."""
    messages = [
        {"role": "assistant", "content": "Let me read the file."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "x" * 200,  # Simulated file content
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "Now I'll edit the file."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_edit_1",
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "arguments": json.dumps({"file_path": "src/main.py", "old_text": "old", "new_text": "new"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "File edited."},
    ]

    config = ReadLifecycleConfig()
    result = classify_reads(messages, config)

    assert result.reads_total == 1
    assert result.reads_stale == 1
    assert result.reads_fresh == 0
    # The stale read should be replaced with a marker
    tool_msg = result.messages[2]
    assert "stale" in tool_msg["content"].lower()


def test_read_lifecycle_superseded_read():
    """A Read followed by a later full Read of the same file should be SUPERSEDED."""
    messages = [
        {"role": "assistant", "content": "Reading file first."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "x" * 200,
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "Reading file again."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_2",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "y" * 200,
            "tool_call_id": "tool_read_2",
        },
    ]

    config = ReadLifecycleConfig()
    result = classify_reads(messages, config)

    assert result.reads_total == 2
    assert result.reads_superseded == 1
    assert result.reads_fresh == 1


def test_read_lifecycle_fresh_read():
    """A Read with no subsequent Edit or re-Read should be FRESH."""
    messages = [
        {"role": "assistant", "content": "Let me read the file."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "x" * 200,
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "File read, now processing."},
    ]

    config = ReadLifecycleConfig()
    result = classify_reads(messages, config)

    assert result.reads_total == 1
    assert result.reads_fresh == 1


def test_read_lifecycle_disabled():
    """When disabled, messages should pass through unchanged."""
    messages = [
        {"role": "assistant", "content": "Reading."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "x" * 200,
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "Editing."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_edit_1",
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "edit_result",
            "tool_call_id": "tool_edit_1",
        },
    ]

    config = ReadLifecycleConfig(enabled=False)
    result = classify_reads(messages, config)

    assert result.reads_total == 0
    assert result.messages == messages


def test_read_lifecycle_min_size():
    """Tiny Read outputs should not be replaced."""
    messages = [
        {"role": "assistant", "content": "Reading."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "tiny",  # Only 4 bytes
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "Editing."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_edit_1",
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "edit_result",
            "tool_call_id": "tool_edit_1",
        },
    ]

    config = ReadLifecycleConfig(min_size_bytes=50)
    result = classify_reads(messages, config)

    assert result.reads_total == 1
    assert result.reads_stale == 1
    # But the content should NOT be replaced (too small)
    assert result.messages[2]["content"] == "tiny"


# ---------------------------------------------------------------------------
# Anthropic format tests
# ---------------------------------------------------------------------------


def test_read_lifecycle_anthropic_format():
    """Read lifecycle should work with Anthropic content blocks."""
    messages = [
        {"role": "assistant", "content": "Reading."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_read_1",
                    "name": "Read",
                    "input": {"file_path": "src/main.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_read_1",
                    "content": "x" * 200,
                }
            ],
        },
        {"role": "assistant", "content": "Editing."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_edit_1",
                    "name": "Edit",
                    "input": {"file_path": "src/main.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_edit_1",
                    "content": "edit_result",
                }
            ],
        },
    ]

    config = ReadLifecycleConfig()
    result = classify_reads(messages, config)

    assert result.reads_total == 1
    assert result.reads_stale == 1
    # The stale read's content block should be replaced
    tool_result = result.messages[2]["content"][0]
    assert "stale" in tool_result["content"].lower()


# ---------------------------------------------------------------------------
# Partial read coverage tests
# ---------------------------------------------------------------------------


def test_read_lifecycle_partial_read_not_superseded():
    """A partial read should NOT be superseded by a different partial read."""
    messages = [
        {"role": "assistant", "content": "Reading lines 1-100."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py", "offset": 0, "limit": 100}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "x" * 200,
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "Reading lines 50-150."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_2",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py", "offset": 50, "limit": 100}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "y" * 200,
            "tool_call_id": "tool_read_2",
        },
    ]

    config = ReadLifecycleConfig()
    result = classify_reads(messages, config)

    # First read is NOT superseded because the second read doesn't fully cover it
    assert result.reads_total == 2
    assert result.reads_fresh == 2  # Neither is stale, and first is not fully covered


def test_read_lifecycle_partial_read_superseded():
    """A partial read should be superseded by a full-file read."""
    messages = [
        {"role": "assistant", "content": "Reading lines 1-100."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py", "offset": 0, "limit": 100}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "x" * 200,
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "Reading entire file."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_2",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),  # No offset/limit = full file
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "y" * 200,
            "tool_call_id": "tool_read_2",
        },
    ]

    config = ReadLifecycleConfig()
    result = classify_reads(messages, config)

    assert result.reads_total == 2
    assert result.reads_superseded == 1  # First read fully covered by second
    assert result.reads_fresh == 1


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


def test_read_lifecycle_in_pipeline():
    """Read Lifecycle should integrate with the full pipeline."""
    messages = [
        {"role": "assistant", "content": "Reading."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps([{"id": i} for i in range(50)], indent=2),
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "Editing."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_edit_1",
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "tool_edit_1",
        },
        {"role": "user", "content": "Recent message"},
    ]

    result = compress(messages, model="gpt-4o", config=CompressConfig(protect_recent=1))
    assert result.tokens_saved >= 0


def test_read_lifecycle_disabled_in_pipeline():
    """read_lifecycle_enabled=False should skip the phase."""
    messages = [
        {"role": "assistant", "content": "Reading."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "x" * 200,
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "Editing."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_edit_1",
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "tool_edit_1",
        },
    ]

    result = compress(
        messages,
        model="gpt-4o",
        config=CompressConfig(read_lifecycle_enabled=False),
    )
    assert "read_lifecycle" not in result.transforms_applied


def test_read_lifecycle_compress_stale_false():
    """compress_stale=False prevents stale classification and replacement."""
    messages = [
        {"role": "assistant", "content": "Reading."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "x" * 200,
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "Editing."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_edit_1",
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "tool_edit_1",
        },
    ]

    config = ReadLifecycleConfig(compress_stale=False)
    result = classify_reads(messages, config)

    # When compress_stale=False, no stale classification happens
    assert result.reads_total == 1
    assert result.reads_fresh == 1
    # Content should NOT be replaced
    assert result.messages[2]["content"] == "x" * 200


def test_read_lifecycle_compress_superseded_false():
    """compress_superseded=False prevents superseded classification."""
    messages = [
        {"role": "assistant", "content": "Reading."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "x" * 200,
            "tool_call_id": "tool_read_1",
        },
        {"role": "assistant", "content": "Reading again."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_read_2",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "src/main.py"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "y" * 200,
            "tool_call_id": "tool_read_2",
        },
    ]

    config = ReadLifecycleConfig(compress_superseded=False)
    result = classify_reads(messages, config)

    # When compress_superseded=False, no superseded classification happens
    assert result.reads_total == 2
    assert result.reads_fresh == 2
    # Neither read should be replaced
    assert result.messages[2]["content"] == "x" * 200
