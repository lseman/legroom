"""Tests for JsonCanonicalizer — JSON canonicalization for llama.cpp KV cache alignment."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

# Ensure legroom is importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legroom.compressors.json_canonicalizer import (
    JsonCanonicalizer,
    JsonCanonicalizeResult,
    _find_json_spans,
)


@pytest.fixture
def canonicalizer():
    return JsonCanonicalizer()


# ── Standalone canonicalization ──────────────────────────────────────


class TestCanonicalizeValue:
    """Test canonicalization of parsed JSON values."""

    def test_object_key_sorting(self, canonicalizer):
        """Keys should be sorted alphabetically."""
        original = {"zebra": 1, "apple": 2, "mango": 3}
        result = canonicalizer._canonicalize_value(original)
        assert list(result.keys()) == ["apple", "mango", "zebra"]

    def test_nested_object_key_sorting(self, canonicalizer):
        """Nested objects should also have sorted keys."""
        original = {"z": {"b": 2, "a": 1}, "a": {"y": 2, "x": 1}}
        result = canonicalizer._canonicalize_value(original)
        assert list(result["z"].keys()) == ["a", "b"]
        assert list(result["a"].keys()) == ["x", "y"]

    def test_float_to_int_normalization(self, canonicalizer):
        """Floats that are mathematically integers should become ints."""
        assert canonicalizer._canonicalize_value(1.0) == 1
        assert canonicalizer._canonicalize_value(0.0) == 0
        assert canonicalizer._canonicalize_value(-5.0) == -5
        # Non-integer floats should stay as floats
        assert canonicalizer._canonicalize_value(1.5) == 1.5
        assert canonicalizer._canonicalize_value(-3.14) == -3.14

    def test_preserves_regular_ints(self, canonicalizer):
        """Regular ints should remain ints."""
        assert canonicalizer._canonicalize_value(42) == 42
        assert canonicalizer._canonicalize_value(0) == 0
        assert canonicalizer._canonicalize_value(-100) == -100

    def test_preserves_strings(self, canonicalizer):
        """Strings should remain strings."""
        assert canonicalizer._canonicalize_value("hello") == "hello"
        assert canonicalizer._canonicalize_value("") == ""

    def test_preserves_booleans_and_none(self, canonicalizer):
        """Booleans and None should remain unchanged."""
        assert canonicalizer._canonicalize_value(True) is True
        assert canonicalizer._canonicalize_value(False) is False
        assert canonicalizer._canonicalize_value(None) is None

    def test_list_order_preserved(self, canonicalizer):
        """List element order should be preserved."""
        original = [3, 1, 2]
        result = canonicalizer._canonicalize_value(original)
        assert result == [3, 1, 2]

    def test_list_elements_canonicalized(self, canonicalizer):
        """Elements within lists should be canonicalized."""
        original = [{"z": 1, "a": 2}, {"b": 3.0}]
        result = canonicalizer._canonicalize_value(original)
        assert list(result[0].keys()) == ["a", "z"]
        assert result[1]["b"] == 3


class TestRoundTrip:
    """Ensure canonicalization is semantically lossless."""

    def test_roundtrip_dict(self):
        """Canonicalized JSON should parse back to the same dict."""
        original = {"zebra": 1, "apple": {"y": 2.0, "x": [3, 1]}}
        canonicalizer = JsonCanonicalizer()
        canonical = canonicalizer._canonicalize_value(original)
        serialized = json.dumps(canonical, sort_keys=False, separators=(",", ":"))
        parsed = json.loads(serialized)
        assert parsed == original

    def test_roundtrip_float_to_int(self):
        """1.0 canonicalized to 1 should parse back to the same number."""
        canonicalizer = JsonCanonicalizer()
        assert canonicalizer._canonicalize_value(1.0) == 1
        # 1 (int) and 1.0 (float) are equal in JSON
        assert json.loads(json.dumps(1)) == json.loads(json.dumps(1.0))


# ── JSON span finding ────────────────────────────────────────────────


class TestFindJsonSpans:
    """Test JSON span detection in text."""

    def test_standalone_object(self):
        """Should find a standalone JSON object (must be >= 30 bytes)."""
        text = 'Hello {"name": "world", "value": 12345}!'
        spans = _find_json_spans(text)
        assert len(spans) == 1
        start, end = spans[0]
        assert text[start:end] == '{"name": "world", "value": 12345}'

    def test_standalone_array(self):
        """Should find a standalone JSON array (must be >= 30 bytes)."""
        text = "Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"
        spans = _find_json_spans(text)
        assert len(spans) == 1
        start, end = spans[0]
        assert text[start:end] == "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"

    def test_nested_json(self):
        """Should find nested JSON correctly."""
        text = 'Outer {"inner": {"deep": true}, "z": 1}'
        spans = _find_json_spans(text)
        assert len(spans) == 1
        start, end = spans[0]
        assert text[start:end] == '{"inner": {"deep": true}, "z": 1}'

    def test_no_json(self):
        """Should return empty for text without JSON."""
        text = "Just plain text with no JSON"
        assert _find_json_spans(text) == []

    def test_multiple_json_spans(self):
        """Should find multiple JSON spans (each >= 30 bytes)."""
        text = '{"key_one": "value_one", "key_two": "value_two"} and {"other": "data", "item": "more_data_here"}'
        spans = _find_json_spans(text)
        assert len(spans) == 2

    def test_skips_small_json(self):
        """Should skip JSON spans smaller than the minimum."""
        text = '{"x": 1}'  # 8 chars, below 30-char minimum
        spans = _find_json_spans(text)
        assert spans == []

    def test_skips_templating(self):
        """Should skip JSON-like content after letters or $."""
        text = 'foo{"bar": 1} and ${x}'
        spans = _find_json_spans(text)
        assert spans == []


# ── Text-level canonicalization ──────────────────────────────────────


class TestCanonicalizeText:
    """Test full text canonicalization."""

    def test_key_ordering(self, canonicalizer):
        """Unsorted keys should become sorted."""
        text = '{"zebra": 1, "apple": 2, "mango": 3}'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        # Keys should be alphabetically sorted
        data = json.loads(result)
        assert list(data.keys()) == ["apple", "mango", "zebra"]

    def test_compact_formatting(self, canonicalizer):
        """Whitespace should be removed."""
        text = '{\n  "name": "test",\n  "value": 42\n}'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        # Should be compact
        assert '\n' not in result
        assert '  ' not in result

    def test_numeric_normalization(self, canonicalizer):
        """1.0 should become 1, 1.5 should stay."""
        text = '{"float_int": 1.0, "float_val": 1.5, "int": 42}'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        data = json.loads(result)
        assert isinstance(data["float_int"], int)
        assert data["float_int"] == 1
        assert isinstance(data["float_val"], float)
        assert data["float_val"] == 1.5
        assert isinstance(data["int"], int)

    def test_preserves_non_json_text(self, canonicalizer):
        """Non-JSON text should pass through unchanged."""
        text = "This is plain text with no JSON at all."
        result, count = canonicalizer._canonicalize_text(text)
        assert result == text
        assert count == 0

    def test_mixed_content(self, canonicalizer):
        """JSON embedded in text should be canonicalized; rest unchanged."""
        text = 'Before {"zebra": 1.0, "apple": 2.0, "mango": 3.0} After'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        assert result.startswith("Before ")
        assert result.endswith(" After")
        # JSON part should be canonicalized
        assert '"a"' in result or '"apple"' in result
        assert result.index('"apple"') < result.index('"zebra"')

    def test_large_json_skipped(self, canonicalizer):
        """JSON spans larger than max should be skipped."""
        large_json = '{"data": ' + json.dumps({"key": "x" * 40000}) + "}"
        _result, count = canonicalizer._canonicalize_text(large_json)
        assert count == 0  # Skipped due to 32KB size limit

    def test_empty_text(self, canonicalizer):
        """Empty text should return unchanged."""
        result, count = canonicalizer._canonicalize_text("")
        assert result == ""
        assert count == 0


# ── Message-level canonicalization ───────────────────────────────────


class TestCanonicalizeMessages:
    """Test canonicalization across message lists."""

    def test_llama_cpp_backend_enabled(self, canonicalizer):
        """Should canonicalize when backend='llama_cpp'."""
        messages = [
            {"role": "user", "content": '{"zebra_key": "value", "apple_key": "value", "mango_key": "value", "key": "extra"}'},
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        assert result.canonicalized_count == 1
        data = json.loads(result.messages[0]["content"])
        assert list(data.keys()) == ["apple_key", "key", "mango_key", "zebra_key"]

    def test_openai_backend_disabled(self, canonicalizer):
        """Should NOT canonicalize when backend='openai'."""
        messages = [
            {"role": "user", "content": '{"zebra": 1, "apple": 2}'},
        ]
        result = canonicalizer.canonicalize(messages, backend="openai")
        assert result.canonicalized_count == 0
        assert result.messages == messages

    def test_multiple_messages(self, canonicalizer):
        """Should handle multiple messages."""
        messages = [
            {"role": "user", "content": '{"zebra": "val", "apple": "val", "mango": "val", "key": "extra"}'},
            {"role": "assistant", "content": "Plain text"},
            {"role": "user", "content": '{"y_key": 2.0, "x_key": 3.0, "z_key": 4.0}'},
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        assert result.canonicalized_count == 2

    def test_tool_role_messages(self, canonicalizer):
        """Should handle tool role messages."""
        messages = [
            {"role": "tool", "tool_call_id": "1", "content": '{"zebra_result": true, "apple_result": true, "mango_result": true, "key": "val"}'},
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        assert result.canonicalized_count == 1

    def test_list_result(self, canonicalizer):
        """Should return proper JsonCanonicalizeResult."""
        messages = [
            {"role": "user", "content": '{"z": 1.0, "a": 2.0}'},
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        assert isinstance(result, JsonCanonicalizeResult)
        assert isinstance(result.messages, list)
        assert result.tokens_saved >= 0


# ── KV cache alignment benefit ───────────────────────────────────────


class TestKvCacheBenefit:
    """Demonstrate KV cache alignment benefit."""

    def test_same_semantics_different_tokenization(self):
        """Show that different JSON formatting tokens differently."""
        json_a = '{"name": "test", "value": 42}'
        json_b = '{"value": 42, "name": "test"}'
        # Different key order → different token sequences
        assert json_a != json_b

    def test_canonicalization_makes_them_identical(self):
        """After canonicalization, both should be identical."""
        canonicalizer = JsonCanonicalizer()
        data_a = json.loads('{"name": "test", "value": 42}')
        data_b = json.loads('{"value": 42, "name": "test"}')
        canon_a = canonicalizer._canonicalize_value(data_a)
        canon_b = canonicalizer._canonicalize_value(data_b)
        canon_a_str = json.dumps(canon_a, separators=(",", ":"))
        canon_b_str = json.dumps(canon_b, separators=(",", ":"))
        assert canon_a_str == canon_b_str
        # Both should have sorted keys
        assert canon_a_str == '{"name":"test","value":42}'

    def test_numeric_normalization_benefit(self):
        """1.0 and 1 tokenize differently; canonicalization fixes this."""
        canonicalizer = JsonCanonicalizer()
        val_1_float = canonicalizer._canonicalize_value(1.0)
        val_1_int = canonicalizer._canonicalize_value(1)
        assert val_1_float == val_1_int == 1
        assert json.dumps(val_1_float) == json.dumps(val_1_int)


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and potential crash scenarios."""

    def test_very_large_numbers(self, canonicalizer):
        """Handle very large numbers gracefully."""
        text = '{"big": 1e308, "small": 1e-308}'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        data = json.loads(result)
        assert data["big"] == 1e308
        assert data["small"] == 1e-308

    def test_unicode_strings(self, canonicalizer):
        """Handle unicode in strings."""
        text = '{"emoji": "🎉", "chinese": "你好"}'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        data = json.loads(result)
        assert data["emoji"] == "🎉"
        assert data["chinese"] == "你好"

    def test_deeply_nested_json(self, canonicalizer):
        """Handle deeply nested structures."""
        text = '{"a": {"b": {"c": {"d": {"e": 1}}}}}'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        data = json.loads(result)
        assert data["a"]["b"]["c"]["d"]["e"] == 1

    def test_arrays_of_objects(self, canonicalizer):
        """Handle arrays of objects (order preserved, keys sorted)."""
        text = '[{"z": 1, "a": 2}, {"y": 3, "b": 4}]'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        data = json.loads(result)
        # List order preserved
        assert len(data) == 2
        assert list(data[0].keys()) == ["a", "z"]
        assert list(data[1].keys()) == ["b", "y"]

    def test_empty_objects_and_arrays(self, canonicalizer):
        """Handle empty containers."""
        text = '{"empty_obj": {}, "empty_arr": []}'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        data = json.loads(result)
        assert data["empty_obj"] == {}
        assert data["empty_arr"] == []

    def test_null_and_booleans(self, canonicalizer):
        """Handle null, true, false."""
        text = '{"a": null, "b": true, "c": false}'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        data = json.loads(result)
        assert data["a"] is None
        assert data["b"] is True
        assert data["c"] is False

    def test_special_floats(self, canonicalizer):
        """Handle special float values (Infinity not valid JSON)."""
        # Note: Infinity/NaN are not valid JSON, so these are
        # typically represented as strings in practice
        text = '{"normal": 3.14, "scientific": 1e10, "neg_exp": 1e-5}'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        data = json.loads(result)
        assert data["normal"] == 3.14

    def test_mixed_types_in_array(self, canonicalizer):
        """Handle arrays with mixed types."""
        text = '[1, "two", true, null, {"key": "val"}]'
        result, count = canonicalizer._canonicalize_text(text)
        assert count == 1
        data = json.loads(result)
        assert data == [1, "two", True, None, {"key": "val"}]


# ── Tool call argument canonicalization ──────────────────────────────


class TestToolCallCanonicalization:
    """Test canonicalization of tool call arguments."""

    def test_tool_call_key_sorting(self, canonicalizer):
        """Tool call arguments should have sorted keys."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"zebra": 1, "apple": 2, "mango": 3}'
                    }
                }]
            }
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        assert result.tool_call_canonicalized == 1
        args = json.loads(result.messages[0]["tool_calls"][0]["function"]["arguments"])
        assert list(args.keys()) == ["apple", "mango", "zebra"]

    def test_tool_call_no_mutation(self, canonicalizer):
        """Original message should NOT be mutated."""
        original_args = '{"zebra": 1, "apple": 2, "mango": 3}'
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "read", "arguments": original_args}
                }]
            }
        ]
        canonicalizer.canonicalize(messages, backend="llama_cpp")
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == original_args

    def test_tool_call_kv_cache_alignment(self, canonicalizer):
        """Two identical semantic tool calls produce identical output."""
        msg1 = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "read", "arguments": '{"apple": 2.0, "zebra": 1.0}'}
            }]
        }]
        msg2 = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_2",
                "function": {"name": "read", "arguments": '{"zebra": 1.0, "apple": 2.0}'}
            }]
        }]
        r1 = canonicalizer.canonicalize(msg1, backend="llama_cpp")
        r2 = canonicalizer.canonicalize(msg2, backend="llama_cpp")
        assert r1.messages[0]["tool_calls"][0]["function"]["arguments"] == \
               r2.messages[0]["tool_calls"][0]["function"]["arguments"]

    def test_tool_call_float_to_int(self, canonicalizer):
        """Floats that are integers should become ints."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "write",
                        "arguments": '{"line": 1.0, "count": 2.0}'
                    }
                }]
            }
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        args = result.messages[0]["tool_calls"][0]["function"]["arguments"]
        # 1.0 → 1 (int), 2.0 → 2 (int)
        assert "1" in args and "2" in args
        assert "1.0" not in args and "2.0" not in args

    def test_tool_call_multiple_calls(self, canonicalizer):
        """Handle multiple tool calls in one message."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "read", "arguments": '{"z": 1.0, "a": 2.0}'}},
                    {"id": "call_2", "function": {"name": "write", "arguments": '{"y": 3.0, "x": 4.0}'}},
                ]
            }
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        assert result.tool_call_canonicalized == 2
        assert result.messages[0]["tool_calls"][0]["function"]["arguments"] == '{"a":2,"z":1}'
        assert result.messages[0]["tool_calls"][1]["function"]["arguments"] == '{"x":4,"y":3}'

    def test_tool_call_no_arguments(self, canonicalizer):
        """Handle tool calls without arguments."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "ping"}
                }]
            }
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        assert result.tool_call_canonicalized == 0

    def test_tool_call_invalid_json(self, canonicalizer):
        """Handle tool calls with invalid JSON arguments."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "read", "arguments": "not valid json"}
                }]
            }
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        assert result.tool_call_canonicalized == 0

    def test_tool_call_empty_arguments(self, canonicalizer):
        """Handle tool calls with empty arguments."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "read", "arguments": ""}
                }]
            }
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        assert result.tool_call_canonicalized == 0

    def test_tool_call_openai_backend_noop(self, canonicalizer):
        """Tool calls should NOT be canonicalized for openai backend."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "read", "arguments": '{"z": 1, "a": 2}'}
                }]
            }
        ]
        result = canonicalizer.canonicalize(messages, backend="openai")
        assert result.tool_call_canonicalized == 0
        assert result.messages[0]["tool_calls"][0]["function"]["arguments"] == '{"z": 1, "a": 2}'

    def test_tool_call_nested_objects(self, canonicalizer):
        """Handle nested objects in tool call arguments."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "create",
                        "arguments": '{"zebra": {"b": 2, "a": 1}, "apple": 3}'
                    }
                }]
            }
        ]
        result = canonicalizer.canonicalize(messages, backend="llama_cpp")
        args = json.loads(result.messages[0]["tool_calls"][0]["function"]["arguments"])
        assert list(args.keys()) == ["apple", "zebra"]
        assert list(args["zebra"].keys()) == ["a", "b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
