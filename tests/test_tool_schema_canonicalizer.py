"""Tests for ToolSchemaCanonicalizer — tool schema canonicalization for llama.cpp KV cache."""

from __future__ import annotations

import json
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legroom.compressors.tool_schema_canonicalizer import (
    ToolSchemaCanonicalizer,
    ToolSchemaCanonicalizeResult,
)


@pytest.fixture
def canonicalizer():
    return ToolSchemaCanonicalizer()


# ── Value canonicalization ───────────────────────────────────────────


class TestCanonicalizeValue:
    """Test canonicalization of parsed JSON values."""

    def test_dict_key_sorting(self, canonicalizer):
        """Dict keys should be sorted alphabetically."""
        original = {"zebra": 1, "apple": 2, "mango": 3}
        result = canonicalizer._canonicalize_value(original)
        assert list(result.keys()) == ["apple", "mango", "zebra"]

    def test_nested_dict_sorting(self, canonicalizer):
        """Nested dicts should also have sorted keys."""
        original = {"z": {"b": 2, "a": 1}, "a": 3}
        result = canonicalizer._canonicalize_value(original)
        assert list(result["z"].keys()) == ["a", "b"]
        assert list(result.keys()) == ["a", "z"]

    def test_float_to_int(self, canonicalizer):
        """Floats that are integers should become ints."""
        assert canonicalizer._canonicalize_value(1.0) == 1
        assert canonicalizer._canonicalize_value(0.0) == 0
        assert canonicalizer._canonicalize_value(1.5) == 1.5
        assert canonicalizer._canonicalize_value(-3.14) == -3.14

    def test_list_order_preserved(self, canonicalizer):
        """List element order should be preserved."""
        original = [{"z": 1}, {"a": 2}]
        result = canonicalizer._canonicalize_value(original)
        assert result[0]["z"] == 1
        assert result[1]["a"] == 2
        assert list(result[0].keys()) == ["z"]
        assert list(result[1].keys()) == ["a"]

    def test_preserves_scalars(self, canonicalizer):
        """Scalars should pass through unchanged."""
        assert canonicalizer._canonicalize_value("hello") == "hello"
        assert canonicalizer._canonicalize_value(42) == 42
        assert canonicalizer._canonicalize_value(True) is True
        assert canonicalizer._canonicalize_value(False) is False
        assert canonicalizer._canonicalize_value(None) is None


# ── Tool canonicalization ────────────────────────────────────────────


class TestCanonicalizeTool:
    """Test single tool definition canonicalization."""

    def test_basic_tool(self, canonicalizer):
        """Basic tool schema should be canonicalized."""
        tool = {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"zebra": 1, "apple": 2}
            }
        }
        result = canonicalizer._canonicalize_tool(tool)
        assert list(result["function"]["parameters"].keys()) == ["apple", "zebra"]

    def test_tool_top_level_keys_sorted(self, canonicalizer):
        """Top-level tool keys should be sorted."""
        tool = {"zebra": "a", "type": "function", "function": {}}
        result = canonicalizer._canonicalize_tool(tool)
        assert list(result.keys()) == ["function", "type", "zebra"]

    def test_tool_without_function(self, canonicalizer):
        """Tool without function key should pass through."""
        tool = {"type": "image"}
        result = canonicalizer._canonicalize_tool(tool)
        assert result == tool

    def test_tool_function_not_dict(self, canonicalizer):
        """Tool with non-dict function should pass through."""
        tool = {"type": "function", "function": "invalid"}
        result = canonicalizer._canonicalize_tool(tool)
        assert result == tool


# ── Body canonicalization ────────────────────────────────────────────


class TestCanonicalizeBody:
    """Test full request body canonicalization."""

    def test_body_with_tools(self, canonicalizer):
        """Body with tools should be canonicalized."""
        body = {
            "model": "llama3",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {"zebra": 1, "apple": 2}
                }
            }]
        }
        result = canonicalizer.canonicalize_body(body)
        assert result.canonicalized_count == 1
        args = result.body["tools"][0]["function"]["parameters"]
        assert list(args.keys()) == ["apple", "zebra"]

    def test_body_without_tools(self, canonicalizer):
        """Body without tools should pass through unchanged."""
        body = {
            "model": "llama3",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = canonicalizer.canonicalize_body(body)
        assert result.canonicalized_count == 0
        assert result.body == body

    def test_body_empty_tools(self, canonicalizer):
        """Body with empty tools array should pass through."""
        body = {
            "model": "llama3",
            "tools": [],
        }
        result = canonicalizer.canonicalize_body(body)
        assert result.canonicalized_count == 0
        assert result.body["tools"] == []

    def test_body_multiple_tools(self, canonicalizer):
        """Body with multiple tools should canonicalize all."""
        body = {
            "model": "llama3",
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"z": 1}}},
                {"type": "function", "function": {"name": "write", "parameters": {"y": 2, "x": 3}}},
            ]
        }
        result = canonicalizer.canonicalize_body(body)
        assert result.canonicalized_count == 2
        assert list(result.body["tools"][0]["function"]["parameters"].keys()) == ["z"]
        assert list(result.body["tools"][1]["function"]["parameters"].keys()) == ["x", "y"]

    def test_body_no_mutation(self, canonicalizer):
        """Original body should NOT be mutated."""
        original_tools = [{
            "type": "function",
            "function": {"name": "read", "parameters": {"z": 1, "a": 2}}
        }]
        body = {"model": "llama3", "tools": original_tools}
        result = canonicalizer.canonicalize_body(body)
        # Original should still have unsorted keys
        assert "z" in body["tools"][0]["function"]["parameters"]
        assert "a" in body["tools"][0]["function"]["parameters"]
        # Result should have sorted keys
        assert list(result.body["tools"][0]["function"]["parameters"].keys()) == ["a", "z"]

    def test_result_type(self, canonicalizer):
        """Should return ToolSchemaCanonicalizeResult."""
        body = {"tools": [{"type": "function", "function": {"name": "read", "parameters": {"z": 1}}}] }
        result = canonicalizer.canonicalize_body(body)
        assert isinstance(result, ToolSchemaCanonicalizeResult)
        assert isinstance(result.body, dict)
        assert isinstance(result.canonicalized_count, int)


# ── JSON string canonicalization ─────────────────────────────────────


class TestCanonicalizeJsonString:
    """Test canonicalizing JSON tool schemas as strings."""

    def test_basic_string(self, canonicalizer):
        """Basic JSON string should be canonicalized."""
        json_str = '{"zebra": 1, "apple": 2}'
        result = canonicalizer.canonicalize_json_string(json_str)
        assert result == '{"apple":2,"zebra":1}'

    def test_invalid_json(self, canonicalizer):
        """Invalid JSON should pass through unchanged."""
        json_str = 'not valid json'
        result = canonicalizer.canonicalize_json_string(json_str)
        assert result == 'not valid json'

    def test_empty_json(self, canonicalizer):
        """Empty JSON should pass through."""
        result = canonicalizer.canonicalize_json_string("")
        assert result == ""

    def test_nested_json(self, canonicalizer):
        """Nested JSON should be recursively canonicalized."""
        json_str = '{"z": {"b": 2, "a": 1}, "a": 3}'
        result = canonicalizer.canonicalize_json_string(json_str)
        data = json.loads(result)
        assert list(data.keys()) == ["a", "z"]
        assert list(data["z"].keys()) == ["a", "b"]

    def test_tool_schema_string(self, canonicalizer):
        """Full tool schema JSON string should be canonicalized."""
        json_str = json.dumps({
            "name": "read_file",
            "parameters": {"type": "object", "properties": {"zebra_path": {"type": "string"}, "apple_line": {"type": "integer"}}}
        })
        result = canonicalizer.canonicalize_json_string(json_str)
        data = json.loads(result)
        # Parameters should have sorted keys
        assert list(data["parameters"].keys()) == ["properties", "type"]
        props = data["parameters"]["properties"]
        assert list(props.keys()) == ["apple_line", "zebra_path"]


# ── KV cache alignment benefit ───────────────────────────────────────


class TestKvCacheBenefit:
    """Demonstrate KV cache alignment benefit for tool schemas."""

    def test_same_semantics_different_formatting(self, canonicalizer):
        """Different tool schema formatting → same canonical output."""
        body1 = {"tools": [{"type": "function", "function": {"name": "read", "parameters": {"zebra": 1, "apple": 2}}}]}
        body2 = {"tools": [{"type": "function", "function": {"name": "read", "parameters": {"apple": 2, "zebra": 1}}}]}

        r1 = canonicalizer.canonicalize_body(body1)
        r2 = canonicalizer.canonicalize_body(body2)

        # Both should have identical canonical output
        assert r1.body["tools"][0]["function"]["parameters"] == \
               r2.body["tools"][0]["function"]["parameters"]
        assert json.dumps(r1.body["tools"]) == json.dumps(r2.body["tools"])

    def test_float_normalization_in_schemas(self, canonicalizer):
        """Float values in schemas should be normalized."""
        body = {"tools": [{"type": "function", "function": {"name": "set", "parameters": {"value": 1.0}}}]}
        result = canonicalizer.canonicalize_body(body)
        args = result.body["tools"][0]["function"]["parameters"]
        assert args["value"] == 1  # int, not float


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
