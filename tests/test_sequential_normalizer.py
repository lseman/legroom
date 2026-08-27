"""Tests for SequentialNumberNormalizer — sequential number normalization for llama.cpp KV cache."""

from __future__ import annotations

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from legroom.compressors.sequential_normalizer import (
    SequentialNumberNormalizer,
    SeqNormalizeResult,
)


@pytest.fixture
def normalizer():
    return SequentialNumberNormalizer()


# ── Line number normalization ────────────────────────────────────────


class TestLineNumbers:
    """Test line number normalization in grep/search output."""

    def test_grep_line_numbers(self, normalizer):
        """Grep results should have normalized line numbers."""
        text = "src/main.py:42: def foo()\nsrc/main.py:43: return bar()"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 2
        assert "src/main.py:LN: def foo()" in result.text
        assert "src/main.py:LN: return bar()" in result.text

    def test_grep_different_files(self, normalizer):
        """Line numbers in different files should all be normalized."""
        text = "file_a.py:10: x\nfile_b.py:100: y\nfile_a.py:11: z"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 3
        assert "file_a.py:LN: x" in result.text
        assert "file_b.py:LN: y" in result.text

    def test_port_numbers_preserved(self, normalizer):
        """Port numbers should NOT be normalized."""
        text = "port 8080, port 443, port 80"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 0
        assert result.text == text


# ── Index normalization ──────────────────────────────────────────────


class TestIndices:
    """Test array index normalization."""

    def test_array_indices(self, normalizer):
        """Array indices should be normalized."""
        text = "items[0]: 'apple'\nitems[1]: 'banana'\nitems[2]: 'cherry'"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 3
        assert "items[IDX]: 'apple'" in result.text

    def test_nested_indices(self, normalizer):
        """Nested indices should be normalized."""
        text = "data[0][1]: 'value'\nmatrix[2][3]: 'another'"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 4  # 4 indices total
        assert "[IDX][IDX]" in result.text


# ── Step/iteration normalization ─────────────────────────────────────


class TestSteps:
    """Test step/iteration number normalization."""

    def test_step_numbers(self, normalizer):
        """Step numbers should be normalized."""
        text = "step 1: init\nstep 2: train\nstep 3: evaluate"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 3
        assert "step IDX: init" in result.text

    def test_iteration_numbers(self, normalizer):
        """Iteration numbers should be normalized."""
        text = "iteration 5: loss=0.5\niteration 6: loss=0.3"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 2
        assert "iteration IDX: loss=0.5" in result.text

    def test_case_insensitive(self, normalizer):
        """Step detection should be case-insensitive."""
        text = "Step 1: a\nSTEP 2: b\nsTeP 3: c"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 3


# ── Result/match normalization ───────────────────────────────────────


class TestResults:
    """Test result/match number normalization."""

    def test_result_numbers(self, normalizer):
        """Result numbers should be normalized."""
        text = "result 1: match found\nresult 2: no match"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 2
        assert "result IDX: match found" in result.text

    def test_match_numbers(self, normalizer):
        """Match numbers should be normalized."""
        text = "match 3: pattern\nmatch 4: pattern"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 2
        assert "match IDX: pattern" in result.text


# ── Log line normalization ───────────────────────────────────────────


class TestLogLines:
    """Test log line number normalization."""

    def test_log_line_reference(self, normalizer):
        """Log line references should be normalized."""
        text = "[line:42] Config loaded\n[LINE:100] Connected"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 2
        assert "[line:LN]" in result.text
        assert "[LINE:LN]" in result.text

    def test_line_prefix(self, normalizer):
        """Line prefix references should be normalized."""
        text = "Line 45: Processing\nLine 46: Done"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 2
        assert "Line LN: Processing" in result.text


# ── Row normalization ────────────────────────────────────────────────


class TestRows:
    """Test row/item number normalization."""

    def test_row_numbers(self, normalizer):
        """Row numbers should be normalized."""
        text = "Row 1: apple\nRow 2: banana\nRow 3: cherry"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 3
        assert "Row IDX: apple" in result.text

    def test_item_numbers(self, normalizer):
        """Item numbers should be normalized."""
        text = "Item 1: x\nItem 2: y"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 2
        assert "Item IDX: x" in result.text


# ── OpenAI backend (no-op) ───────────────────────────────────────────


class TestOpenAIBackend:
    """Test that openai backend skips normalization."""

    def test_no_normalization_openai(self, normalizer):
        """OpenAI backend should not normalize anything."""
        text = "file.py:42: code\nitems[0]: 'a'\nstep 1: init"
        result = normalizer.normalize(text, backend="openai")
        assert result.normalized_count == 0
        assert result.text == text


# ── KV cache alignment benefit ───────────────────────────────────────


class TestKvCacheBenefit:
    """Demonstrate KV cache alignment benefit."""

    def test_grep_alignment(self, normalizer):
        """Grep results from different turns should align."""
        text_a = "file.py:42: code\nfile.py:43: code\nfile.py:44: code"
        text_b = "file.py:45: code\nfile.py:46: code\nfile.py:47: code"
        r_a = normalizer.normalize(text_a, backend="llama_cpp")
        r_b = normalizer.normalize(text_b, backend="llama_cpp")
        assert r_a.text == r_b.text

    def test_result_alignment(self, normalizer):
        """Search results from different turns should align."""
        text_a = "result 1: match\nresult 2: match\nresult 3: match"
        text_b = "result 4: match\nresult 5: match\nresult 6: match"
        r_a = normalizer.normalize(text_a, backend="llama_cpp")
        r_b = normalizer.normalize(text_b, backend="llama_cpp")
        assert r_a.text == r_b.text


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and potential crash scenarios."""

    def test_empty_text(self, normalizer):
        """Empty text should return unchanged."""
        result = normalizer.normalize("", backend="llama_cpp")
        assert result.normalized_count == 0
        assert result.text == ""

    def test_no_numbers(self, normalizer):
        """Text without sequential numbers should be unchanged."""
        text = "hello world, no numbers here"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert result.normalized_count == 0
        assert result.text == text

    def test_mixed_content(self, normalizer):
        """Mixed content should handle all patterns."""
        text = "file.py:42: x\nitems[0]: y\nstep 1: z\nPort 8080 preserved"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert "file.py:LN:" in result.text
        assert "[IDX]" in result.text
        assert "step IDX" in result.text
        assert "Port 8080" in result.text  # Preserved

    def test_result_type(self, normalizer):
        """Should return SeqNormalizeResult."""
        text = "file.py:42: code"
        result = normalizer.normalize(text, backend="llama_cpp")
        assert isinstance(result, SeqNormalizeResult)
        assert isinstance(result.text, str)
        assert isinstance(result.normalized_count, int)
        assert result.tokens_saved >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
