"""Regression tests for code-aware compression."""

from legroom import CompressConfig, compress
from legroom.compressors.code_compressor import CodeCompressor


def test_long_fenced_string_literal_is_compressed_without_group_error():
    inner = "x" * 100
    content = f'```python\nvalue = "{inner}"\n```'

    result = CodeCompressor().compress(content)

    assert '"' + ("x" * 30) + "..." + ("x" * 20) + '"' in result.compressed
    assert inner not in result.compressed


def test_prefixed_and_escaped_string_literal_keeps_prefix():
    inner = "path=\\\"quoted\\\"/" + ("z" * 80)
    content = f'```python\nvalue = r"{inner}"\n```'

    result = CodeCompressor().compress(content)

    assert 'r"' in result.compressed
    assert "..." in result.compressed


def test_pipeline_does_not_report_compress_phase_failure_for_long_literal():
    inner = "payload" * 30
    messages = [{"role": "tool", "content": f'```python\ndata = "{inner}"\n```'}]

    result = compress(
        messages,
        config=CompressConfig(
            output_shaping=False,
            cache_align_enabled=False,
            cross_turn_dedup_enabled=False,
            read_lifecycle_enabled=False,
            ccr_enabled=False,
        ),
    )

    assert not any("Compression failed" in warning for warning in result.warnings)
