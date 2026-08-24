"""Regression coverage for the post-proxy correctness audit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from legroom import CompressConfig, compress
from legroom.ccr.compression_store import CompressionStore
from legroom.ccr.marker_resolution import parse_markers
from legroom.pipeline import TransformPipeline
from legroom.proxy.protocols import compression_view
from legroom.proxy.proxy_state import ProxyState
from legroom.read_lifecycle import (
    ReadClassification,
    ReadLifecycleConfig,
    ReadState,
    _replace_content,
)


def test_responses_view_only_selects_message_items():
    function_call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{}",
    }
    function_output = {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "result",
    }
    message = {"type": "message", "role": "user", "content": "continue"}
    body = {"input": [function_call, function_output, message]}

    view = compression_view("/v1/responses", body, "token")
    assert view is not None
    assert view.messages == [message]
    assert view.apply(body, [{**message, "content": "short"}])
    assert body["input"] == [
        function_call,
        function_output,
        {**message, "content": "short"},
    ]


def test_optimize_false_is_true_passthrough():
    messages = [
        {
            "role": "tool",
            "content": "UUID 550e8400-e29b-41d4-a716-446655440000",
        }
    ]
    result = compress(
        messages,
        config=CompressConfig(
            optimize=False,
            cache_align_enabled=True,
            output_shaping=True,
            thinking_compact_enabled=True,
        ),
    )
    assert result.messages is messages
    assert result.transforms_applied == []


def test_destructive_cache_alignment_is_opt_in():
    messages = [
        {
            "role": "tool",
            "content": "UUID 550e8400-e29b-41d4-a716-446655440000",
        }
    ]
    result = compress(messages, config=CompressConfig(cache_align_enabled=False))
    assert result.messages[0]["content"] == messages[0]["content"]


def test_compression_store_is_safe_under_concurrent_eviction():
    store = CompressionStore(max_entries=8)

    def write(index: int) -> str:
        return store.store(f"original-{index}", f"short-{index}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(write, range(500)))

    assert store.get_stats()["entries"] == 8
    assert any(store.retrieve(hash_key) is not None for hash_key in hashes[-8:])


def test_read_lifecycle_storage_failure_preserves_original():
    class BrokenStore:
        def store(self, **kwargs):
            raise RuntimeError("disk unavailable")

    content = "important exact content" * 10
    classification = ReadClassification(
        msg_index=0,
        tool_call_id="tool_1",
        file_path="important.py",
        state=ReadState.STALE,
    )
    replaced, output, hash_key = _replace_content(
        content,
        classification,
        ReadLifecycleConfig(min_size_bytes=1),
        BrokenStore(),
    )
    assert not replaced
    assert output == content
    assert hash_key is None


def test_read_lifecycle_marker_is_discoverable_by_ccr_parser():
    assert parse_markers(
        "[Read content stale: x.py. Retrieve original: hash=abcdef1234567890]"
    ) == ["abcdef1234567890"]


def test_proxy_state_broadcasts_and_bounds_slow_subscribers():
    state = ProxyState()
    first = state.subscribe(max_events=2)
    second = state.subscribe(max_events=2)
    for index in range(3):
        state.record_request(str(index), "gpt", 1, 10, 5, ["compress"], [])

    assert first.qsize() == second.qsize() == 2
    assert first.get_nowait() == second.get_nowait()
    assert state.dropped_live_events == 2
    state.unsubscribe(first)
    state.unsubscribe(second)
    assert not state._subscribers


def test_strict_pipeline_reraises_phase_failure(monkeypatch: pytest.MonkeyPatch):
    pipeline = TransformPipeline(strict=True)

    def fail(*args, **kwargs):
        raise RuntimeError("broken compressor")

    monkeypatch.setattr(pipeline.compressor, "apply", fail)
    with pytest.raises(RuntimeError, match="broken compressor"):
        pipeline.apply([{"role": "user", "content": "hello"}])


def test_fail_open_pipeline_reports_phase_and_exception(monkeypatch: pytest.MonkeyPatch):
    pipeline = TransformPipeline(strict=False)

    def fail(*args, **kwargs):
        raise RuntimeError("broken compressor")

    monkeypatch.setattr(pipeline.compressor, "apply", fail)
    result = pipeline.apply([{"role": "user", "content": "hello"}])
    assert result.warnings == ["Compression failed: RuntimeError: broken compressor"]
