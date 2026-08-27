from __future__ import annotations

from legroom.integration.provider_adapters import OpenAIChatAdapter, OpenAIResponsesAdapter
from legroom.runtime.ir import Conversation
from legroom.runtime.phases import CallablePhase, PhaseContext, PhaseRunner


def test_chat_adapter_round_trips_unknown_fields():
    body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "long text",
                "provider_extension": {"future": True},
            }
        ],
        "top_level_extension": 7,
    }
    adapted = OpenAIChatAdapter().parse(body)
    assert adapted is not None
    message = adapted.conversation.messages[0].with_text("short")
    assert adapted.apply(body, Conversation((message,)))
    assert body["messages"][0]["provider_extension"] == {"future": True}
    assert body["top_level_extension"] == 7


def test_responses_adapter_preserves_opaque_items_and_blocks():
    function_call = {"type": "function_call", "call_id": "1", "arguments": "{}"}
    image = {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
    body = {
        "input": [
            function_call,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello", "x": 1}, image],
            },
        ]
    }
    adapted = OpenAIResponsesAdapter().parse(body)
    assert adapted is not None
    assert not adapted.apply(body, adapted.conversation)
    assert body["input"][0] == function_call
    assert body["input"][1]["content"][1] == image


def test_phase_runner_reports_contract_metrics():
    messages = [{"role": "user", "content": "word " * 100}]
    phase = CallablePhase(
        "truncate_for_test",
        lambda current: [{**current[0], "content": "word"}],
        reversible=True,
        confidence=0.9,
    )
    outcome = PhaseRunner().run(phase, messages, PhaseContext(model="gpt-4o"))
    assert outcome.report.status == "applied"
    assert outcome.report.token_delta < 0
    assert outcome.report.reversible
    assert outcome.report.confidence == 0.9
    assert outcome.report.latency_ms >= 0


def test_phase_runner_rejects_inflation():
    messages = [{"role": "user", "content": "short"}]
    phase = CallablePhase(
        "inflate",
        lambda current: [{**current[0], "content": "long " * 100}],
    )
    outcome = PhaseRunner().run(phase, messages, PhaseContext(model="gpt-4o"))
    assert outcome.messages == messages
    assert outcome.report.status == "rejected"
    assert outcome.report.error == "token count increased"


def test_phase_runner_standardizes_fail_open_errors():
    messages = [{"role": "user", "content": "short"}]

    def fail(current):
        raise RuntimeError("backend unavailable")

    outcome = PhaseRunner().run(
        CallablePhase("learned_compression", fail),
        messages,
        PhaseContext(model="gpt-4o", protected_spans=("message:0",)),
    )
    assert outcome.messages == messages
    assert outcome.report.status == "failed"
    assert outcome.report.error == "RuntimeError: backend unavailable"
    assert outcome.report.protected_spans == ("message:0",)
    assert outcome.warnings == (
        "learned_compression failed: RuntimeError: backend unavailable",
    )


def test_pipeline_exposes_standard_report_for_each_enabled_phase():
    from legroom import CompressConfig, compress

    result = compress(
        [{"role": "user", "content": "hello"}],
        config=CompressConfig(
            read_lifecycle_enabled=False,
            cache_align_enabled=True,
            cross_turn_dedup_enabled=True,
            semantic_dedup_enabled=True,
            kv_cache_optimization_enabled=True,
            output_shaping=True,
            thinking_compact_enabled=True,
            ccr_enabled=True,
        ),
    )
    reports = result.metadata["phase_reports"]
    assert {report["name"] for report in reports} == {
        "read_lifecycle",
        "output_shaper",
        "cache_aligner",
        "cross_turn_dedup",
        "semantic_dedup",
        "kv_cache_optimization",
        "Compression",
        "thinking_compactor",
        "ccr_tool_injection",
    }
    required = {
        "token_delta",
        "protected_spans",
        "reversible",
        "latency_ms",
        "confidence",
        "status",
        "error",
    }
    assert all(required <= report.keys() for report in reports)


def test_kv_cache_optimization_disabled_for_llama_cpp_backend():
    """The prefix-pointer rewrite breaks byte-identical prefix matching that
    llama.cpp's real KV cache relies on, so it must never run against that
    backend even when the flag is explicitly enabled."""
    from legroom import CompressConfig, compress

    shared_prefix = "You are a helpful assistant. " * 20
    messages = [
        {"role": "system", "content": shared_prefix},
        {"role": "user", "content": shared_prefix + "hi"},
        {"role": "user", "content": shared_prefix + "bye"},
    ]
    config = CompressConfig(
        kv_cache_optimization_enabled=True,
        backend="llama_cpp",
        compress_enabled=False,
        cross_turn_dedup_enabled=False,
        semantic_dedup_enabled=False,
        thinking_compact_enabled=False,
        ccr_enabled=False,
        risk_policy_enabled=False,
        read_lifecycle_enabled=False,
    )

    result = compress(messages, config=config)

    assert "kv_cache_optimization" not in result.transforms_applied
    # Alignment transforms (whitespace canonicalization) may still run
    # when backend="llama_cpp" even with compress_enabled=False.
    # The test only verifies that kv_cache_optimization is disabled.


def test_pipeline_skips_phase_disabled_by_calibration():
    from legroom import CompressConfig, compress

    result = compress(
        [{"role": "assistant", "content": "verbose " * 100}],
        config=CompressConfig(
            read_lifecycle_enabled=False,
            output_shaping=False,
            cache_align_enabled=False,
            cross_turn_dedup_enabled=False,
            semantic_dedup_enabled=False,
            kv_cache_optimization_enabled=False,
            thinking_compact_enabled=False,
            ccr_enabled=False,
            disabled_phases=("compress",),
        ),
    )

    report = next(
        phase for phase in result.metadata["phase_reports"] if phase["name"] == "Compression"
    )
    assert report["status"] == "skipped"
    assert report["metadata"]["disabled_by_calibration"] is True
