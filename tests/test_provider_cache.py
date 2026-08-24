from __future__ import annotations

import pytest

from legroom.provider_cache import (
    CachePricing,
    ProviderCachePolicy,
    ProviderCacheUsage,
    StreamingUsageParser,
    parse_cache_usage,
)
from legroom.proxy.observability import ProxyMetrics


def test_cache_policy_adds_stable_controls_without_overwriting_caller():
    first = {"model": "gpt-5.6", "input": [{"role": "user", "content": "one"}]}
    second = {"model": "gpt-5.6", "input": [{"role": "user", "content": "two"}]}
    policy = ProviderCachePolicy(mode="explicit", ttl="24h")
    assert policy.apply(first, protocol="openai_responses")
    assert policy.apply(second, protocol="openai_responses")
    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert first["prompt_cache_retention"] == "24h"

    caller = {"model": "gpt", "prompt_cache_key": "caller-key"}
    ProviderCachePolicy(key="legroom-key").apply(caller, protocol="openai_chat")
    assert caller["prompt_cache_key"] == "caller-key"


def test_cache_policy_rejects_unsupported_retention():
    with pytest.raises(ValueError, match="only '24h'"):
        ProviderCachePolicy(mode="explicit", ttl="30m")


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (
            {"usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 60}}},
            ProviderCacheUsage(input_tokens=100, cached_tokens=60),
        ),
        (
            {"usage": {"input_tokens": 120, "input_tokens_details": {"cached_tokens": 70, "cache_write_tokens": 20}}},
            ProviderCacheUsage(input_tokens=120, cached_tokens=70, cache_write_tokens=20),
        ),
    ],
)
def test_parse_cache_usage(document, expected):
    assert parse_cache_usage(document) == expected


def test_cache_cost_separates_reads_writes_and_uncached_tokens():
    usage = ProviderCacheUsage(input_tokens=1_000, cache_write_tokens=200, cached_tokens=500)
    pricing = CachePricing(uncached_input=10, cache_write=12.5, cache_read=1)
    assert usage.uncached_tokens == 300
    assert usage.cost(pricing) == pytest.approx(0.006)


def test_cache_and_shadow_metrics_are_exported():
    metrics = ProxyMetrics()
    metrics.record_cache_usage(
        input_tokens=100, write_tokens=20, read_tokens=60, cost_usd=0.001
    )
    metrics.record_shadow(100, 70)
    metrics.record_phase_report(
        {"name": "compress", "status": "applied", "latency_ms": 2.5, "token_delta": -30}
    )
    rendered = metrics.render_prometheus()
    assert "legroom_provider_cache_read_tokens_total 60" in rendered
    assert "legroom_provider_cache_write_tokens_total 20" in rendered
    assert "legroom_provider_input_cost_usd_total 0.001000000" in rendered
    assert "legroom_shadow_tokens_potentially_saved_total 30" in rendered
    assert 'legroom_phase_runs_total{phase="compress",status="applied"} 1' in rendered
    assert 'legroom_phase_token_delta_total{phase="compress"} -30' in rendered


def test_streaming_usage_parser_handles_fragmented_sse():
    parser = StreamingUsageParser()
    parser.feed(b'event: response.completed\ndata: {"response":{"usage":{"input_tokens":')
    parser.feed(
        b'120,"input_tokens_details":{"cached_tokens":80,"cache_write_tokens":10}}}}\n\n'
    )
    assert parser.usage == ProviderCacheUsage(
        input_tokens=120, cached_tokens=80, cache_write_tokens=10
    )
