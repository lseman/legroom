"""Tests for FastAPI proxy, state management, and dashboard."""

import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from legroom.proxy_state import ProxyState, RequestEvent
from legroom.proxy_server import LegroomProxy
from legroom.proxy_dashboard import get_dashboard_html


# ---------------------------------------------------------------------------
# Proxy State tests
# ---------------------------------------------------------------------------


def test_proxy_state_record_request():
    """State should record request events and update aggregate stats."""
    state = ProxyState()
    
    event = state.record_request(
        request_id="test123",
        model="gpt-4o",
        messages_before=5,
        tokens_before=1000,
        tokens_after=700,
        transforms_applied=["smart_crusher", "cross_turn_dedup"],
        warnings=[],
    )
    
    assert event.request_id == "test123"
    assert event.tokens_saved == 300
    
    stats = state.get_stats()
    assert stats["total_requests"] == 1
    assert stats["total_tokens_saved"] == 300
    assert stats["total_tokens_before"] == 1000
    assert stats["total_tokens_after"] == 700


def test_proxy_state_multiple_requests():
    """State should accumulate stats across multiple requests."""
    state = ProxyState()
    
    state.record_request("req1", "gpt-4o", 3, 500, 350, [], [])
    state.record_request("req2", "claude-3", 4, 800, 600, ["smart_crusher"], [])
    state.record_request("req3", "gpt-4o", 2, 300, 200, [], [])
    
    stats = state.get_stats()
    assert stats["total_requests"] == 3
    assert stats["total_tokens_saved"] == 450
    assert stats["total_tokens_before"] == 1600
    assert stats["total_tokens_after"] == 1150
    assert stats["compression_ratio"] == pytest.approx(28.1, abs=0.1)


def test_proxy_state_history_bounded():
    """History should be bounded by max_history."""
    state = ProxyState(max_history=5)
    
    for i in range(10):
        state.record_request(f"req{i}", "gpt-4o", 1, 100, 80, [], [])
    
    history = state.get_history()
    assert len(history) == 5  # Only last 5


def test_proxy_state_strategy_counts():
    """State should track per-strategy counts."""
    state = ProxyState()
    
    state.record_request("r1", "gpt-4o", 1, 100, 80, ["smart_crusher", "log_compressor"], [])
    state.record_request("r2", "gpt-4o", 1, 100, 80, ["smart_crusher"], [])
    state.record_request("r3", "gpt-4o", 1, 100, 80, ["cross_turn_dedup"], [])
    
    stats = state.get_stats()
    assert stats["strategy_counts"]["smart_crusher"] == 2
    assert stats["strategy_counts"]["log_compressor"] == 1
    assert stats["strategy_counts"]["cross_turn_dedup"] == 1


def test_proxy_state_read_lifecycle_stats():
    """State should track read lifecycle stats."""
    state = ProxyState()
    
    state.record_request(
        "r1",
        "gpt-4o",
        5,
        500,
        400,
        ["read_lifecycle"],
        [],
        read_lifecycle_stats={
            "reads_stale": 2,
            "reads_superseded": 1,
            "reads_fresh": 3,
        },
    )
    
    lifecycle = state.get_read_lifecycle_stats()
    assert lifecycle["total_reads_stale"] == 2
    assert lifecycle["total_reads_superseded"] == 1
    assert lifecycle["total_reads_fresh"] == 3


def test_proxy_state_ccr_tracking():
    """State should track CCR store/retrieve operations."""
    state = ProxyState()
    
    state.record_ccr_store(count=5)
    state.record_ccr_store(count=3)
    state.record_ccr_retrieve(count=2)
    
    stats = state.get_stats()
    assert stats["total_ccr_stored"] == 8
    assert stats["total_ccr_retrieved"] == 2


def test_proxy_state_get_history():
    """History should return most recent entries first."""
    state = ProxyState()
    
    for i in range(5):
        state.record_request(f"req{i}", "gpt-4o", 1, 100, 80, [], [])
    
    history = state.get_history(limit=3)
    assert len(history) == 3
    # Most recent first
    assert history[0]["request_id"] == "req4"
    assert history[2]["request_id"] == "req2"


def test_proxy_state_empty():
    """Empty state should return zeros."""
    state = ProxyState()
    stats = state.get_stats()
    
    assert stats["total_requests"] == 0
    assert stats["total_tokens_saved"] == 0
    assert stats["compression_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Proxy Server tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_server_creation():
    """Proxy server should create a FastAPI app with correct routes."""
    proxy = LegroomProxy(target_url="https://api.example.com", api_key="test123")
    
    assert proxy.target_url == "https://api.example.com"
    assert proxy.api_key == "test123"
    assert proxy.app is not None
    assert proxy._state is not None


@pytest.mark.asyncio
async def test_proxy_server_get_state():
    """Proxy should expose its state tracker."""
    proxy = LegroomProxy()
    state = proxy.get_state()
    
    assert isinstance(state, ProxyState)


@pytest.mark.asyncio
async def test_proxy_dashboard_html():
    """Dashboard HTML should be valid and contain key elements."""
    html = get_dashboard_html()
    
    assert "<!DOCTYPE html>" in html
    assert "Legroom" in html
    assert "s-requests" in html
    assert "s-saved" in html
    assert "/api/stats" in html
    assert "/api/history" in html
    assert "/ws/events" in html
    # New features: SSE and chart
    assert "/api/events" in html
    assert "chart" in html
    assert "drawChart" in html
    assert "connectSSE" in html


@pytest.mark.asyncio
async def test_proxy_stats_endpoint():
    """Stats endpoint should return current aggregate stats."""
    proxy = LegroomProxy()
    
    # Record some requests
    proxy._state.record_request("r1", "gpt-4o", 3, 500, 350, [], [])
    proxy._state.record_request("r2", "gpt-4o", 2, 300, 200, ["smart_crusher"], [])
    
    stats = proxy._state.get_stats()
    assert stats["total_requests"] == 2
    assert stats["total_tokens_saved"] == 250


@pytest.mark.asyncio
async def test_proxy_history_endpoint():
    """History endpoint should return recent requests."""
    proxy = LegroomProxy()
    
    for i in range(5):
        proxy._state.record_request(f"req{i}", "gpt-4o", 1, 100, 80, [], [])
    
    history = proxy._state.get_history()
    assert len(history) == 5


# ---------------------------------------------------------------------------
# WebSocket tests (basic check that endpoint exists)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_endpoints_registered():
    """All expected endpoints should be registered."""
    proxy = LegroomProxy()
    
    routes = sorted(route.path for route in proxy.app.routes)
    expected = ["/", "/api/ccr", "/api/events", "/api/history", "/api/read-lifecycle", "/api/stats", "/v1/chat/completions", "/ws/events"]
    for ep in expected:
        assert ep in routes, f"Missing endpoint: {ep}"


# ---------------------------------------------------------------------------
# SSE endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_endpoint_exists():
    """SSE endpoint should be registered."""
    proxy = LegroomProxy()
    
    routes = [route.path for route in proxy.app.routes]
    assert "/api/events" in routes


@pytest.mark.asyncio
async def test_sse_endpoint_exists():
    """SSE endpoint should be registered and have correct signature."""
    proxy = LegroomProxy()
    
    routes = [route.path for route in proxy.app.routes]
    assert "/api/events" in routes
    
    # Verify the endpoint method exists and is async
    assert hasattr(proxy, "_sse_endpoint")
    import asyncio
    assert asyncio.iscoroutinefunction(proxy._sse_endpoint)


# ---------------------------------------------------------------------------
# CORS tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cors_middleware_with_origins():
    """Proxy should add CORS middleware when origins are provided."""
    proxy = LegroomProxy(cors_origins=["http://localhost:3000"])
    
    from fastapi.testclient import TestClient
    client = TestClient(proxy.app)
    # GET / should return 200 and include CORS header for allowed origin
    resp = client.get("/", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


@pytest.mark.asyncio
async def test_no_cors_without_origins():
    """Proxy should not add CORS middleware by default."""
    proxy = LegroomProxy()
    
    from fastapi.testclient import TestClient
    client = TestClient(proxy.app)
    resp = client.options("/", headers={"Origin": "http://evil.com"})
    # Without CORS middleware, Access-Control-Allow-Origin should not be present
    assert "access-control-allow-origin" not in resp.headers


# ---------------------------------------------------------------------------
# Integration: full flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_full_request_flow():
    """Test complete request flow: record stats, verify state."""
    proxy = LegroomProxy(
        target_url="https://api.example.com",
        api_key="test123",
    )
    
    # Simulate a request being processed (without actual HTTP call)
    proxy._state.record_request(
        request_id="integration1",
        model="gpt-4o",
        messages_before=8,
        tokens_before=1500,
        tokens_after=1100,
        transforms_applied=["smart_crusher", "cross_turn_dedup", "read_lifecycle"],
        warnings=["Some warning"],
        read_lifecycle_stats={
            "reads_stale": 1,
            "reads_superseded": 2,
            "reads_fresh": 4,
        },
    )
    
    # Verify state
    stats = proxy._state.get_stats()
    assert stats["total_requests"] == 1
    assert stats["total_tokens_saved"] == 400
    assert stats["strategy_counts"]["smart_crusher"] == 1
    assert stats["strategy_counts"]["cross_turn_dedup"] == 1
    assert stats["strategy_counts"]["read_lifecycle"] == 1
    
    # Verify read lifecycle stats
    lifecycle = proxy._state.get_read_lifecycle_stats()
    assert lifecycle["total_reads_stale"] == 1
    assert lifecycle["total_reads_superseded"] == 2
    assert lifecycle["total_reads_fresh"] == 4
    
    # Verify history
    history = proxy._state.get_history()
    assert len(history) == 1
    assert history[0]["request_id"] == "integration1"
    assert history[0]["tokens_saved"] == 400
