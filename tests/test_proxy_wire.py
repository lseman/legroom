"""Behavioral tests for the proxy's HTTP wire contract."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse

from legroom.proxy.headers import filter_request_headers
from legroom.proxy.proxy_server import LegroomProxy


class _BytesStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def _request(
    body: bytes,
    *,
    path: str = "/v1/chat/completions",
    method: str = "POST",
    query: bytes = b"",
) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query,
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        },
        receive,
    )


@pytest.mark.asyncio
async def test_unmodified_request_preserves_original_bytes():
    original = b'{ "model": "gpt-4o", "messages": [] , "emoji": "\xf0\x9f\x94\xa5" }'
    captured: list[bytes] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(await request.aread())
        return httpx.Response(
            200,
            stream=_BytesStream(b'{"ok":true}'),
            headers={"content-type": "application/json"},
        )

    proxy = LegroomProxy(target_url="https://upstream.test/v1/chat/completions")
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await proxy._handle_request(_request(original))
    finally:
        await proxy.app.state.http_client.aclose()

    assert response.status_code == 200
    assert captured == [original]


@pytest.mark.asyncio
async def test_mutated_request_is_canonical_utf8(monkeypatch: pytest.MonkeyPatch):
    original = json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "🔥 verbose"}]},
        indent=2,
        ensure_ascii=False,
    ).encode()
    captured: list[bytes] = []

    result = SimpleNamespace(
        messages=[{"role": "user", "content": "🔥"}],
        tokens_before=2,
        tokens_after=1,
        transforms_applied=["test"],
        warnings=[],
        metadata={},
    )
    monkeypatch.setattr("legroom.proxy.proxy_server.compress", lambda *args, **kwargs: result)

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(await request.aread())
        return httpx.Response(200, stream=_BytesStream(b'{"ok":true}'))

    proxy = LegroomProxy(target_url="https://upstream.test/v1/chat/completions")
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        await proxy._handle_request(_request(original))
    finally:
        await proxy.app.state.http_client.aclose()

    assert captured == [
        '{"model":"gpt-4o","messages":[{"role":"user","content":"🔥"}]}'.encode()
    ]


@pytest.mark.asyncio
async def test_non_json_response_is_preserved_verbatim():
    payload = b"upstream overloaded\n"

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            stream=_BytesStream(payload),
            headers={"content-type": "text/plain", "x-upstream": "yes"},
        )

    proxy = LegroomProxy(target_url="https://upstream.test/v1/chat/completions")
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await proxy._handle_request(_request(b'{"messages":[]}'))
    finally:
        await proxy.app.state.http_client.aclose()

    assert response.status_code == 503
    assert response.body == payload
    assert response.headers["content-type"] == "text/plain"
    assert response.headers["x-upstream"] == "yes"


@pytest.mark.asyncio
async def test_unrelated_tools_do_not_force_a_second_response_read():
    payload = b'{"choices":[{"finish_reason":"stop","message":{"content":"ok"}}]}'
    calls = 0

    async def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=_BytesStream(payload))

    request_body = json.dumps(
        {
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "other_tool"}}],
        }
    ).encode()
    proxy = LegroomProxy(target_url="https://upstream.test/v1/chat/completions")
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await proxy._handle_request(_request(request_body))
    finally:
        await proxy.app.state.http_client.aclose()

    assert calls == 1
    assert response.body == payload


@pytest.mark.asyncio
async def test_stream_preserves_status_headers_and_raw_chunks():
    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            stream=_BytesStream(b'data: {"error":', b'"slow down"}\r\n\r\n'),
            headers={"content-type": "text/event-stream", "retry-after": "7"},
        )

    proxy = LegroomProxy(target_url="https://upstream.test/v1/chat/completions")
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    response = await proxy._handle_request(_request(b'{"messages":[],"stream":true}'))
    assert isinstance(response, StreamingResponse)
    assert response.status_code == 429
    assert response.headers["content-type"] == "text/event-stream"
    assert response.headers["retry-after"] == "7"
    assert proxy._metrics.inflight == 1
    assert b"".join([chunk async for chunk in response.body_iterator]) == (
        b'data: {"error":"slow down"}\r\n\r\n'
    )
    assert proxy._metrics.inflight == 0
    await proxy.app.state.http_client.aclose()


def test_connection_nominated_headers_are_removed():
    filtered = filter_request_headers(
        [
            (b"host", b"proxy.test"),
            (b"connection", b"keep-alive, X-Internal-Hop"),
            (b"x-internal-hop", b"secret"),
            (b"x-end-to-end", b"keep"),
        ]
    )
    assert filtered == [(b"x-end-to-end", b"keep")]


@pytest.mark.asyncio
async def test_responses_input_is_compressed_in_native_document(monkeypatch: pytest.MonkeyPatch):
    seen_by_compressor: list[list[dict]] = []
    captured: list[dict] = []

    def fake_compress(messages, **kwargs):
        seen_by_compressor.append(messages)
        return SimpleNamespace(
            messages=[{**messages[0], "content": "short"}],
            tokens_before=10,
            tokens_after=2,
            transforms_applied=["test"],
            warnings=[],
            metadata={},
        )

    monkeypatch.setattr("legroom.proxy.proxy_server.compress", fake_compress)

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(await request.aread()))
        assert request.url.path == "/v1/responses"
        return httpx.Response(200, stream=_BytesStream(b'{"id":"resp_1"}'))

    body = {
        "model": "gpt-5",
        "input": [{"type": "message", "role": "user", "content": "very long"}],
        "metadata": {"keep": True},
    }
    proxy = LegroomProxy(target_url="https://upstream.test/v1/chat/completions")
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await proxy._handle_request(
            _request(json.dumps(body).encode(), path="/v1/responses")
        )
    finally:
        await proxy.app.state.http_client.aclose()

    assert response.status_code == 200
    assert seen_by_compressor == [body["input"]]
    assert captured[0]["input"][0]["content"] == "short"
    assert captured[0]["metadata"] == {"keep": True}


@pytest.mark.asyncio
async def test_cache_mode_freezes_prior_response_items(monkeypatch: pytest.MonkeyPatch):
    compressed_inputs: list[list[dict]] = []
    captured: list[dict] = []

    def fake_compress(messages, **kwargs):
        compressed_inputs.append(messages)
        return SimpleNamespace(
            messages=[{**messages[0], "content": "live-short"}],
            tokens_before=5,
            tokens_after=2,
            transforms_applied=["test"],
            warnings=[],
            metadata={},
        )

    monkeypatch.setattr("legroom.proxy.proxy_server.compress", fake_compress)

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(await request.aread()))
        return httpx.Response(200, stream=_BytesStream(b"{}"))

    prior = {"type": "message", "role": "assistant", "content": "frozen"}
    live = {"type": "message", "role": "user", "content": "live-long"}
    proxy = LegroomProxy(target_url="https://upstream.test", mode="cache")
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        await proxy._handle_request(
            _request(
                json.dumps({"input": [prior, live]}).encode(),
                path="/v1/responses",
            )
        )
    finally:
        await proxy.app.state.http_client.aclose()

    assert compressed_inputs == [[live]]
    assert captured[0]["input"] == [prior, {**live, "content": "live-short"}]


@pytest.mark.asyncio
async def test_catch_all_preserves_method_path_query_and_binary_body():
    captured = {}

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.update(
            method=request.method,
            path=request.url.path,
            query=request.url.query,
            body=await request.aread(),
        )
        return httpx.Response(201, stream=_BytesStream(b"accepted"))

    proxy = LegroomProxy(target_url="https://upstream.test/base")
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    payload = b"\x00\xffnot-json"
    try:
        response = await proxy._handle_request(
            _request(payload, path="/uploads/blob", method="PUT", query=b"part=2")
        )
    finally:
        await proxy.app.state.http_client.aclose()

    assert captured == {
        "method": "PUT",
        "path": "/base/uploads/blob",
        "query": b"part=2",
        "body": payload,
    }
    assert response.status_code == 201
    assert response.body == b"accepted"


@pytest.mark.asyncio
async def test_compression_result_cache_and_prometheus_metrics(monkeypatch: pytest.MonkeyPatch):
    compression_calls = 0

    def fake_compress(messages, **kwargs):
        nonlocal compression_calls
        compression_calls += 1
        return SimpleNamespace(
            messages=[{**messages[0], "content": "short"}],
            tokens_before=9,
            tokens_after=2,
            transforms_applied=["test"],
            warnings=[],
            metadata={},
        )

    monkeypatch.setattr("legroom.proxy.proxy_server.compress", fake_compress)

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_BytesStream(b"{}"))

    proxy = LegroomProxy(target_url="https://upstream.test")
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    body = json.dumps({"messages": [{"role": "user", "content": "long"}]}).encode()
    try:
        await proxy._handle_request(_request(body))
        await proxy._handle_request(_request(body))
    finally:
        await proxy.app.state.http_client.aclose()

    assert compression_calls == 1
    assert proxy._compression_cache.hits == 1
    metrics = proxy._metrics.render_prometheus()
    assert 'route="/v1/chat/completions",status="200"} 2' in metrics
    assert "legroom_proxy_request_duration_seconds_count" in metrics


@pytest.mark.asyncio
async def test_provider_cache_controls_and_usage_metrics():
    captured: dict = {}

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(await request.aread()))
        payload = {
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 70},
            }
        }
        return httpx.Response(200, stream=_BytesStream(json.dumps(payload).encode()))

    proxy = LegroomProxy(
        target_url="https://upstream.test",
        compress_context=False,
        provider_cache_mode="explicit",
        provider_cache_ttl="24h",
    )
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    body = json.dumps(
        {"model": "gpt-5.6", "messages": [{"role": "user", "content": "hello"}]}
    ).encode()
    try:
        await proxy._handle_request(_request(body))
    finally:
        await proxy.app.state.http_client.aclose()

    assert captured["prompt_cache_key"].startswith("legroom-")
    assert captured["prompt_cache_retention"] == "24h"
    metrics = proxy._metrics.render_prometheus()
    assert "legroom_provider_cache_read_tokens_total 70" in metrics
    assert "legroom_provider_input_tokens_total 100" in metrics


@pytest.mark.asyncio
async def test_shadow_mode_measures_without_mutating_request(monkeypatch: pytest.MonkeyPatch):
    original_messages = [{"role": "user", "content": "long original"}]
    captured: dict = {}
    result = SimpleNamespace(
        messages=[{"role": "user", "content": "short"}],
        tokens_before=10,
        tokens_after=2,
        transforms_applied=["compress"],
        warnings=[],
        metadata={"phase_reports": []},
    )
    monkeypatch.setattr("legroom.proxy.proxy_server.compress", lambda *args, **kwargs: result)

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(await request.aread()))
        return httpx.Response(200, stream=_BytesStream(b"{}"))

    proxy = LegroomProxy(target_url="https://upstream.test", shadow_mode=True)
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        await proxy._handle_request(_request(json.dumps({"messages": original_messages}).encode()))
    finally:
        await proxy.app.state.http_client.aclose()

    assert captured["messages"] == original_messages
    assert proxy._metrics.shadow_requests == 1
    assert proxy._metrics.shadow_tokens_potentially_saved == 8


@pytest.mark.asyncio
async def test_quality_failure_rolls_back_candidate(monkeypatch: pytest.MonkeyPatch):
    original_messages = [{"role": "user", "content": "critical identifier abc123"}]
    captured: dict = {}
    result = SimpleNamespace(
        messages=[{"role": "user", "content": "short"}],
        tokens_before=10,
        tokens_after=2,
        transforms_applied=["compress"],
        warnings=[],
        metadata={
            "phase_reports": [{"name": "Compression", "status": "applied"}]
        },
    )
    monkeypatch.setattr("legroom.proxy.proxy_server.compress", lambda *args, **kwargs: result)

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(await request.aread()))
        return httpx.Response(200, stream=_BytesStream(b"{}"))

    proxy = LegroomProxy(
        target_url="https://upstream.test",
        quality_evaluator=lambda original, candidate: 0.0,
    )
    proxy.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        await proxy._handle_request(_request(json.dumps({"messages": original_messages}).encode()))
    finally:
        await proxy.app.state.http_client.aclose()

    assert captured["messages"] == original_messages
    assert proxy._metrics.errors["quality_rollback"] == 1


@pytest.mark.asyncio
async def test_ccr_retry_uses_identity_encoding_and_returns_raw_bytes():
    captured_headers = None
    final = b'{"choices":[{"finish_reason":"stop","message":{"content":"done"}}]}'

    async def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal captured_headers
        captured_headers = request.headers
        return httpx.Response(
            200,
            stream=_BytesStream(final),
            headers={"content-type": "application/json", "content-length": str(len(final))},
        )

    proxy = LegroomProxy(target_url="https://upstream.test")
    hash_key = proxy._compression_store.store("original", "short")
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    response_document = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "ccr_retrieve",
                                "arguments": json.dumps({"hash": hash_key}),
                            },
                        }
                    ],
                },
            }
        ]
    }
    try:
        document, response, raw = await proxy._resolve_ccr_retrieve_loop(
            client,
            {"messages": []},
            response_document,
            [(b"accept-encoding", b"gzip")],
            httpx.Response(200),
            "https://upstream.test/v1/chat/completions",
        )
    finally:
        await client.aclose()

    assert captured_headers is not None
    assert captured_headers["accept-encoding"] == "identity"
    assert response.status_code == 200
    assert raw == final
    assert document["choices"][0]["message"]["content"] == "done"
