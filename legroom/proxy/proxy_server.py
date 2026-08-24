"""FastAPI proxy server — HTTP reverse proxy with compression and live stats."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..ccr.compression_store import CompressionStore
from ..ccr.tool_injection import create_ccr_tool_definition
from ..compress import compress
from ..config import CompressConfig
from .body_forwarding import select_outbound_body
from .compression_cache import CachedCompression, CompressionResultCache
from .headers import filter_request_headers, filter_response_headers
from .observability import ProxyMetrics
from .protocols import ProxyMode, compression_view, normalize_mode
from .proxy_state import ProxyState

_CCR_RETRIEVE_MAX_HOPS = 4

logger = logging.getLogger(__name__)


def _with_raw_headers(response: Response, headers: list[tuple[bytes, bytes]]) -> Response:
    """Attach filtered raw headers without collapsing repeated fields."""
    response.raw_headers = headers
    return response


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage httpx client lifecycle."""
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        http2=False,
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
    )
    _app.state.http_client = client
    yield
    await client.aclose()


def _resolve_api_key(api_key: str | None = None) -> str | None:
    """Resolve API key from parameter or environment variable."""
    if api_key:
        return api_key
    # Check environment variables in order of precedence
    for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LEGROOM_API_KEY"):
        key = os.environ.get(env_var)
        if key:
            return key
    return None


class LegroomProxy:
    """FastAPI-based reverse proxy with compression tracking and dashboard."""

    def __init__(
        self,
        target_url: str | None = None,
        api_key: str | None = None,
        compress_context: bool = True,
        max_history: int = 1000,
        cors_origins: list[str] | None = None,
        max_compression_concurrency: int = 4,
        mode: str = "token",
        compression_cache_size: int = 256,
        compression_cache_ttl: float = 300.0,
    ) -> None:
        # Resolve API key from parameter or environment variable
        self.api_key = _resolve_api_key(api_key)
        
        # Use default target if not specified
        if target_url is None:
            target_url = os.environ.get(
                "LEGROOM_TARGET_URL",
                "http://127.0.0.1:8080/v1/chat/completions",
            )
        self.target_url = target_url
        self.compress_context = compress_context
        self.mode: ProxyMode = normalize_mode(mode)
        self._state = ProxyState(max_history=max_history)
        self._compression_cache = CompressionResultCache(
            maxsize=compression_cache_size, ttl_seconds=compression_cache_ttl
        )
        # Compatibility alias for dashboard callers from pre-P2 releases.
        self._request_dedup = self._compression_cache
        self._metrics = ProxyMetrics()
        self._cors_origins = cors_origins
        self._compression_store = CompressionStore()
        if max_compression_concurrency < 1:
            raise ValueError("max_compression_concurrency must be at least 1")
        self._compression_slots = asyncio.Semaphore(max_compression_concurrency)

        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title="Legroom — Context Compression Proxy",
            description="Reverse proxy with real-time compression tracking",
            version="0.3.0",
            lifespan=_lifespan,
        )

        # Add CORS middleware
        if cors_origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Add routes
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up all API routes."""
        # Proxy routes
        self.app.add_api_route(
            "/v1/chat/completions",
            self._handle_request,
            methods=["POST"],
            summary="Proxy request to LLM API",
        )
        self.app.add_api_route(
            "/v1/responses",
            self._handle_request,
            methods=["POST"],
            summary="Proxy OpenAI Responses request",
        )
        self.app.add_api_route(
            "/",
            self._handle_request,
            methods=["POST"],
            summary="Proxy request to LLM API",
        )

        # API routes for dashboard
        self.app.add_api_route("/api/stats", self._get_stats, methods=["GET"])
        self.app.add_api_route("/api/history", self._get_history, methods=["GET"])
        self.app.add_api_route("/api/read-lifecycle", self._get_read_lifecycle, methods=["GET"])
        self.app.add_api_route("/api/ccr", self._get_ccr_stats, methods=["GET"])
        self.app.add_api_route("/api/cache-stats", self._get_cache_stats, methods=["GET"])
        self.app.add_api_route("/livez", self._get_liveness, methods=["GET"])
        self.app.add_api_route("/readyz", self._get_readiness, methods=["GET"])
        self.app.add_api_route("/metrics", self._get_metrics, methods=["GET"])

        # WebSocket for live updates
        self.app.add_api_route("/ws/events", self._websocket_endpoint, methods=["GET"])

        # Serve dashboard at root
        self.app.add_api_route(
            "/",
            self._serve_dashboard,
            methods=["GET"],
            response_class=HTMLResponse,
        )

        # SSE for live event stream
        self.app.add_api_route(
            "/api/events",
            self._sse_endpoint,
            methods=["GET"],
        )
        self.app.add_api_route(
            "/{path:path}",
            self._handle_request,
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
            summary="Byte-faithful upstream passthrough",
        )

    async def _serve_dashboard(self, request: Request) -> HTMLResponse:
        """Serve the dashboard HTML."""
        from .proxy_dashboard import get_dashboard_html
        return HTMLResponse(content=get_dashboard_html())

    async def _handle_request(self, request: Request) -> Response:
        """Compress recognized protocols and transparently forward everything else."""
        route = request.url.path if request.url.path in {
            "/v1/chat/completions", "/v1/responses"
        } else "passthrough"
        started = self._metrics.begin()
        try:
            response = await self._handle_request_inner(request)
        except BaseException:
            self._metrics.finish(request.method, route, 500, started)
            raise
        response.headers.setdefault("x-legroom-request-id", request.state.request_id)
        if isinstance(response, StreamingResponse):
            original_iterator = response.body_iterator

            async def measured_stream() -> AsyncIterator[bytes | str]:
                try:
                    async for chunk in original_iterator:
                        yield chunk
                finally:
                    self._metrics.finish(
                        request.method, route, response.status_code, started
                    )

            response.body_iterator = measured_stream()
        else:
            self._metrics.finish(request.method, route, response.status_code, started)
        return response

    async def _handle_request_inner(self, request: Request) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        try:
            original_body = await request.body()
            path = request.url.path
            body: dict[str, Any] | None = None
            view = None
            body_mutated = False

            if path in {"/v1/chat/completions", "/v1/responses"}:
                body = json.loads(original_body)
                if not isinstance(body, dict):
                    return JSONResponse(
                        {"error": {"type": "invalid_request", "message": "JSON body must be an object"}},
                        status_code=400,
                    )
                view = compression_view(path, body, self.mode)

            if self.compress_context and view is not None and view.messages:
                policy = f"v2:ccr={path == '/v1/chat/completions'}"
                cache_key = self._compression_cache.key(
                    protocol=view.protocol,
                    model=view.model,
                    mode=self.mode,
                    messages=view.messages,
                    policy=policy,
                )
                cached = self._compression_cache.get(cache_key)
                if (
                    cached is not None
                    and cached.ccr_hashes
                    and not self._compression_store.contains_all(cached.ccr_hashes)
                ):
                    self._compression_cache.discard(cache_key)
                    cached = None
                if cached is None:
                    config = CompressConfig(
                        optimize=True,
                        protect_recent=0,
                        ccr_enabled=path == "/v1/chat/completions",
                        read_lifecycle_enabled=True,
                    )
                    async with self._compression_slots:
                        result = await asyncio.to_thread(
                            compress,
                            view.messages,
                            model=view.model,
                            config=config,
                            compression_store=self._compression_store,
                        )
                    ccr_hashes = result.metadata.get("ccr_hashes") if result.metadata else None
                    if ccr_hashes:
                        self._state.record_ccr_store(len(ccr_hashes))
                    for warning in result.warnings:
                        if "failed:" in warning.lower():
                            phase = warning.split(" failed:", 1)[0].lower().replace(" ", "_")
                            self._metrics.record_error(f"phase_{phase}")
                    cached = CachedCompression(
                        messages=result.messages,
                        tokens_before=result.tokens_before,
                        tokens_after=result.tokens_after,
                        transforms=result.transforms_applied,
                        ccr_hashes=tuple(ccr_hashes or ()),
                    )
                    self._compression_cache.put(cache_key, cached)
                    transforms = result.transforms_applied
                    warnings = result.warnings
                else:
                    transforms = [*cached.transforms, "compression_cache_hit"]
                    warnings = []

                body_mutated = view.apply(body, cached.messages)
                self._state.record_request(
                    request_id=request_id,
                    model=view.model,
                    messages_before=len(view.messages),
                    tokens_before=cached.tokens_before,
                    tokens_after=cached.tokens_after,
                    transforms_applied=transforms,
                    warnings=warnings,
                )
                if cached.has_ccr and path == "/v1/chat/completions":
                    tools = body.setdefault("tools", [])
                    names = {
                        tool.get("function", {}).get("name")
                        for tool in tools
                        if isinstance(tool, dict)
                    }
                    if "ccr_retrieve" not in names:
                        tools.append(create_ccr_tool_definition())
                        body_mutated = True

            headers = filter_request_headers(request.headers.raw)
            if self.api_key:
                headers = [(key, value) for key, value in headers if key.lower() != b"authorization"]
                headers.append((b"authorization", f"Bearer {self.api_key}".encode("latin-1")))
            outbound = select_outbound_body(
                body=body or {}, original=original_body, mutated=body_mutated
            )
            client = self.app.state.http_client
            target = self._target_for(request)
            upstream_request = client.build_request(
                request.method, target, content=outbound.content, headers=headers
            )
            upstream = await client.send(upstream_request, stream=True)
            is_stream = bool(body and body.get("stream"))

            if is_stream:
                async def stream_generator() -> AsyncIterator[bytes]:
                    try:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                    finally:
                        await upstream.aclose()

                response = StreamingResponse(
                    stream_generator(), status_code=upstream.status_code, media_type=None
                )
                return _with_raw_headers(
                    response, filter_response_headers(upstream.headers.raw, streaming=True)
                )

            raw_body = b"".join([chunk async for chunk in upstream.aiter_raw()])
            await upstream.aclose()
            if path == "/v1/chat/completions" and body and upstream.status_code == 200 and "tools" in body:
                try:
                    response_document = json.loads(raw_body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    response_document = None
                if isinstance(response_document, dict):
                    response_document, upstream, recalled_body = await self._resolve_ccr_retrieve_loop(
                        client, body, response_document, headers, upstream, target
                    )
                    if recalled_body is not None:
                        raw_body = recalled_body

            response = Response(content=raw_body, status_code=upstream.status_code)
            return _with_raw_headers(
                response, filter_response_headers(upstream.headers.raw, streaming=False)
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(
                {"error": {"type": "invalid_request", "message": "Request body must be valid UTF-8 JSON"}},
                status_code=400,
            )
        except httpx.TimeoutException:
            self._metrics.record_error("upstream_timeout")
            logger.warning("Proxy %s upstream timeout", request_id)
            return JSONResponse(
                {"error": {"type": "upstream_timeout", "message": "Upstream request timed out"}},
                status_code=504,
            )
        except httpx.HTTPError as exc:
            self._metrics.record_error("upstream_transport")
            logger.warning("Proxy %s upstream transport error: %s", request_id, type(exc).__name__)
            return JSONResponse(
                {"error": {"type": "upstream_error", "message": "Upstream request failed"}},
                status_code=502,
            )

    def _target_for(self, request: Request) -> str:
        """Resolve a request path against a base URL or legacy full route URL."""
        target = urlsplit(self.target_url)
        known = {"/v1/chat/completions", "/v1/responses"}
        if request.url.path == target.path:
            path = target.path
        elif target.path in known:
            path = request.url.path
        else:
            path = f"{target.path.rstrip('/')}/{request.url.path.lstrip('/')}"
        return urlunsplit((target.scheme, target.netloc, path, request.url.query, ""))
    async def _resolve_ccr_retrieve_loop(
        self,
        client: httpx.AsyncClient,
        request_body: dict[str, Any],
        response_body: dict[str, Any],
        headers: list[tuple[bytes, bytes]],
        initial_response: httpx.Response,
        target_url: str,
    ) -> tuple[dict[str, Any], httpx.Response, bytes | None]:
        """Resolve any ccr_retrieve tool calls server-side and re-call upstream.

        The proxy — not the client agent harness — owns the CCR store, so it
        must answer ccr_retrieve calls itself: it appends the assistant's
        tool_calls turn plus a synthetic tool result message per call to the
        conversation and re-posts to the upstream API, transparently to the
        caller. Upstream speaks the OpenAI chat-completions wire format (see
        LEGROOM_TARGET_URL / finish_reason handling above), so responses are
        read as `choices[0].message` with `finish_reason == "tool_calls"`,
        not Anthropic's `stop_reason`/`content` blocks. Bounded by
        _CCR_RETRIEVE_MAX_HOPS in case of a pathological loop.
        """
        body = dict(request_body)
        resp_body = response_body
        last_resp: httpx.Response = initial_response
        recalled_body: bytes | None = None

        for _ in range(_CCR_RETRIEVE_MAX_HOPS):
            choices = resp_body.get("choices", [])
            if not choices:
                break
            choice = choices[0]
            if choice.get("finish_reason") != "tool_calls":
                break

            message = choice.get("message", {})
            tool_calls = message.get("tool_calls") or []

            retrieve_calls = [
                tc for tc in tool_calls
                if isinstance(tc, dict) and tc.get("function", {}).get("name") == "ccr_retrieve"
            ]
            if not retrieve_calls:
                break

            tool_messages = []
            for call in retrieve_calls:
                func = call.get("function", {})
                try:
                    args = json.loads(func.get("arguments") or "{}")
                except (ValueError, TypeError):
                    args = {}
                hash_key = args.get("hash", "")
                original = self._compression_store.retrieve(hash_key)
                if original is not None:
                    self._state.record_ccr_retrieve()
                    result_content = original
                else:
                    result_content = f"[No content found for hash={hash_key!r} — it may have expired.]"
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result_content,
                })

            messages = list(body.get("messages", []))
            messages.append(message)
            messages.extend(tool_messages)
            body = {**body, "messages": messages}

            outbound = select_outbound_body(body=body, original=b"", mutated=True)
            retry_headers = [
                (name, value) for name, value in headers if name.lower() != b"accept-encoding"
            ]
            retry_headers.append((b"accept-encoding", b"identity"))
            retry_request = client.build_request(
                "POST", target_url, content=outbound.content, headers=retry_headers
            )
            last_resp = await client.send(retry_request, stream=True)
            recalled_body = b"".join([chunk async for chunk in last_resp.aiter_raw()])
            await last_resp.aclose()
            try:
                resp_body = json.loads(recalled_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                break
            if last_resp.status_code != 200:
                break

        return resp_body, last_resp, recalled_body

    async def _get_stats(self) -> JSONResponse:
        """Return aggregate stats."""
        return JSONResponse(content={
            **self._state.get_stats(),
            "mode": self.mode,
            "inflight_requests": self._metrics.inflight,
            "uptime_seconds": round(__import__("time").time() - self._metrics.started_at, 3),
        })

    async def _get_liveness(self) -> JSONResponse:
        return JSONResponse(content={"service": "legroom-proxy", "alive": True})

    async def _get_readiness(self) -> JSONResponse:
        client = getattr(self.app.state, "http_client", None)
        ready = client is not None and not client.is_closed
        return JSONResponse(
            content={"service": "legroom-proxy", "ready": ready},
            status_code=200 if ready else 503,
        )

    async def _get_metrics(self) -> Response:
        stats = self._state.get_stats()
        cache = self._compression_cache
        operational = self._metrics.render_prometheus()
        compression = (
            "# HELP legroom_compression_tokens_total Compression token totals.\n"
            "# TYPE legroom_compression_tokens_total counter\n"
            f'legroom_compression_tokens_total{{kind="before"}} {stats["total_tokens_before"]}\n'
            f'legroom_compression_tokens_total{{kind="after"}} {stats["total_tokens_after"]}\n'
            f'legroom_compression_tokens_total{{kind="saved"}} {stats["total_tokens_saved"]}\n'
            "# HELP legroom_compression_cache_requests_total Compression cache lookups.\n"
            "# TYPE legroom_compression_cache_requests_total counter\n"
            f'legroom_compression_cache_requests_total{{result="hit"}} {cache.hits}\n'
            f'legroom_compression_cache_requests_total{{result="miss"}} {cache.misses}\n'
        )
        return Response(
            content=operational + compression,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    async def _get_history(self, limit: int = 50, offset: int = 0) -> JSONResponse:
        """Return recent request history."""
        return JSONResponse(content={
            "history": self._state.get_history(limit=limit, offset=offset),
            "total": len(self._state._history),
        })

    async def _get_read_lifecycle(self) -> JSONResponse:
        """Return read lifecycle stats."""
        return JSONResponse(content=self._state.get_read_lifecycle_stats())

    async def _get_ccr_stats(self) -> JSONResponse:
        """Return CCR store stats."""
        return JSONResponse(content={
            "total_stored": self._state.total_ccr_stored,
            "total_retrieved": self._state.total_ccr_retrieved,
        })

    async def _get_cache_stats(self) -> JSONResponse:
        """Return compression-result cache statistics."""
        dedup_cache = self._request_dedup
        return JSONResponse(content={
            "compression_results": {
                "hits": dedup_cache.hits,
                "misses": dedup_cache.misses,
                "size": dedup_cache.size,
                "hit_rate": round(dedup_cache.ratio * 100, 1),
            },
        })

    async def _sse_endpoint(self, request: Request) -> StreamingResponse:
        """Server-Sent Events endpoint for live updates (WebSocket fallback)."""

        async def event_stream() -> AsyncIterator[str]:
            queue = self._state.subscribe()
            try:
                while True:
                    event = await self._state.get_live_event(queue, timeout=15)
                    if event:
                        yield f"data: {json.dumps(event)}\n\n"
                    else:
                        yield ": keepalive\n\n"
            finally:
                self._state.unsubscribe(queue)

        resp = StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )
        # SSE must NOT carry Content-Length — h11 rejects streaming bodies with it
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        resp.headers["X-Accel-Buffering"] = "no"
        # Explicitly drop Content-Length so h11 uses chunked transfer
        del resp.headers["content-length"]
        return resp

    async def _websocket_endpoint(self, websocket: WebSocket) -> None:
        """WebSocket endpoint for live updates."""
        await websocket.accept()
        queue = self._state.subscribe()
        try:
            while True:
                event = await self._state.get_live_event(queue, timeout=30)
                if event:
                    await websocket.send_json(event)
                else:
                    # Send keepalive
                    await websocket.send_json({"type": "keepalive"})
        except WebSocketDisconnect:
            logger.info("Dashboard WebSocket disconnected")
        finally:
            self._state.unsubscribe(queue)

    def get_state(self) -> ProxyState:
        """Return the central state tracker."""
        return self._state
