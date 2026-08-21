"""FastAPI proxy server — HTTP reverse proxy with compression and live stats."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import httpx
import fastapi
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import CompressConfig
from .compress import compress
from .proxy_state import ProxyState
from .tokenizer import count_tokens

logger = logging.getLogger(__name__)


class RequestDedupCache:
    """LRU cache for deduplicating identical proxy requests.

    Keys by (model, compressed message tuple) and stores the full
    compressed message list so repeated requests bypass compression.
    """

    def __init__(self, maxsize: int = 128) -> None:
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def _make_key(self, model: str, messages: list[dict[str, Any]]) -> str:
        """Build a deterministic key from model + message contents."""
        parts = [model]
        for msg in messages:
            content = msg.get("content", "")
            parts.append(f"{msg.get('role', '?')}:{hashlib.sha256(str(content).encode()).hexdigest()[:24]}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]

    def get(self, model: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        key = self._make_key(model, messages)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, model: str, messages: list[dict[str, Any]], result: list[dict[str, Any]]) -> None:
        key = self._make_key(model, messages)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = result

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def ratio(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


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
        self._state = ProxyState(max_history=max_history)
        self._request_dedup = RequestDedupCache()
        self._cors_origins = cors_origins

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

        # WebSocket for live updates
        self.app.add_api_route("/ws/events", self._websocket_endpoint, methods=["GET"])

        # Serve dashboard at root
        from .proxy_dashboard import get_dashboard_html
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

    async def _serve_dashboard(self, request: Request) -> HTMLResponse:
        """Serve the dashboard HTML."""
        from .proxy_dashboard import get_dashboard_html
        return HTMLResponse(content=get_dashboard_html())

    async def _handle_request(self, request: Request) -> Response:
        """Handle incoming proxy request with compression and tracking."""
        request_id = str(uuid.uuid4())[:8]
        start_time = __import__("time").time()

        try:
            body = await request.json()
            messages = body.get("messages", [])
            model = body.get("model", "gpt-4o")
            is_stream = body.get("stream", False)

            # Compress if enabled and there are messages
            if self.compress_context and messages:
                # Proxy-level request dedup: skip compression for identical requests
                cached = self._request_dedup.get(model, messages)
                if cached is not None:
                    body["messages"] = cached
                    # Compute token counts using accurate tiktoken
                    cached_tokens_before = sum(
                        count_tokens(m.get("content", ""), model)
                        for m in messages
                    )
                    cached_tokens_after = sum(
                        count_tokens(m.get("content", ""), model)
                        for m in cached
                    )
                    body["messages"] = cached

                    # Record stats with accurate counts
                    self._state.record_request(
                        request_id=request_id,
                        model=model,
                        messages_before=len(messages),
                        tokens_before=cached_tokens_before,
                        tokens_after=cached_tokens_after,
                        transforms_applied=["request_dedup"],
                        warnings=["cached request"],
                    )
                    logger.info(f"Proxy {request_id}: {model} — cached request (dedup hit)")
                else:
                    config = CompressConfig(
                        optimize=True,
                        protect_recent=2,
                        compress_enabled=True,
                        ccr_enabled=True,
                        read_lifecycle_enabled=True,
                    )
                    result = compress(messages, model=model, config=config)

                    # Update body with compressed messages
                    body["messages"] = result.messages

                    # Record stats
                    tokens_before = result.tokens_before
                    tokens_after = result.tokens_after
                    self._state.record_request(
                        request_id=request_id,
                        model=model,
                        messages_before=len(messages),
                        tokens_before=tokens_before,
                        tokens_after=tokens_after,
                        transforms_applied=result.transforms_applied,
                        warnings=result.warnings,
                    )

                    logger.info(
                        f"Proxy {request_id}: {model} — "
                        f"{tokens_before}->{tokens_after} tokens "
                        f"({result.transforms_applied})"
                    )

                    # Store in request dedup cache
                    self._request_dedup.put(model, messages, result.messages)

            # Forward to target LLM API
            client = self.app.state.http_client
            headers = dict(request.headers)
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Remove hop-by-hop headers
            headers.pop("host", None)
            headers.pop("content-length", None)
            headers.pop("transfer-encoding", None)

            if is_stream:
                # Streaming mode — use streaming HTTP request to forward SSE chunks in real-time
                async def stream_generator():
                    async with client.stream(
                        "POST",
                        self.target_url,
                        json=body,
                        headers=headers,
                    ) as resp:
                        # Forward response headers
                        nonlocal resp_headers
                        resp_headers = {
                            k: v for k, v in resp.headers.items()
                            if k.lower() not in ("transfer-encoding", "connection", "content-length")
                        }
                        # Stream chunks as they arrive
                        async for chunk in resp.aiter_bytes():
                            yield chunk

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                )
            else:
                # Non-streaming mode — collect full response then parse
                resp = await client.post(
                    self.target_url,
                    json=body,
                    headers=headers,
                )

                # Forward response, dropping hop-by-hop headers
                resp_headers = {
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ("transfer-encoding", "connection", "content-length")
                }
                try:
                    body = resp.json()
                except Exception:
                    body = {"error": {"message": resp.text, "type": "proxy_error"}}
                return JSONResponse(
                    content=body,
                    status_code=resp.status_code,
                    headers=resp_headers,
                )

        except Exception as e:
            logger.error(f"Proxy {request_id} error: {e}")
            return JSONResponse(
                content={"error": str(e)},
                status_code=502,
            )
    async def _get_stats(self) -> JSONResponse:
        """Return aggregate stats."""
        return JSONResponse(content=self._state.get_stats())

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
        """Return compression and request dedup cache statistics."""
        dedup_cache = self._request_dedup
        return JSONResponse(content={
            "request_dedup": {
                "hits": dedup_cache.hits,
                "misses": dedup_cache.misses,
                "size": dedup_cache.size,
                "hit_rate": round(dedup_cache.ratio * 100, 1),
            },
        })

    async def _sse_endpoint(self, request: Request) -> StreamingResponse:
        """Server-Sent Events endpoint for live updates (WebSocket fallback)."""

        async def event_stream() -> AsyncIterator[str]:
            while True:
                try:
                    event = await self._state.get_live_events(timeout=15)
                    if event:
                        yield f"data: {json.dumps(event)}\n\n"
                    else:
                        yield ": keepalive\n\n"
                except asyncio.CancelledError:
                    break

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
        try:
            while True:
                event = await self._state.get_live_events(timeout=30)
                if event:
                    await websocket.send_json(event)
                else:
                    # Send keepalive
                    await websocket.send_json({"type": "keepalive"})
        except WebSocketDisconnect:
            logger.info("Dashboard WebSocket disconnected")

    def get_state(self) -> ProxyState:
        """Return the central state tracker."""
        return self._state
