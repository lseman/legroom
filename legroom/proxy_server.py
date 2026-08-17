"""Proxy server — HTTP reverse proxy for LLM APIs with compression."""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Any
from http import HTTPStatus

try:
    import aiohttp
    from aiohttp import web
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

from .compress import compress
from .config import CompressConfig

logger = logging.getLogger(__name__)


class ProxyServer:
    """HTTP reverse proxy that compresses context before forwarding."""

    def __init__(
        self,
        target_url: str = "https://api.openai.com/v1/chat/completions",
        api_key: str | None = None,
        compress_context: bool = True,
    ) -> None:
        self.target_url = target_url
        self.api_key = api_key
        self.compress_context = compress_context
        self._app: web.Application | None = None

    async def create_app(self) -> web.Application:
        """Create the aiohttp web application."""
        if not _HAS_AIOHTTP:
            raise ImportError("Proxy requires aiohttp: pip install aiohttp")

        self._app = web.Application()
        self._app.router.add_post("/", self._handle_request)
        self._app.router.add_post("/v1/chat/completions", self._handle_request)
        return self._app

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Handle incoming proxy request."""
        try:
            body = await request.json()
            messages = body.get("messages", [])

            if self.compress_context and messages:
                config = CompressConfig(protect_recent=2)
                result = compress(messages, model="gpt-4o", config=config)
                body["messages"] = result.messages

            # Forward to target
            headers = dict(request.headers)
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.target_url,
                    json=body,
                    headers=headers,
                ) as resp:
                    response_body = await resp.read()
                    return web.Response(
                        status=resp.status,
                        body=response_body,
                        content_type="application/json",
                    )
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Run the proxy server."""
        if not _HAS_AIOHTTP:
            raise ImportError("Proxy requires aiohttp")

        app = asyncio.new_event_loop()
        web_runner = web.AppRunner(app)
        app.run_task(host=host, port=port)
