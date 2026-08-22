"""Proxy — FastAPI reverse proxy, live state tracking, and dashboard."""

from .proxy_server import LegroomProxy
from .proxy_state import ProxyState, RequestEvent
from .proxy_dashboard import get_dashboard_html

__all__ = [
    "LegroomProxy",
    "ProxyState",
    "RequestEvent",
    "get_dashboard_html",
]
