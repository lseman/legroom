"""Proxy — FastAPI reverse proxy, live state tracking, and dashboard."""

from .proxy_dashboard import get_dashboard_html
from .proxy_server import LegroomProxy
from .proxy_state import ProxyState, RequestEvent

__all__ = [
    "LegroomProxy",
    "ProxyState",
    "RequestEvent",
    "get_dashboard_html",
]
