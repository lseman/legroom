"""CCR — Compression Cache Retrieval module."""

from .compression_store import CompressionStore
from .marker_resolution import create_resolution_prompt, parse_markers, resolve_marker
from .tool_injection import CCRToolInjector, create_ccr_tool_definition, create_system_instructions

__all__ = [
    "CCRToolInjector",
    "CompressionStore",
    "create_ccr_tool_definition",
    "create_resolution_prompt",
    "create_system_instructions",
    "parse_markers",
    "resolve_marker",
]
