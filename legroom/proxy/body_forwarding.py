"""Request-body selection for byte-faithful proxy forwarding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OutboundBody:
    """Bytes to forward and whether Legroom changed the request document."""

    content: bytes
    mutated: bool


def serialize_canonical(body: dict[str, Any]) -> bytes:
    """Serialize a mutated JSON body once, compactly and as literal UTF-8."""
    return json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def select_outbound_body(
    *,
    body: dict[str, Any],
    original: bytes,
    mutated: bool,
) -> OutboundBody:
    """Keep client bytes unless a transform actually changed the document."""
    if not mutated:
        return OutboundBody(content=original, mutated=False)
    return OutboundBody(content=serialize_canonical(body), mutated=True)
