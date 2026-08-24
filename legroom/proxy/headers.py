"""HTTP header filtering for a transparent reverse proxy."""

from __future__ import annotations

from collections.abc import Iterable

_HOP_BY_HOP = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}


def _connection_tokens(headers: Iterable[tuple[bytes, bytes]]) -> set[bytes]:
    tokens: set[bytes] = set()
    for name, value in headers:
        if name.lower() == b"connection":
            tokens.update(part.strip().lower() for part in value.split(b",") if part.strip())
    return tokens


def filter_request_headers(
    headers: Iterable[tuple[bytes, bytes]],
) -> list[tuple[bytes, bytes]]:
    """Remove hop-by-hop and body-length headers before forwarding."""
    materialized = list(headers)
    excluded = _HOP_BY_HOP | _connection_tokens(materialized) | {b"host", b"content-length"}
    return [(name, value) for name, value in materialized if name.lower() not in excluded]


def filter_response_headers(
    headers: Iterable[tuple[bytes, bytes]],
    *,
    streaming: bool,
) -> list[tuple[bytes, bytes]]:
    """Remove hop-by-hop headers while retaining duplicate end-to-end fields."""
    materialized = list(headers)
    excluded = _HOP_BY_HOP | _connection_tokens(materialized)
    if streaming:
        excluded.add(b"content-length")
    return [(name, value) for name, value in materialized if name.lower() not in excluded]
