"""Persistent JSON-lines SDK worker for embedding Legroom in other runtimes."""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from typing import Any, TextIO

from .compress import compress
from .config import CompressConfig

_CONFIG_FIELDS = {field.name for field in fields(CompressConfig)}


def _response(request: object) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object")

    request_id = request.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request.id must be a non-empty string")
    if request.get("method") != "compress":
        raise ValueError("request.method must be 'compress'")

    messages = request.get("messages")
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise ValueError("request.messages must be an array of objects")
    model = request.get("model", "gpt-4o")
    if not isinstance(model, str) or not model:
        raise ValueError("request.model must be a non-empty string")

    raw_config = request.get("config", {})
    if not isinstance(raw_config, dict):
        raise TypeError("request.config must be an object")
    unknown = sorted(set(raw_config) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"unknown compression config field(s): {', '.join(unknown)}")
    if raw_config.get("ccr_enabled") is True:
        raise ValueError("ccr_enabled is not supported by the compression-only SDK worker")

    compression_config = {"ccr_enabled": False, **raw_config}
    result = compress(messages, model=model, config=CompressConfig(**compression_config))
    return {
        "id": request_id,
        "ok": True,
        "messages": result.messages,
        "stats": {
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "tokens_saved": result.tokens_saved,
            "transforms_applied": result.transforms_applied,
            "warnings": result.warnings,
        },
    }


def serve(input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout) -> None:
    """Serve compression requests until stdin reaches EOF."""
    for line in input_stream:
        if not line.strip():
            continue
        request_id: object = None
        try:
            request = json.loads(line)
            if isinstance(request, dict):
                request_id = request.get("id")
            response = _response(request)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            response = {"id": request_id, "ok": False, "error": str(error)}
        # Deliberate process boundary: report compression failures without killing
        # a long-lived worker that may successfully serve the next request.
        except Exception as error:  # noqa: BLE001
            response = {"id": request_id, "ok": False, "error": f"compression failed: {error}"}
        output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
        output_stream.flush()


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
