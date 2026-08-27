"""Persistent JSON-lines SDK worker for embedding Legroom in other runtimes.

Protocol v2 — extends the original compression-only worker with:
  - CCR store (compress_with_store, store_retrieve, store_stats)
  - Compression result cache (cache_get)
  - Phase calibration (calibration_record, calibration_status)
  - Worker observability (worker_stats, worker_history)

All new methods are opt-in via the ``method`` field. The original
``compress`` method continues to work exactly as before (with the
``ccr_enabled`` restriction lifted — the worker now manages its own
stores).
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from typing import Any, TextIO

from ..ccr.compression_store import CompressionStore
from ..proxy.compression_cache import CachedCompression, CompressionResultCache
from ..proxy.observability import ProxyMetrics
from ..runtime.compress import compress
from ..runtime.config import CompressConfig, CompressResult
from .calibration import CalibrationConfig as _CalibrationConfig
from .calibration import CalibrationController

# ── Config fields ────────────────────────────────────────────────────────────

_CONFIG_FIELDS = {field.name for field in fields(CompressConfig)}

# Extra fields accepted by the SDK worker protocol but not part of
# CompressConfig.  They control worker-level behaviour.
_SDK_EXTRA_FIELDS = frozenset(
    {
        "shadow_mode",
        "disabled_phases",
        "calibration",
    }
)


# ── Worker state ─────────────────────────────────────────────────────────────


@dataclass
class RequestEvent:
    """Record of a single compression request."""

    request_id: str
    timestamp: float
    model: str
    messages_before: int
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    transforms_applied: list[str]
    warnings: list[str]


class LegroomWorkerState:
    """Mutable state shared across all JSONL requests in this process."""

    def __init__(self) -> None:
        self.stores: dict[str, CompressionStore] = {}
        self.cache = CompressionResultCache(maxsize=256, ttl_seconds=300.0)
        self.calibration = CalibrationController()
        self.metrics = ProxyMetrics()
        self._history: deque[RequestEvent] = deque(maxlen=1000)
        self.started_at: float = time.time()

    def _store(self, store_id: str) -> CompressionStore:
        if store_id not in self.stores:
            self.stores[store_id] = CompressionStore(max_entries=1000)
        return self.stores[store_id]

    def record_request(
        self,
        request_id: str,
        model: str,
        messages_before: int,
        result: CompressResult,
    ) -> RequestEvent:
        tokens_saved = result.tokens_before - result.tokens_after
        event = RequestEvent(
            request_id=request_id,
            timestamp=time.time(),
            model=model,
            messages_before=messages_before,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            tokens_saved=tokens_saved,
            transforms_applied=list(result.transforms_applied),
            warnings=list(result.warnings),
        )
        self._history.append(event)
        self.metrics.total_requests += 1
        self.metrics.total_tokens_before += result.tokens_before
        self.metrics.total_tokens_after += result.tokens_after
        self.metrics.total_tokens_saved += tokens_saved
        for transform in result.transforms_applied:
            self.metrics.strategy_counts[transform] += 1
        return event


# ── Shared state ─────────────────────────────────────────────────────────────

_state: LegroomWorkerState | None = None


def _ensure_state() -> LegroomWorkerState:
    global _state
    if _state is None:
        _state = LegroomWorkerState()
    return _state


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_compression_config(
    raw_config: dict[str, Any],
) -> CompressConfig:
    """Merge SDK extra fields into a CompressConfig.

    Extra fields that are not part of CompressConfig are stripped before
    construction, then re-applied as ``setattr`` so the pipeline can see
    them (e.g. ``shadow_mode``).
    """
    known = {k: v for k, v in raw_config.items() if k in _CONFIG_FIELDS}
    cfg = CompressConfig(**known)

    # Apply SDK-specific overrides via setattr so the pipeline sees them.
    if raw_config.get("shadow_mode"):
        cfg.shadow_mode = True  # type: ignore[attr-defined]
    disabled = raw_config.get("disabled_phases")
    if isinstance(disabled, list):
        cfg.disabled_phases = tuple(disabled)

    return cfg


def _build_compression_config_with_store(
    raw_config: dict[str, Any],
    store: CompressionStore,
) -> CompressConfig:
    cfg = _build_compression_config(raw_config)
    # Ensure CCR is enabled when a store is provided.
    cfg.ccr_enabled = True
    return cfg


def _compression_cache_policy(raw_config: dict[str, Any], *, store_id: str | None = None) -> str:
    """Return a stable cache policy covering every compression-affecting option."""
    compression_config = {
        key: value for key, value in raw_config.items() if key not in {"shadow_mode", "calibration"}
    }
    policy: dict[str, Any] = {"version": 2, "config": compression_config}
    if store_id is not None:
        policy["store_id"] = store_id
    return json.dumps(policy, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _response_for_compress(
    request_id: str,
    result: CompressResult,
    *,
    store: CompressionStore | None = None,
    shadow_mode: bool = False,
) -> dict[str, Any]:
    """Build a v2 compress response with metadata."""
    stats: dict[str, Any] = {
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_saved,
        "transforms_applied": result.transforms_applied,
        "warnings": result.warnings,
    }

    metadata: dict[str, Any] = {}

    # Always include messages in metadata — compressed in normal mode,
    # original in shadow mode.  (They used to live at the top level of
    # the response in v1; v2 moves them into stats.metadata.)
    metadata["messages"] = result.messages

    # CCR hashes
    ccr_hashes = result.metadata.get("ccr_hashes", [])
    if ccr_hashes:
        metadata["ccr_hashes"] = ccr_hashes

    # Phase reports
    phase_reports = result.metadata.get("phase_reports", [])
    if phase_reports:
        metadata["phase_reports"] = phase_reports

    # Salience scores
    if "salience_scores_before" in result.metadata:
        metadata["salience_scores_before"] = result.metadata["salience_scores_before"]
    if "salience_scores_after" in result.metadata:
        metadata["salience_scores_after"] = result.metadata["salience_scores_after"]

    # Store stats when CCR was used
    if store is not None:
        store_stats = store.get_stats()
        metadata["store_stats"] = store_stats

    if metadata:
        stats["metadata"] = metadata

    return {
        "id": request_id,
        "ok": True,
        "stats": stats,
    }


# ── Request handlers ─────────────────────────────────────────────────────────


def _handle_compress(request: dict[str, Any]) -> dict[str, Any]:
    """Handle a compression request (v1 or v2)."""
    request_id = request.get("id", "")
    messages = request.get("messages")
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise ValueError("request.messages must be an array of objects")

    model = request.get("model", "gpt-4o")
    if not isinstance(model, str) or not model:
        raise ValueError("request.model must be a non-empty string")

    raw_config = request.get("config", {})
    if not isinstance(raw_config, dict):
        raise TypeError("request.config must be an object")

    # Validate known fields
    unknown_sdk = sorted(set(raw_config) - _CONFIG_FIELDS - _SDK_EXTRA_FIELDS)
    if unknown_sdk:
        raise ValueError(f"unknown compression config field(s): {', '.join(unknown_sdk)}")

    cfg = _build_compression_config(raw_config)
    shadow_mode = raw_config.get("shadow_mode", False)

    # Calibration config
    calib_cfg = raw_config.get("calibration")
    if isinstance(calib_cfg, dict):
        calib = _CalibrationConfig(**calib_cfg)
        _ensure_state().calibration = CalibrationController(calib)

    # CCR store
    store: CompressionStore | None = None
    if cfg.ccr_enabled:
        store = _ensure_state()._store("_default")
        cfg.ccr_enabled = True  # ensure it stays True

    # Compression result cache — check for hit first
    cache = _ensure_state().cache
    cache_key = cache.key(
        protocol="sdk",
        model=model,
        mode="token",
        messages=messages,
        policy=_compression_cache_policy(raw_config),
    )
    cached = cache.get(cache_key)

    if cached is not None and not shadow_mode:
        metadata: dict[str, Any] = {}
        if cached.ccr_hashes:
            metadata["ccr_hashes"] = list(cached.ccr_hashes)
        cached_result = CompressResult(
            messages=cached.messages,
            tokens_before=cached.tokens_before,
            tokens_after=cached.tokens_after,
            tokens_saved=cached.tokens_before - cached.tokens_after,
            transforms_applied=[*cached.transforms, "compression_cache_hit"],
            metadata=metadata,
        )
        _ensure_state().record_request(
            request_id=request_id,
            model=model,
            messages_before=len(messages),
            result=cached_result,
        )
        return _response_for_compress(request_id, cached_result, store=store)

    # Run compression
    result = compress(messages, model=model, config=cfg)

    # Shadow mode: return originals but still compute stats
    if shadow_mode:
        result = CompressResult(
            messages=messages,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            tokens_saved=result.tokens_saved,
            transforms_applied=result.transforms_applied,
            warnings=result.warnings,
            metadata=result.metadata,
        )

    # Cache the result
    if cached is None and not shadow_mode:
        cache.put(
            cache_key,
            CachedCompression(
                messages=result.messages,
                tokens_before=result.tokens_before,
                tokens_after=result.tokens_after,
                transforms=list(result.transforms_applied),
                ccr_hashes=tuple(result.metadata.get("ccr_hashes", [])),
            ),
        )

    # Record calibration data if available
    phase_reports = result.metadata.get("phase_reports", [])
    if phase_reports:
        _ensure_state().calibration.record_reports(phase_reports)

    # Record request event
    _ensure_state().record_request(
        request_id=request_id,
        model=model,
        messages_before=len(messages),
        result=result,
    )

    return _response_for_compress(request_id, result, store=store, shadow_mode=shadow_mode)


def _handle_compress_with_store(request: dict[str, Any]) -> dict[str, Any]:
    """Handle compress with a named CCR store."""
    request_id = request.get("id", "")
    store_id = request.get("store_id")
    if not isinstance(store_id, str) or not store_id:
        raise ValueError("request.store_id must be a non-empty string")

    messages = request.get("messages")
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise ValueError("request.messages must be an array of objects")

    model = request.get("model", "gpt-4o")
    if not isinstance(model, str) or not model:
        raise ValueError("request.model must be a non-empty string")

    raw_config = request.get("config", {})
    if not isinstance(raw_config, dict):
        raise TypeError("request.config must be an object")

    cfg = _build_compression_config(raw_config)
    store = _ensure_state()._store(store_id)
    cfg = _build_compression_config_with_store(raw_config, store)

    result = compress(messages, model=model, config=cfg, compression_store=store)

    # Cache the result
    cache = _ensure_state().cache
    cache_key = cache.key(
        protocol="sdk",
        model=model,
        mode="token",
        messages=messages,
        policy=_compression_cache_policy(raw_config, store_id=store_id),
    )
    cache.put(
        cache_key,
        CachedCompression(
            messages=result.messages,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            transforms=list(result.transforms_applied),
            ccr_hashes=tuple(result.metadata.get("ccr_hashes", [])),
        ),
    )

    # Record calibration data
    phase_reports = result.metadata.get("phase_reports", [])
    if phase_reports:
        _ensure_state().calibration.record_reports(phase_reports)

    # Record request event
    _ensure_state().record_request(
        request_id=request_id,
        model=model,
        messages_before=len(messages),
        result=result,
    )

    return _response_for_compress(request_id, result, store=store, shadow_mode=False)


def _handle_store_retrieve(request: dict[str, Any]) -> dict[str, Any]:
    """Retrieve original content from a CCR store."""
    request_id = request.get("id", "")
    store_id = request.get("store_id")
    if not isinstance(store_id, str) or not store_id:
        raise ValueError("request.store_id must be a non-empty string")

    hash_key = request.get("hash")
    if not isinstance(hash_key, str) or not hash_key:
        raise ValueError("request.hash must be a non-empty string")

    store = _ensure_state().stores.get(store_id)
    if store is None:
        return {"id": request_id, "ok": False, "error": f"store not found: {store_id}"}

    content = store.retrieve(hash_key)
    if content is None:
        return {"id": request_id, "ok": False, "error": f"hash not found: {hash_key}"}

    return {"id": request_id, "ok": True, "content": content}


def _handle_store_stats(request: dict[str, Any]) -> dict[str, Any]:
    """Return CCR store statistics."""
    request_id = request.get("id", "")
    store_id = request.get("store_id")
    if not isinstance(store_id, str) or not store_id:
        raise ValueError("request.store_id must be a non-empty string")

    store = _ensure_state().stores.get(store_id)
    if store is None:
        return {"id": request_id, "ok": False, "error": f"store not found: {store_id}"}

    return {
        "id": request_id,
        "ok": True,
        "stats": store.get_stats(),
    }


def _handle_cache_get(request: dict[str, Any]) -> dict[str, Any]:
    """Query the compression result cache."""
    request_id = request.get("id", "")
    key = request.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("request.key must be a non-empty string")

    cached = _ensure_state().cache.get(key)
    if cached is None:
        return {"id": request_id, "ok": True, "hit": False}

    return {
        "id": request_id,
        "ok": True,
        "hit": True,
        "messages": cached.messages,
        "stats": {
            "tokens_before": cached.tokens_before,
            "tokens_after": cached.tokens_after,
            "tokens_saved": cached.tokens_before - cached.tokens_after,
            "transforms_applied": cached.transforms,
        },
    }


def _handle_calibration_record(request: dict[str, Any]) -> dict[str, Any]:
    """Record quality feedback for calibration."""
    request_id = request.get("id", "")
    phase_reports = request.get("phase_reports")
    if not isinstance(phase_reports, list):
        raise ValueError("request.phase_reports must be an array")

    quality = request.get("quality")
    if quality is not None:
        if not isinstance(quality, (int, float)) or isinstance(quality, bool):
            raise ValueError("request.quality must be a number between 0 and 1")
        if not (0 <= quality <= 1):
            raise ValueError("request.quality must be between 0 and 1")

    _ensure_state().calibration.record_reports(phase_reports, quality=quality)

    return {
        "id": request_id,
        "ok": True,
        "calibration": {
            "disabled_phases": list(_ensure_state().calibration.disabled_phases),
            "snapshots": [asdict(s) for s in _ensure_state().calibration.snapshots()],
        },
    }


def _handle_calibration_status(request: dict[str, Any]) -> dict[str, Any]:
    """Query current calibration state."""
    request_id = request.get("id", "")
    return {
        "id": request_id,
        "ok": True,
        "calibration": {
            "disabled_phases": list(_ensure_state().calibration.disabled_phases),
            "snapshots": [asdict(s) for s in _ensure_state().calibration.snapshots()],
        },
    }


def _handle_worker_stats(request: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate worker statistics."""
    request_id = request.get("id", "")
    state = _ensure_state()
    metrics = state.metrics

    total_requests = metrics.total_requests
    if total_requests == 0 or metrics.total_tokens_before == 0:
        compression_ratio = 0.0
    else:
        compression_ratio = round(
            (1.0 - metrics.total_tokens_after / metrics.total_tokens_before) * 100, 1
        )

    return {
        "id": request_id,
        "ok": True,
        "stats": {
            "total_requests": total_requests,
            "total_tokens_before": metrics.total_tokens_before,
            "total_tokens_after": metrics.total_tokens_after,
            "total_tokens_saved": metrics.total_tokens_saved,
            "compression_ratio": compression_ratio,
            "strategy_counts": dict(metrics.strategy_counts),
            "cache_hits": state.cache.hits,
            "cache_misses": state.cache.misses,
            "uptime_seconds": round(time.time() - state.started_at, 3),
        },
    }


def _handle_worker_history(request: dict[str, Any]) -> dict[str, Any]:
    """Return recent request history."""
    request_id = request.get("id", "")
    limit = request.get("limit", 50)
    offset = request.get("offset", 0)

    if not isinstance(limit, int) or limit < 0:
        limit = 50
    if not isinstance(offset, int) or offset < 0:
        offset = 0

    history = list(_ensure_state()._history)
    total = len(history)

    # Most recent first
    history = list(reversed(history))
    entries = [asdict(e) for e in history[offset : offset + limit]]

    return {
        "id": request_id,
        "ok": True,
        "history": entries,
        "total": total,
    }


# ── Dispatch table ───────────────────────────────────────────────────────────

_METHOD_HANDLERS: dict[str, Any] = {
    "compress": _handle_compress,
    "compress_with_store": _handle_compress_with_store,
    "store_retrieve": _handle_store_retrieve,
    "store_stats": _handle_store_stats,
    "cache_get": _handle_cache_get,
    "calibration_record": _handle_calibration_record,
    "calibration_status": _handle_calibration_status,
    "worker_stats": _handle_worker_stats,
    "worker_history": _handle_worker_history,
}


def _response(request: object) -> dict[str, Any]:
    """Parse and dispatch a single JSONL request."""
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object")

    request_id = request.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request.id must be a non-empty string")

    method = request.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("request.method must be a non-empty string")

    handler = _METHOD_HANDLERS.get(method)
    if handler is None:
        raise ValueError(
            f"unknown method: {method!r}. Supported: {', '.join(sorted(_METHOD_HANDLERS))}"
        )

    return handler(request)


# ── Serve loop ───────────────────────────────────────────────────────────────


def serve(input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout) -> None:
    """Serve requests until stdin reaches EOF."""
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
        except Exception as error:  # noqa: BLE001
            response = {"id": request_id, "ok": False, "error": f"worker error: {error}"}
        output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
        output_stream.flush()


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
