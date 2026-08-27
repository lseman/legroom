import io
import json

import pytest

from legroom.integration.sdk_worker import serve


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global worker state between tests."""
    import legroom.integration.sdk_worker as _sw

    _sw._state = None
    yield
    _sw._state = None


def run_worker(*requests: object) -> list[dict[str, object]]:
    source = io.StringIO("".join(json.dumps(request) + "\n" for request in requests))
    output = io.StringIO()
    serve(source, output)
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_worker_compresses_and_correlates_requests():
    [response] = run_worker(
        {
            "id": "request-1",
            "method": "compress",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "config": {"optimize": False},
        }
    )

    assert response["id"] == "request-1"
    assert response["ok"] is True
    assert response["stats"]["metadata"]["messages"] == [{"role": "user", "content": "hello"}]
    assert response["stats"]["tokens_saved"] == 0


def test_worker_reports_bad_requests_and_keeps_serving():
    responses = run_worker(
        {"id": "bad", "method": "compress", "messages": [], "config": {"wat": True}},
        {"id": "good", "method": "compress", "messages": [], "config": {}},
    )

    assert responses[0]["ok"] is False
    assert "unknown compression config" in responses[0]["error"]
    assert responses[1]["ok"] is True


def test_worker_rejects_unknown_method():
    [response] = run_worker(
        {
            "id": "bad-method",
            "method": "nonexistent",
        }
    )

    assert response["ok"] is False
    assert "unknown method" in response["error"]


def test_worker_accepts_ccr_enabled_in_compress():
    """ccr_enabled is now supported — creates a default store."""
    [response] = run_worker(
        {
            "id": "ccr",
            "method": "compress",
            "messages": [{"role": "user", "content": "hello"}],
            "config": {"ccr_enabled": True},
        }
    )

    assert response["ok"] is True
    # CCR store stats should be present in metadata
    assert "metadata" in response["stats"]
    assert "store_stats" in response["stats"]["metadata"]


def test_worker_compress_with_store():
    """Named store creation and retrieval."""
    # First compress with store
    [compress_resp] = run_worker(
        {
            "id": "compress-1",
            "method": "compress_with_store",
            "store_id": "test-store",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello world"}],
            "config": {},
        }
    )

    assert compress_resp["ok"] is True
    assert compress_resp["stats"]["metadata"]["store_stats"]["entries"] >= 0

    # Then retrieve from the store
    [store_resp] = run_worker(
        {
            "id": "store-1",
            "method": "store_stats",
            "store_id": "test-store",
        }
    )

    assert store_resp["ok"] is True
    assert "entries" in store_resp["stats"]


def test_worker_store_retrieve_not_found():
    """Retrieve from non-existent store returns error."""
    [response] = run_worker(
        {
            "id": "retrieve-1",
            "method": "store_retrieve",
            "store_id": "nonexistent",
            "hash": "abc123",
        }
    )

    assert response["ok"] is False
    assert "store not found" in response["error"]


def test_worker_cache_get_hit_and_miss():
    """Compression result cache lookups."""
    # First request — cache miss
    [resp1] = run_worker(
        {
            "id": "cache-1",
            "method": "compress",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "test"}],
            "config": {"optimize": False},
        }
    )
    assert resp1["ok"] is True

    # Get cache stats
    [stats_resp] = run_worker(
        {
            "id": "stats-1",
            "method": "worker_stats",
        }
    )
    assert stats_resp["ok"] is True
    assert stats_resp["stats"]["cache_misses"] >= 1


def test_worker_cache_key_includes_compression_config():
    responses = run_worker(
        {
            "id": "config-one",
            "method": "compress",
            "messages": [{"role": "user", "content": "same input"}],
            "config": {"optimize": False},
        },
        {
            "id": "config-two",
            "method": "compress",
            "messages": [{"role": "user", "content": "same input"}],
            "config": {"optimize": True},
        },
    )

    assert "compression_cache_hit" not in responses[1]["stats"]["transforms_applied"]


def test_worker_shadow_result_does_not_pollute_live_cache():
    responses = run_worker(
        {
            "id": "shadow",
            "method": "compress",
            "messages": [{"role": "user", "content": "same input"}],
            "config": {"optimize": False, "shadow_mode": True},
        },
        {
            "id": "live",
            "method": "compress",
            "messages": [{"role": "user", "content": "same input"}],
            "config": {"optimize": False},
        },
    )

    assert "compression_cache_hit" not in responses[1]["stats"]["transforms_applied"]


def test_worker_cache_hits_are_recorded_in_metrics_and_history():
    request = {
        "method": "compress",
        "messages": [{"role": "user", "content": "cache me"}],
        "config": {"optimize": False},
    }
    responses = run_worker(
        {"id": "miss", **request},
        {"id": "hit", **request},
        {"id": "stats", "method": "worker_stats"},
        {"id": "history", "method": "worker_history"},
    )

    assert "compression_cache_hit" in responses[1]["stats"]["transforms_applied"]
    assert responses[2]["stats"]["total_requests"] == 2
    assert responses[3]["total"] == 2


def test_worker_calibration_record_and_status():
    """Phase calibration feedback loop."""
    # Record calibration data
    [record_resp] = run_worker(
        {
            "id": "cal-1",
            "method": "calibration_record",
            "phase_reports": [
                {"name": "compress", "status": "applied"},
                {"name": "cross_turn_dedup", "status": "applied"},
            ],
            "quality": 0.95,
        }
    )

    assert record_resp["ok"] is True
    assert "calibration" in record_resp
    assert "snapshots" in record_resp["calibration"]

    # Query calibration status
    [status_resp] = run_worker(
        {
            "id": "cal-2",
            "method": "calibration_status",
        }
    )

    assert status_resp["ok"] is True
    assert "calibration" in status_resp
    assert "disabled_phases" in status_resp["calibration"]


def test_worker_worker_stats():
    """Aggregate worker statistics."""
    # Run a compression first
    run_worker(
        {
            "id": "stats-test-1",
            "method": "compress",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "config": {"optimize": False},
        }
    )

    [stats_resp] = run_worker(
        {
            "id": "stats-query",
            "method": "worker_stats",
        }
    )

    assert stats_resp["ok"] is True
    assert "total_requests" in stats_resp["stats"]
    assert stats_resp["stats"]["total_requests"] >= 1
    assert "cache_hits" in stats_resp["stats"]
    assert "cache_misses" in stats_resp["stats"]


def test_worker_worker_history():
    """Recent request history."""
    # Run some compressions
    for i in range(3):
        run_worker(
            {
                "id": f"history-{i}",
                "method": "compress",
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": f"message {i}"}],
                "config": {"optimize": False},
            }
        )

    [history_resp] = run_worker(
        {
            "id": "history-query",
            "method": "worker_history",
            "limit": 10,
            "offset": 0,
        }
    )

    assert history_resp["ok"] is True
    assert "history" in history_resp
    assert "total" in history_resp
    assert history_resp["total"] >= 3
    assert len(history_resp["history"]) >= 3


def test_worker_shadow_mode():
    """Shadow mode returns original messages but computes stats."""
    [response] = run_worker(
        {
            "id": "shadow-1",
            "method": "compress",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "config": {"optimize": False, "shadow_mode": True},
        }
    )

    assert response["ok"] is True
    # Shadow mode returns original messages in metadata
    assert response["stats"]["metadata"]["messages"] == [{"role": "user", "content": "hello"}]
    assert response["stats"]["tokens_saved"] == 0  # no tokens saved in shadow mode


def test_worker_validation_errors():
    """Request validation rejects malformed inputs."""
    # Missing id
    [response] = run_worker(
        {
            "method": "compress",
            "messages": [],
        }
    )
    assert response["ok"] is False
    assert "id" in response["error"]

    # Missing method
    [response] = run_worker(
        {
            "id": "test",
            "messages": [],
        }
    )
    assert response["ok"] is False
    assert "method" in response["error"]

    # Invalid messages
    [response] = run_worker(
        {
            "id": "test",
            "method": "compress",
            "messages": "not a list",
        }
    )
    assert response["ok"] is False
    assert "messages" in response["error"]
