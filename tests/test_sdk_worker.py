import io
import json

from legroom.sdk_worker import serve


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
    assert response["messages"] == [{"role": "user", "content": "hello"}]
    assert response["stats"]["tokens_saved"] == 0


def test_worker_reports_bad_requests_and_keeps_serving():
    responses = run_worker(
        {"id": "bad", "method": "compress", "messages": [], "config": {"wat": True}},
        {"id": "good", "method": "compress", "messages": [], "config": {}},
    )

    assert responses[0]["ok"] is False
    assert "unknown compression config" in responses[0]["error"]
    assert responses[1]["ok"] is True


def test_worker_rejects_ccr_until_retrieval_lifecycle_is_supported():
    [response] = run_worker(
        {
            "id": "ccr",
            "method": "compress",
            "messages": [],
            "config": {"ccr_enabled": True},
        }
    )

    assert response["ok"] is False
    assert "ccr_enabled is not supported" in response["error"]
