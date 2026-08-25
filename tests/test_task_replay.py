from __future__ import annotations

import json
import sys

import pytest

from legroom.evaluation import Fixture
from legroom.task_replay import (
    SubprocessTaskRunner,
    TaskReplayRequest,
    TaskRunnerEvaluator,
    TaskRunnerProtocolError,
)


def _fixture() -> Fixture:
    return Fixture(
        "patch_task",
        "apply a patch and run tests",
        [{"role": "user", "content": "preserve Python 3.10"}],
        (),
        task={"repository": "sample", "test_command": ["pytest", "-q"]},
    )


def test_subprocess_runner_round_trips_context_and_returns_verified_evidence():
    script = """
import json, sys
request = json.load(sys.stdin)
ok = (
    request["schema_version"] == 1
    and request["fixture"] == "patch_task"
    and request["compressed_messages"][-1]["content"] == "short context"
    and request["task"]["test_command"] == ["pytest", "-q"]
)
json.dump({"score": 1.0 if ok else 0.0, "passed": ok, "details": {"tests_passed": 4}}, sys.stdout)
"""
    runner = SubprocessTaskRunner((sys.executable, "-c", script), timeout_seconds=5)
    evaluator = TaskRunnerEvaluator(runner)

    result = evaluator.evaluate(
        _fixture(), [{"role": "user", "content": "short context"}]
    )

    assert result.passed
    assert result.score == 1.0
    assert result.evidence_kind == "task_verified"
    assert result.details["tests_passed"] == 4
    assert result.details["runner_exit_code"] == 0


def test_subprocess_runner_records_nonzero_process_as_failed_task():
    runner = SubprocessTaskRunner(
        (sys.executable, "-c", "import sys; print('test failed', file=sys.stderr); sys.exit(2)"),
        timeout_seconds=5,
    )
    result = runner.run(
        TaskReplayRequest.from_fixture(_fixture(), _fixture().messages)
    )
    assert not result.passed
    assert result.score == 0.0
    assert result.details["exit_code"] == 2
    assert "test failed" in result.details["stderr"]


def test_subprocess_runner_rejects_malformed_success_output():
    runner = SubprocessTaskRunner(
        (sys.executable, "-c", "print('not json')"), timeout_seconds=5
    )
    with pytest.raises(TaskRunnerProtocolError, match="valid JSON"):
        runner.run(TaskReplayRequest.from_fixture(_fixture(), _fixture().messages))


def test_subprocess_runner_rejects_claimed_success_on_nonzero_exit():
    document = json.dumps({"score": 1.0, "passed": True, "details": {}})
    runner = SubprocessTaskRunner(
        (sys.executable, "-c", f"import sys; print({document!r}); sys.exit(1)"),
        timeout_seconds=5,
    )
    with pytest.raises(TaskRunnerProtocolError, match="non-zero"):
        runner.run(TaskReplayRequest.from_fixture(_fixture(), _fixture().messages))


def test_subprocess_runner_timeout_is_a_failed_task():
    runner = SubprocessTaskRunner(
        (sys.executable, "-c", "import time; time.sleep(1)"),
        timeout_seconds=0.01,
    )
    result = runner.run(
        TaskReplayRequest.from_fixture(_fixture(), _fixture().messages)
    )
    assert not result.passed
    assert result.details["runner_error"] == "timeout"
