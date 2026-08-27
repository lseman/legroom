"""Opt-in downstream task replay for compression evaluation.

Suite manifests may carry inert task metadata, but only the caller can select a
runner.  This keeps opening or benchmarking an untrusted suite from executing
commands implicitly.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .evaluation import Fixture, Messages, TaskEvaluation

_MAX_CAPTURE_BYTES = 1_000_000
_MAX_DETAIL_CHARS = 4_000


@dataclass(frozen=True)
class TaskReplayRequest:
    """Serializable input supplied to a downstream task runner."""

    schema_version: int
    fixture: str
    description: str
    original_messages: Messages
    compressed_messages: Messages
    task: dict[str, Any]

    @classmethod
    def from_fixture(cls, fixture: Fixture, output: Messages) -> TaskReplayRequest:
        return cls(
            1,
            fixture.name,
            fixture.description,
            fixture.messages,
            output,
            dict(fixture.task),
        )


@dataclass(frozen=True)
class TaskRunResult:
    """Outcome returned by a runner after executing the downstream task."""

    score: float
    passed: bool
    details: dict[str, Any]

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("task run score must be between 0 and 1")


class TaskRunner(Protocol):
    """Deep seam for model replay, patch execution, or repository tests."""

    def run(self, request: TaskReplayRequest) -> TaskRunResult: ...


class TaskRunnerProtocolError(RuntimeError):
    """The configured runner did not satisfy the JSON protocol."""


class TaskRunnerEvaluator:
    """Adapt a downstream runner to the evaluation suite interface."""

    def __init__(self, runner: TaskRunner) -> None:
        self._runner = runner

    def evaluate(self, fixture: Fixture, output: Messages) -> TaskEvaluation:
        result = self._runner.run(TaskReplayRequest.from_fixture(fixture, output))
        return TaskEvaluation(result.score, result.passed, result.details, "task_verified")


class SubprocessTaskRunner:
    """Run an explicitly selected JSON-over-stdio task harness.

    The command receives one JSON request on stdin and must write one JSON object
    with ``score``, ``passed``, and optional ``details`` fields to stdout.  A
    non-zero process without a valid result is recorded as a failed downstream
    task; a zero-exit process with malformed output is a protocol error.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float = 300.0,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not command or not all(isinstance(part, str) and part for part in command):
            raise ValueError("task runner command must contain non-empty arguments")
        if timeout_seconds <= 0:
            raise ValueError("task runner timeout must be positive")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd
        self.environment = dict(environment) if environment is not None else None

    def run(self, request: TaskReplayRequest) -> TaskRunResult:
        payload = json.dumps(asdict(request), ensure_ascii=False)
        try:
            completed = subprocess.run(
                self.command,
                input=payload,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
                cwd=self.cwd,
                env=self.environment,
            )
        except subprocess.TimeoutExpired:
            return TaskRunResult(
                0.0,
                False,
                {"runner_error": "timeout", "timeout_seconds": self.timeout_seconds},
            )
        except OSError as exc:
            raise TaskRunnerProtocolError(f"could not start task runner: {exc}") from exc

        stdout = completed.stdout
        stderr = completed.stderr[-_MAX_DETAIL_CHARS:]
        if len(stdout.encode("utf-8")) > _MAX_CAPTURE_BYTES:
            raise TaskRunnerProtocolError("task runner output exceeded 1 MB")
        try:
            document = json.loads(stdout)
        except json.JSONDecodeError as exc:
            if completed.returncode != 0:
                return TaskRunResult(
                    0.0,
                    False,
                    {
                        "runner_error": "nonzero_exit",
                        "exit_code": completed.returncode,
                        "stderr": stderr,
                    },
                )
            raise TaskRunnerProtocolError("task runner did not emit valid JSON") from exc

        result = _parse_result(document)
        details = {
            **result.details,
            "runner_exit_code": completed.returncode,
        }
        if stderr:
            details["runner_stderr"] = stderr
        if completed.returncode != 0 and result.passed:
            raise TaskRunnerProtocolError(
                "task runner reported passed=true with a non-zero exit code"
            )
        return TaskRunResult(result.score, result.passed, details)


def _parse_result(document: Any) -> TaskRunResult:
    if not isinstance(document, dict):
        raise TaskRunnerProtocolError("task runner result must be a JSON object")
    score = document.get("score")
    passed = document.get("passed")
    details = document.get("details", {})
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TaskRunnerProtocolError("task runner score must be numeric")
    if not isinstance(passed, bool):
        raise TaskRunnerProtocolError("task runner passed must be boolean")
    if not isinstance(details, dict):
        raise TaskRunnerProtocolError("task runner details must be an object")
    try:
        return TaskRunResult(float(score), passed, details)
    except ValueError as exc:
        raise TaskRunnerProtocolError(str(exc)) from exc
