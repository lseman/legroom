from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_cli_emits_task_verified_evidence_with_explicit_runner(
    tmp_path: Path,
):
    fixture = tmp_path / "trace.json"
    fixture.write_text(
        json.dumps(
            {
                "description": "executable task",
                "messages": [{"role": "user", "content": "important constraint"}],
            }
        ),
        encoding="utf-8",
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "name": "task-suite",
                "fixtures": [
                    {
                        "path": fixture.name,
                        "expected_terms": ["important"],
                        "task": {"expected_action": "edit"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = tmp_path / "runner.py"
    runner.write_text(
        """\
import json
import sys

request = json.load(sys.stdin)
passed = request["task"].get("expected_action") == "edit"
json.dump(
    {"score": 1.0 if passed else 0.0, "passed": passed, "details": {"action": "edit"}},
    sys.stdout,
)
""",
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "benchmarks" / "run_benchmark.py"),
            "--json",
            "--repeat",
            "1",
            "--suite",
            str(suite),
            "--strategies",
            "identity,legroom",
            "--task-runner-command",
            f"{sys.executable} {runner}",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=project_root,
    )

    report = json.loads(completed.stdout)
    assert len(report["points"]) == 2
    assert all(point["quality_evidence"] == "task_verified" for point in report["points"])
    assert all(point["task_passed"] for point in report["points"])
