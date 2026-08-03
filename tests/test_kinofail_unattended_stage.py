from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.kinofail_unattended_stage import artifacts_pass, run_write_once_stage


def test_artifacts_pass_requires_declared_boolean(tmp_path: Path) -> None:
    artifact = tmp_path / "audit.json"
    artifact.write_text(json.dumps({"passed": True}) + "\n")
    specification = [{"path": str(artifact), "key": "passed", "allowed": [True]}]
    assert artifacts_pass(specification) == (True, [])
    artifact.write_text(json.dumps({"passed": False}) + "\n")
    passed, issues = artifacts_pass(specification)
    assert not passed
    assert issues and issues[0].startswith("unexpected_value:")


def test_completed_stage_reconciles_without_second_launch(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    artifact = tmp_path / "artifact.json"
    launches = tmp_path / "launches.txt"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import json; "
            f"p=Path({str(launches)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
            f"Path({str(artifact)!r}).write_text(json.dumps({{'passed': True}}))"
        ),
    ]
    observed_states: list[tuple[str, str]] = []

    def write_state(stage: str, state: str, **_: object) -> None:
        observed_states.append((stage, state))

    arguments = {
        "root": tmp_path,
        "state_root": state_root,
        "name": "one_shot",
        "command": command,
        "environment": {},
        "write_state": write_state,
        "artifacts": [{"path": str(artifact), "key": "passed", "allowed": [True]}],
    }
    assert run_write_once_stage(**arguments) == 0
    assert run_write_once_stage(**arguments) == 0
    assert launches.read_text() == "x"
    assert ("one_shot", "completed_reconciled") in observed_states


def test_partial_unrecorded_output_blocks_fresh_launch(tmp_path: Path) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()

    with pytest.raises(RuntimeError, match="partial write-once output"):
        run_write_once_stage(
            root=tmp_path,
            state_root=tmp_path / "state",
            name="blind_prediction",
            command=[sys.executable, "-c", "raise SystemExit(0)"],
            environment={},
            write_state=lambda *_args, **_kwargs: None,
            artifacts=[{"path": str(tmp_path / "missing.json")}],
            forbid_fresh_paths=(partial,),
        )


def test_disallowed_reaped_exit_cannot_be_reconciled_by_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    arguments = {
        "root": tmp_path,
        "state_root": tmp_path / "state",
        "name": "failed_stage",
        "command": [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import json; "
                f"Path({str(artifact)!r}).write_text(json.dumps({{'passed': True}})); "
                "raise SystemExit(1)"
            ),
        ],
        "environment": {},
        "write_state": lambda *_args, **_kwargs: None,
        "artifacts": [{"path": str(artifact), "key": "passed", "allowed": [True]}],
    }
    with pytest.raises(RuntimeError, match="returncode=1"):
        run_write_once_stage(**arguments)
    with pytest.raises(RuntimeError, match="returncode=1"):
        run_write_once_stage(**arguments)
