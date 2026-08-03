from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_o4_scene_stream_v28 as audit
from scripts import run_o4_scene_candidate_v28 as runner


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_isaac_command_uses_audited_setup_and_fresh_process() -> None:
    command = runner._isaac_command(Path("/tmp/example.py"), ["--headless"])
    assert command[:2] == [str(runner.SHELL), "-lc"]
    assert command[4] == str(runner.ISAAC_SETUP)
    assert command[5] == str(runner.ISAAC_PYTHON)
    assert command[-2:] == ["/tmp/example.py", "--headless"]


def test_run_stage_requires_zero_return_and_passed_audit(
    tmp_path: Path, monkeypatch
) -> None:
    audit_path = tmp_path / "audit.json"

    def fake_run(*args, **kwargs):
        _write_json(audit_path, {"passed": True})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    passed, record = runner._run_stage(
        name="route",
        command=["fake"],
        audit_path=audit_path,
        environment={},
    )
    assert passed is True
    assert record["audit_sha256"] == _sha256(audit_path)


def test_admitted_receipt_must_bind_full_stage_prefix_and_hashes(
    tmp_path: Path,
) -> None:
    stage_audit = tmp_path / "stage.json"
    stack_v17 = tmp_path / "stack_v17.json"
    receipt_path = tmp_path / "receipt.json"
    _write_json(stage_audit, {"passed": True})
    _write_json(stack_v17, {"passed": True})
    stages = [
        {
            "stage": name,
            "audit": str(stage_audit),
            "audit_sha256": _sha256(stage_audit),
            "passed": True,
        }
        for name in audit.STAGE_ORDER
    ]
    scene = {
        "scene_id": "indoor_office_test",
        "room_type": "Office",
        "source_seed": 11,
        "runtime_seed": 12,
        "material_id": "train_tiles141",
    }
    receipt = {
        "schema_version": "kinofail.o4-scene-candidate-v28-receipt.v1",
        **scene,
        "config_sha256": "abc",
        "sealed": True,
        "o4_outcomes_observed": False,
        "counts_as_a0_a7_evidence": False,
        "fully_admitted": True,
        "passed": True,
        "terminal_stage": "stack_v17",
        "stages": stages,
    }
    _write_json(receipt_path, receipt)
    passed, checks = audit._receipt_integrity(
        config_hash="abc",
        scene=scene,
        paths={"receipt": receipt_path, "stack_v17": stack_v17},
    )
    assert passed is True
    assert all(checks.values())

    receipt["stages"][-1]["audit_sha256"] = "stale"
    _write_json(receipt_path, receipt)
    passed, checks = audit._receipt_integrity(
        config_hash="abc",
        scene=scene,
        paths={"receipt": receipt_path, "stack_v17": stack_v17},
    )
    assert passed is False
    assert checks["recorded_artifact_hashes_match"] is False
