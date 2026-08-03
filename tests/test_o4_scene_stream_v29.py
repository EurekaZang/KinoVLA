from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_o4_scene_stream_v29 as audit
from scripts import run_o4_scene_candidate_v29 as runner


def test_v29_config_adapter_changes_only_implementation_schema(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.json"
    base = {
        "candidate_stream": [{"scene_id": "scene_a"}],
        "selection_policy": {},
        "pipeline": {},
        "locked_files": [],
    }
    base_path.write_text(json.dumps(base), encoding="utf-8")
    path = tmp_path / "config.json"
    original = {
        "schema_version": "kinofail.o4-scene-stream-v29-freeze.v1",
        "freeze_status": "frozen_before_any_v29_candidate_generation_or_o4_execution",
        "base_config": {
            "path": str(base_path),
            "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        },
        "candidate_ids": ["scene_a"],
        "new_candidates": [],
        "selection_policy_overrides": {},
        "pipeline_overrides": {
            "candidate_runner": "scripts/run_o4_scene_candidate_v29.py"
        },
        "operator_output_root": "unused",
        "additional_locked_files": [],
        "purpose": "test",
        "evidence_policy": {},
    }
    path.write_text(json.dumps(original), encoding="utf-8")
    adapted = audit._json_v29(path)
    assert adapted["schema_version"] == "kinofail.o4-scene-stream-v28-freeze.v1"
    assert adapted["pipeline"]["candidate_runner"].endswith("v28.py")
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_v29_usd_stage_is_wrapped_in_isaac_setup(monkeypatch) -> None:
    captured = {}

    def fake_original(**kwargs):
        captured.update(kwargs)
        return True, {}

    monkeypatch.setattr(runner, "_original_run_stage", fake_original)
    runner._run_stage_v29(
        name="source_preflight",
        command=["python", "/tmp/preflight.py", "--out", "/tmp/out.json"],
        audit_path=Path("/tmp/out.json"),
        environment={},
    )
    command = captured["command"]
    assert command[:2] == [str(runner.implementation.SHELL), "-lc"]
    assert str(runner.implementation.ISAAC_SETUP) in command
    assert command[-2:] == ["--out", "/tmp/out.json"]


def test_v29_receipt_path_is_versioned(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_write(path, payload):
        captured["path"] = path
        captured["payload"] = payload

    monkeypatch.setattr(runner, "_original_write_receipt", fake_write)
    runner._write_receipt_v29(
        tmp_path / "scene_v28_candidate_receipt.json",
        {"schema_version": "old"},
    )
    assert captured["path"].name == "scene_v29_candidate_receipt.json"
    assert captured["payload"]["schema_version"] == (
        "kinofail.o4-scene-candidate-v29-receipt.v1"
    )
