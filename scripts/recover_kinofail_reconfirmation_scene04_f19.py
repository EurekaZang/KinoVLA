#!/usr/bin/env python3
"""One-shot F19 recovery of scene04 after storage-only T2 postprocess failure."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_kinofail_reconfirmation_scene_pipeline_v4 as pipeline


SCENE = "confirm_v2_production_scene_04"
STATE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration/"
    f"{SCENE}/pipeline_state.json"
)
PRESTATE = STATE.with_name("pipeline_state.f19_pre_recovery.json")
AUDIT = STATE.with_name("f19_storage_postprocess_recovery.json")
PREDICTION_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_prestate(amendment: dict[str, Any]) -> dict[str, Any]:
    correction = amendment["correction"]
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    if list(PREDICTION_ROOT.glob("**/*prediction*")):
        raise RuntimeError("prediction artifact exists before F19 recovery")
    if (
        not STATE.is_file()
        or not PRESTATE.is_file()
        or pipeline.predecessor.base._sha256(STATE)
        != correction["scene04_prestate_sha256"]
        or STATE.read_bytes() != PRESTATE.read_bytes()
    ):
        raise RuntimeError("scene04 failure state changed after F19 seal")
    state = pipeline.predecessor.base._json(STATE)
    paths = pipeline.predecessor.base._scene_paths(SCENE)
    if (
        state.get("status") != "terminal_failure"
        or state.get("error")
        != "RuntimeError: model-blind postprocess failed: t2_features"
        or not paths["physical"].is_dir()
        or (paths["derived"] / "c2_t2/features").exists()
        or (
            pipeline.predecessor.base.CONFLICT_CAPSULE_ROOT
            / SCENE
        ).exists()
        or (
            pipeline.predecessor.base.RECEIPT_ROOT
            / SCENE
            / "completed.json"
        ).exists()
    ):
        raise RuntimeError("scene04 is not eligible for F19 recovery")
    return state


def main() -> int:
    amendment = pipeline._validate_f19()
    state = _validate_prestate(amendment)
    paths = pipeline.predecessor.base._scene_paths(SCENE)
    log_dir = (
        pipeline.predecessor.base.ORCHESTRATION_ROOT / SCENE / "logs"
    )
    started = datetime.now(UTC).isoformat()
    audit: dict[str, Any] = {
        "schema_version": "kinofail.reconfirmation-f19-scene04-recovery.v1",
        "scene_id": SCENE,
        "state": "started",
        "started_utc": started,
        "passed": False,
        "operational_amendment": str(pipeline.AMENDMENT),
        "operational_amendment_sha256": (
            pipeline.predecessor.base._sha256(pipeline.AMENDMENT)
        ),
        "prestate": str(PRESTATE),
        "prestate_sha256": pipeline.predecessor.base._sha256(PRESTATE),
        "physical_acquisition_reexecuted": False,
        "scale_or_t3_postprocess_reexecuted": False,
        "model_or_prediction_loaded": False,
        "scientific_content_changed": False,
        "steps": {},
    }
    _write_json(AUDIT, audit)
    try:
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        t2_command = [
            sys.executable,
            str(pipeline.STORAGE_T2),
            "--corpus",
            str(paths["physical"] / "c2_t2"),
            "--output",
            str(paths["derived"] / "c2_t2/features"),
            "--scene-id",
            SCENE,
            "--batch-size",
            "64",
        ]
        t2 = pipeline._ORIGINAL_RUN(
            "t2_features_f19",
            t2_command,
            log_dir=log_dir,
            environment=environment,
        )
        audit["steps"]["t2_features"] = t2
        _write_json(AUDIT, audit)
        if t2["returncode"] != 0:
            raise RuntimeError("F19 T2 feature recovery failed")

        pipeline.install_storage_overrides()
        capsule = pipeline.predecessor.base._build_conflict_capsule(
            SCENE,
            paths,
            log_dir=log_dir,
        )
        audit["steps"]["conflict_evidence_capsule"] = capsule
        _write_json(AUDIT, audit)
        prune = pipeline.predecessor.base._prune(
            SCENE,
            paths,
            log_dir=log_dir,
        )
        audit["steps"]["prune"] = prune

        predecessor_error = state.pop("error")
        predecessor_failed_utc = state.pop("failed_utc")
        state["schema_version"] = "kinofail.reconfirmation-scene-pipeline.v4"
        state["stages"]["f19_storage_postprocess_recovery"] = audit["steps"]
        state["f19_recovery"] = {
            "audit": str(AUDIT),
            "prestate": str(PRESTATE),
            "predecessor_error": predecessor_error,
            "predecessor_failed_utc": predecessor_failed_utc,
            "physical_acquisition_reexecuted": False,
            "scale_or_t3_postprocess_reexecuted": False,
            "scientific_content_changed": False,
        }
        state["status"] = "complete"
        state["completed_utc"] = datetime.now(UTC).isoformat()
        pipeline.predecessor.base._write_json(STATE, state)

        audit["state"] = "terminal"
        audit["passed"] = True
        audit["completed_utc"] = datetime.now(UTC).isoformat()
        audit["receipt"] = str(
            pipeline.predecessor.base.RECEIPT_ROOT
            / SCENE
            / "completed.json"
        )
        audit["result_dependent_retry_or_selection"] = False
        _write_json(AUDIT, audit)
    except BaseException as error:
        audit["state"] = "terminal_failure"
        audit["failed_utc"] = datetime.now(UTC).isoformat()
        audit["error"] = f"{type(error).__name__}: {error}"
        _write_json(AUDIT, audit)
        raise
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
