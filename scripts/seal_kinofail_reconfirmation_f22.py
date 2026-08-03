#!/usr/bin/env python3
"""Seal the F22 orphan-wrapper recovery before resuming scene05."""

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

from scripts import recover_kinofail_reconfirmation_scene05_f21 as f21_recovery
from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as runner
from scripts import seal_kinofail_reconfirmation_f21 as f21_sealer


F22 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f22_orphan_wrapper_amendment1/"
    "amendment_manifest.json"
)
RESUMER = (
    ROOT / "scripts/resume_kinofail_reconfirmation_scene05_f22.py"
)


def _topology() -> dict[str, list[dict[str, Any]]]:
    return {
        "all_scenes_v4": f21_recovery._exact_token_processes(
            "run_kinofail_reconfirmation_all_scene_pipelines_v4.py"
        ),
        "scene_pipelines_v4": f21_recovery._exact_token_processes(
            "run_kinofail_reconfirmation_scene_pipeline_v4.py",
            "--scene",
            f21_sealer.SCENE,
        ),
        "source_t2_collectors": f21_recovery._exact_token_processes(
            "isaac_collect_kinofail_confirmatory_t2_v1.py",
            "--scene",
            f21_sealer.SCENE,
        ),
        "orphan_slotted_t2": f21_recovery._exact_token_processes(
            "run_kinofail_reconfirmation_slotted_t2_v1.py",
            "--scene",
            f21_sealer.SCENE,
        ),
        "failed_f21_recovery": f21_recovery._exact_token_processes(
            "recover_kinofail_reconfirmation_scene05_f21.py"
        ),
    }


def main() -> int:
    f21 = runner._validate_f21()
    if F22.exists():
        raise FileExistsError(F22)
    for path in (
        RESUMER,
        Path(__file__).resolve(),
        f21_recovery.AUDIT_PATH,
        runner.F21,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if f21_recovery.RECOVERY_LOG.exists():
        raise RuntimeError("unexpected F21 T2 recovery log before F22")
    if runner._prediction_files():
        raise RuntimeError("prediction or score exists before F22 seal")

    audit = runner.read_json(f21_recovery.AUDIT_PATH)
    state = f21_sealer.incident_case_state()
    _, inventory_sha256 = f21_sealer.authenticated_inventory(state)
    expected = f21["incident"]
    if (
        audit.get("schema_version")
        != "kinofail.reconfirmation-f21-scene05-recovery.v1"
        or audit.get("state") != "started"
        or audit.get("t2_recovery_state") != "started"
        or audit.get("existing_completed_case_reexecuted") is not False
        or state["sealed_manifests"] != expected["sealed_manifests"]
        or state["empty_zero_observation_case_ids"]
        != expected["empty_zero_observation_case_ids"]
        or state["never_attempted_case_ids"]
        != expected["never_attempted_case_ids"]
        or inventory_sha256
        != f21["evidence"][
            "authenticated_relative_size_content_inventory_sha256"
        ]
    ):
        raise RuntimeError("F22 incident state differs from sealed F21")

    topology = _topology()
    if (
        len(topology["all_scenes_v4"]) != 1
        or topology["all_scenes_v4"][0]["state"] != "T"
        or len(topology["scene_pipelines_v4"]) != 2
        or any(
            row["state"] != "T"
            for row in topology["scene_pipelines_v4"]
        )
        or topology["source_t2_collectors"]
        or len(topology["orphan_slotted_t2"]) != 1
        or topology["failed_f21_recovery"]
    ):
        raise RuntimeError("unexpected process topology before F22 seal")

    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-f22-orphan-wrapper-amendment.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_after_f21_source_termination_before_f22_resume",
        "passed": True,
        "predecessor_f21": str(runner.F21.relative_to(ROOT)),
        "predecessor_f21_sha256": runner.sha256(runner.F21),
        "incident": {
            "classification": "orphaned_slotted_t2_wrapper_after_child_exit",
            "scene_id": f21_sealer.SCENE,
            "f21_recovery_audit": str(f21_recovery.AUDIT_PATH),
            "f21_recovery_audit_sha256": runner.sha256(
                f21_recovery.AUDIT_PATH
            ),
            "live_process_topology": topology,
            "source_t2_collector_absent": True,
            "old_orchestration_paused": True,
            **state,
        },
        "correction": {
            "resumer_f22_sha256": runner.sha256(RESUMER),
            "sealer_f22_sha256": runner.sha256(Path(__file__).resolve()),
            "terminate_orphan_wrapper_before_atomic_resume": True,
            "f21_atomic_runner_unchanged": True,
            "f21_atomic_runner_sha256": runner.sha256(
                Path(runner.__file__).resolve()
            ),
        },
        "scientific_contract": {
            "schedule_protocol_or_case_membership_changed": False,
            "physics_sensor_render_or_seed_changed": False,
            "existing_completed_case_reexecuted": False,
            "feature_model_route_threshold_or_analysis_changed": False,
            "result_dependent_retry_enabled": False,
        },
    }
    F22.parent.mkdir(parents=True, exist_ok=True)
    temporary = F22.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, F22)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
