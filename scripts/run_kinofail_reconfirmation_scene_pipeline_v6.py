#!/usr/bin/env python3
"""F23 scene pipeline validating F17 before installing the F21 T2 runner."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_kinofail_reconfirmation_scene_pipeline_v5 as f21
from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as t2


F23 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f23_preflight_order_amendment1/"
    "amendment_manifest.json"
)
F17_PIPELINE = f21.predecessor.predecessor


def _validate_f23() -> dict[str, Any]:
    amendment = t2.read_json(F23)
    correction = amendment.get("correction", {})
    scientific = amendment.get("scientific_contract", {})
    paths = {
        "scene_pipeline_v6": Path(__file__).resolve(),
        "all_scenes_v6": (
            ROOT
            / "scripts/run_kinofail_reconfirmation_all_scene_pipelines_v6.py"
        ),
        "finalizer_v8": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v8.py"
        ),
        "restart_f23": (
            ROOT / "scripts/restart_kinofail_reconfirmation_f23.py"
        ),
        "sealer_f23": (
            ROOT / "scripts/seal_kinofail_reconfirmation_f23.py"
        ),
    }
    if (
        amendment.get("schema_version")
        != "kinofail.reconfirmation-f23-preflight-order-amendment.v1"
        or amendment.get("status")
        != "sealed_after_scene06_preflight_failure_before_acquisition"
        or amendment.get("passed") is not True
        or any(
            not path.is_file()
            or correction.get(f"{name}_sha256") != t2.sha256(path)
            for name, path in paths.items()
        )
        or correction.get("f17_validation_before_t2_entrypoint_swap")
        is not True
        or correction.get("f21_atomic_t2_entrypoint_preserved") is not True
        or scientific.get("schedule_protocol_or_case_membership_changed")
        is not False
        or scientific.get("physics_sensor_render_or_seed_changed") is not False
        or scientific.get("feature_model_route_threshold_or_analysis_changed")
        is not False
        or scientific.get("result_dependent_retry_enabled") is not False
    ):
        raise RuntimeError("F23 preflight-order amendment is invalid")
    return amendment


def install_f23_t2_runner() -> dict[str, Any]:
    validated_f17 = F17_PIPELINE._validate_f17()
    F17_PIPELINE._validate_f17 = lambda: validated_f17
    f21.install_f21_t2_runner()
    return validated_f17


def _augment_state(scene: str, amendment: dict[str, Any]) -> None:
    state_path = (
        f21.predecessor.predecessor.base.ORCHESTRATION_ROOT
        / scene
        / "pipeline_state.json"
    )
    if not state_path.is_file():
        return
    state = f21.predecessor.predecessor.base._json(state_path)
    state["schema_version"] = "kinofail.reconfirmation-scene-pipeline.v6"
    state["f23_preflight_order_amendment"] = {
        "path": str(F23),
        "sha256": f21.predecessor.predecessor.base._sha256(F23),
        "status": amendment["status"],
        "f17_validation_before_t2_entrypoint_swap": True,
        "f21_atomic_t2_entrypoint_preserved": True,
        "scientific_content_changed": False,
    }
    state["f23_state_wrapped_utc"] = datetime.now(UTC).isoformat()
    f21.predecessor.predecessor.base._write_json(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scene", required=True)
    known, _ = parser.parse_known_args()
    amendment = _validate_f23()
    f21_amendment = f21._validate_f21()
    install_f23_t2_runner()
    try:
        return int(f21.predecessor.main())
    finally:
        f21._augment_state(known.scene, f21_amendment)
        _augment_state(known.scene, amendment)


if __name__ == "__main__":
    raise SystemExit(main())
