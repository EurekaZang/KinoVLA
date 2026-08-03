#!/usr/bin/env python3
"""Audit v34 scene acquisition with a six-scene held-out quota."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_o4_scene_stream_v30 as v30  # noqa: E402


def _quota_contract(policy: dict[str, Any]) -> bool:
    return (
        policy.get("minimum_admitted_scenes") == 6
        and policy.get("minimum_admitted_families") == 4
    )


def _preflight(config_path: Path, out: Path) -> int:
    config = v30._json_v30(config_path)
    implementation = v30.implementation
    pipeline = config.get("pipeline", {})
    policy = config.get("selection_policy", {})
    scenes = config.get("candidate_stream", [])
    material_lock = implementation._json(
        implementation._resolve(config.get("material_lock", {}).get("path", ""))
    )
    materials = {item.get("id"): item for item in material_lock.get("materials", [])}
    ids = [scene.get("scene_id") for scene in scenes]
    families = Counter(scene.get("room_type") for scene in scenes)
    checks: dict[str, bool] = {
        "supported_v34_wrapper_schema": v30.predecessor._original_json(config_path).get(
            "schema_version"
        )
        == "kinofail.o4-scene-stream-v30-freeze.v1"
        and v30.predecessor._original_json(config_path).get("freeze_status")
        == "frozen_before_any_v34_heldout_candidate_generation_or_o4_execution",
        "frozen_before_stream_execution": config.get("freeze_status")
        == implementation.FREEZE_STATUS,
        "strict_evidence_quarantine": config.get("evidence_policy", {}).get(
            "counts_as_a0_a7_evidence"
        )
        is False
        and config.get("evidence_policy", {}).get("development_only") is True
        and config.get("evidence_policy", {}).get("a8_in_scope") is False,
        "twenty_four_candidate_fixed_order": len(scenes) == 24
        and ids == policy.get("fixed_order")
        and len(set(ids)) == 24,
        "balanced_six_family_stream": set(families)
        == {"Office", "LivingRoom", "Kitchen", "House", "Bedroom", "DiningRoom"}
        and all(count == 4 for count in families.values()),
        "strict_prefix_stop": policy.get("processing_order")
        == "strict_candidate_stream_order"
        and policy.get("stop_at_first_prefix_meeting_quota") is True
        and policy.get("maximum_candidates") == 24,
        "heldout_six_scene_quota": _quota_contract(policy),
        "one_attempt_no_repair": policy.get("one_generation_attempt_per_candidate")
        is True
        and policy.get("replacement_seed_allowed") is False
        and policy.get("source_repair_allowed") is False,
        "operator_blind_acquisition": policy.get("o4_withheld_until_cohort_frozen")
        is True,
        "simple_only": all(scene.get("complexity") == "simple" for scene in scenes),
        "unique_positive_seeds": len({scene.get("source_seed") for scene in scenes})
        == len(scenes)
        and all(
            isinstance(scene.get("source_seed"), int) and scene["source_seed"] > 0
            for scene in scenes
        ),
        "locked_files_match": bool(config.get("locked_files"))
        and all(implementation._locked(item) for item in config.get("locked_files", [])),
        "material_lock_matches": implementation._locked(config.get("material_lock", {})),
        "operator_output_root_unused": not implementation._resolve(
            config.get("operator_output_root", "")
        ).exists(),
        "audited_candidate_runner_only": pipeline.get("candidate_runner")
        == "scripts/run_o4_scene_candidate_v28.py",
    }
    for scene in scenes:
        paths = v30._candidate_paths_v30(config, scene)
        material = materials.get(scene.get("material_id"), {})
        checks[f"{scene['scene_id']}_unused"] = not any(
            paths[name].exists()
            for name in ("source", "request", "generation_exception", "receipt")
        )
        checks[f"{scene['scene_id']}_material_frozen"] = (
            material.get("id") == scene.get("material_id")
            and material.get("split") == scene.get("material_split")
        )
    passed = all(checks.values())
    result = {
        "schema_version": "kinofail.o4-scene-stream-v34-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": implementation._sha256(config_path),
        "checks": checks,
        "passed": passed,
        "scene_stream_execution_authorized": passed,
        "o4_execution_authorized": False,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": passed}, indent=2))
    return 0 if passed else 2


def _patch_postrun(out: Path) -> None:
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["schema_version"] = "kinofail.o4-scene-stream-v34-postrun.v1"
    payload["audit_adapter"] = "v34_six_scene_heldout_quota"
    payload["v30_audit_implementation_reused"] = True
    payload["v34_heldout_confirmation_authorized"] = payload.get("passed") is True
    payload["next_gate"] = (
        "Freeze the unchanged v32 runtime and prospective v34 phase-aware route contract "
        "against every admitted scene, then execute each scene exactly once."
    )
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "postrun"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    out = args.out.resolve()
    if args.mode == "preflight":
        if args.preflight is not None:
            parser.error("--preflight is valid only in postrun mode")
        return _preflight(config, out)
    if args.preflight is None:
        parser.error("--preflight is required in postrun mode")
    original_argv = sys.argv
    try:
        sys.argv = [
            str(Path(v30.__file__)),
            "--mode",
            "postrun",
            "--config",
            str(config),
            "--preflight",
            str(args.preflight.resolve()),
            "--out",
            str(out),
        ]
        code = v30.main()
    finally:
        sys.argv = original_argv
    _patch_postrun(out)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
