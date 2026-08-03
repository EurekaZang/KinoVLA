#!/usr/bin/env python3
"""Seal the exposed-scene evidence that motivates the v16 O2/runtime contract."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "outputs/kinofail_realistic/operator_development/o2_continuous_support_v4_phase_aware/development_audit.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(relative: str) -> tuple[Path, dict[str, Any]]:
    path = (ROOT / relative).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return path, value


def _artifact(relative: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path, value = _json(relative)
    return (
        {
            "path": str(path),
            "sha256": _sha256(path),
            "passed": value.get("passed"),
            "schema_version": value.get("schema_version"),
        },
        value,
    )


def main() -> int:
    paths = {
        "v15_postrun": "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v15_unseen_batch_v1/postrun_audit.json",
        "living_low_speed_baseline": "outputs/kinofail_realistic/operator_development/o2_continuous_support_v2/living39/go2_baseline_speed018_target050/go2_scene_audit.json",
        "living_nominal": "outputs/kinofail_realistic/operator_development/o2_continuous_support_v3_speed032/living39/nominal/lane_manifest.json",
        "living_anomaly": "outputs/kinofail_realistic/operator_development/o2_continuous_support_v4_phase_aware/living39/anomaly/lane_manifest.json",
        "kitchen_nominal": "outputs/kinofail_realistic/operator_development/o2_continuous_support_v4_phase_aware/kitchen40/nominal/lane_manifest.json",
        "kitchen_anomaly": "outputs/kinofail_realistic/operator_development/o2_continuous_support_v4_phase_aware/kitchen40/anomaly/lane_manifest.json",
        "failure_safe_base_rejection": "outputs/kinofail_realistic/operator_development/failure_safe_stage_v1/dining38/base_compile/compiled_scene_audit.json",
        "failure_safe_corridor_rejection": "outputs/kinofail_realistic/operator_development/failure_safe_stage_v1/office42/corridor_v2/compiled_scene_audit.json",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        artifacts[name], records[name] = _artifact(path)

    nominal_names = ("living_nominal", "kitchen_nominal")
    anomaly_names = ("living_anomaly", "kitchen_anomaly")
    checks = {
        "v15_not_confirmed": records["v15_postrun"].get("passed") is False
        and records["v15_postrun"].get("policy_confirmation_state") == "not_confirmed",
        "low_speed_continuous_floor_baseline_failed": records[
            "living_low_speed_baseline"
        ].get("passed")
        is False
        and records["living_low_speed_baseline"].get("checks", {}).get(
            "kino_custom_floor_collision_enabled"
        )
        is True,
        "two_scene_nominal_passed": all(records[name].get("passed") is True for name in nominal_names),
        "two_scene_nominal_completed_without_fall": all(
            records[name].get("checks", {}).get("nominal_route_completed") is True
            and records[name].get("checks", {}).get("nominal_robot_not_fallen") is True
            and float(
                records[name].get("measurements", {}).get(
                    "maximum_absolute_route_lateral_offset_m", 1.0
                )
            )
            <= 0.30
            for name in nominal_names
        ),
        "two_scene_anomaly_passed": all(records[name].get("passed") is True for name in anomaly_names),
        "two_scene_anomaly_measured_physics": all(
            records[name].get("checks", {}).get("o2_continuous_seam_free_topology")
            is True
            and records[name].get("checks", {}).get("actual_foot_sinkage") is True
            and records[name].get("checks", {}).get("multiple_loaded_feet") is True
            and records[name].get("checks", {}).get("nonzero_shear_dissipation") is True
            and records[name].get("checks", {}).get("physical_footprints") is True
            for name in anomaly_names
        ),
        "failure_safe_base_rejection_terminal": records[
            "failure_safe_base_rejection"
        ].get("passed")
        is False
        and records["failure_safe_base_rejection"].get("admission_state")
        == "base_compile_rejected",
        "failure_safe_corridor_rejection_terminal": records[
            "failure_safe_corridor_rejection"
        ].get("passed")
        is False
        and records["failure_safe_corridor_rejection"].get("admission_state")
        == "corridor_v2_refinement_rejected",
    }
    sources = {}
    for relative in (
        "kino_vla/sim/terramechanics.py",
        "kino_vla/sim/operators/o2_compliance.py",
        "kino_vla/sim/isaac_policy_backend.py",
        "scripts/isaac_collect_realistic_route_operator_lane_v6.py",
        "scripts/compile_embodiedgen_kinofail_scene.py",
        "scripts/refine_embodiedgen_kinofail_scene_v2.py",
    ):
        path = (ROOT / relative).resolve()
        sources[relative] = {"path": str(path), "sha256": _sha256(path)}
    result = {
        "schema_version": "kinofail.o2-continuous-support-v16-development-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "artifacts": artifacts,
        "source_freeze_candidates": sources,
        "diagnosis": {
            "v15_primary_runtime_confound": "0.18_mps_low_speed_locomotion_instability",
            "v15_secondary_evidence_issue": "non_terminal_failure_artifact_paths",
            "o2_architecture_change": "single_continuous_topology_matched_heightfield_with_explicit_route_axis",
            "next_formal_runtime_speed_mps": 0.32,
            "known_unstable_bedroom28_anchor_retired": True,
        },
        "evidence_boundary": {
            "development_only": True,
            "counts_as_realistic_corpus_evidence": False,
            "counts_as_a0_a7_evidence": False,
            "realistic_a0_a7_readiness": "0/8",
            "old_a0_a7_results_are_reference_only": True,
            "a8_excluded": True,
        },
    }
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite development audit: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
