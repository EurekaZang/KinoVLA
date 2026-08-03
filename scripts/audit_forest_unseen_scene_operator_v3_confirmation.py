#!/usr/bin/env python3
"""Fail-closed pre/post audit for the G06/G07 unseen-scene v3 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.eval.realistic_operator_admission_v3 import (
    audit_o1_phase_robust_candidate,
    audit_o2_sinkage_normalized_visual_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_LAYERS = ("collision.usda", "prop_collision.usda", "route.usda")
VISUAL_LAYERS = ("appearance.usda", "background.usda")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _frozen_json(spec: dict[str, str]) -> tuple[Path, dict[str, Any], bool]:
    path = _resolve(spec["path"])
    valid = path.is_file() and _sha256(path) == spec["sha256"]
    return path, _json(path) if path.is_file() else {}, valid


def _strict_pair_path(scene: dict[str, Any], operator: str) -> Path:
    planned = scene["planned_outputs"]
    if operator == "o1":
        anomaly = _resolve(planned["o1_anomaly"])
        return anomaly.parent / anomaly.name.replace("_anomaly", "_pair_audit.json")
    return _resolve(planned[f"{operator}_pair"]) / "pair_audit.json"


def _pair_complete(scene: dict[str, Any], operator: str) -> bool:
    planned = scene["planned_outputs"]
    if operator == "o1":
        return (
            _resolve(planned["o1_nominal"]) / "lane_manifest.json"
        ).is_file() and (
            _resolve(planned["o1_anomaly"]) / "lane_manifest.json"
        ).is_file() and _strict_pair_path(scene, operator).is_file()
    base = _resolve(planned[f"{operator}_pair"])
    return all(
        path.is_file()
        for path in (
            base / "nominal/lane_manifest.json",
            base / "anomaly/lane_manifest.json",
            base / "pair_audit.json",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_forest_unseen_scene_operator_v3_confirmation_dev_v1.json",
    )
    parser.add_argument("--phase", choices=("preflight", "postrun"), required=True)
    parser.add_argument(
        "--preflight-audit",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_unseen_scene_operator_v3_confirmation_dev_v1/preflight_audit.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_unseen_scene_operator_v3_confirmation_dev_v1/audit.json",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _json(config_path)
    boundary = config.get("evidence_boundary", {})
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.forest-unseen-scene-operator-v3-confirmation.v1-development",
        "development_only": boundary.get("development_only") is True,
        "not_registry_evidence": boundary.get("scene_registry_eligible") is False,
        "not_a0_a7_evidence": boundary.get("counts_as_a0_a7_evidence") is False,
        "new_visual_families_declared": boundary.get(
            "g06_g07_appearance_families_unseen_by_g01_g05_operator_runs"
        )
        is True,
        "candidate_frozen_before_anomaly_declared": boundary.get(
            "candidate_v3_was_frozen_before_any_g06_g07_anomaly_run"
        )
        is True,
        "one_attempt_policy": config.get("execution_policy", {}).get(
            "one_attempt_per_scene_operator_lane"
        )
        is True,
        "no_retry_policy": config.get("execution_policy", {}).get(
            "no_retry_or_scene_replacement"
        )
        is True,
    }
    if args.phase == "postrun":
        preflight = _json(args.preflight_audit.resolve()) if args.preflight_audit.is_file() else {}
        checks["preflight_audit_passed"] = preflight.get("passed") is True
        checks["postrun_uses_preflight_frozen_config"] = (
            preflight.get("config", {}).get("sha256") == _sha256(config_path)
        )

    frozen_inputs: list[dict[str, str]] = []
    for group_name in ("frozen_sources", "asset_contract"):
        for name, spec in config.get(group_name, {}).items():
            path = _resolve(spec["path"])
            valid = path.is_file() and _sha256(path) == spec["sha256"]
            checks[f"{group_name}_{name}_hash"] = valid
            if path.suffix == ".json" and path.is_file():
                payload = _json(path)
                if name.endswith("preflight"):
                    checks[f"{group_name}_{name}_passed"] = payload.get("passed") is True
            frozen_inputs.append(
                {"name": f"{group_name}.{name}", "path": str(path), "sha256": spec["sha256"]}
            )

    calibration_path, calibration, calibration_valid = _frozen_json(
        config["candidate_origin"]["calibration_result"]
    )
    checks["candidate_calibration_hash"] = calibration_valid
    checks["candidate_calibration_passed"] = calibration.get("passed") is True
    checks["strict_v2_calibration_4_of_6_retained"] = calibration.get(
        "strict_v2_result"
    ) == {"passed_pairs": 4, "total_pairs": 6}
    checks["candidate_v3_calibration_6_of_6_disclosed"] = calibration.get(
        "candidate_v3_calibration_result"
    ) == {"passed_pairs": 6, "total_pairs": 6}
    frozen_inputs.append(
        {
            "name": "candidate_origin.calibration_result",
            "path": str(calibration_path),
            "sha256": config["candidate_origin"]["calibration_result"]["sha256"],
        }
    )

    endpoints = config.get("frozen_primary_endpoints", {})
    checks["o1_candidate_gate_frozen"] = endpoints.get("o1_candidate_v3") == {
        "mean_region_slip_delta_min": 0.30,
        "absolute_lateral_deviation_delta_m_min": 0.10,
        "kinematic_combination": "OR",
        "route_progress_deficit_m_min": 0.75,
        "nominal_completes_without_fall": True,
        "anomaly_falls": True,
    }
    checks["o2_candidate_gate_frozen"] = endpoints.get("o2_candidate_v3") == {
        "absolute_visual_depth_m_min": 0.05,
        "visual_depth_to_sinkage_ratio_min": 0.65,
        "contact_local_imprint_count_min": 4,
        "deformed_vertex_count_min_exclusive": 0,
        "physical_causal_and_outcome_checks_unchanged": True,
    }

    fingerprints: dict[str, dict[str, str]] = {}
    for row in config.get("independence_reference_compiled_audits", []):
        path, audit, valid = _frozen_json(row)
        scene_id = str(row["realization_id"])
        checks[f"{scene_id}_reference_audit_hash"] = valid
        checks[f"{scene_id}_reference_scene_passed"] = audit.get("passed") is True
        fingerprints[scene_id] = {
            layer: str(audit.get("files", {}).get(layer, ""))
            for layer in (*PHYSICAL_LAYERS, *VISUAL_LAYERS)
        }
        frozen_inputs.append(
            {"name": f"reference.{scene_id}", "path": str(path), "sha256": row["sha256"]}
        )

    scene_results: list[dict[str, Any]] = []
    strict_outcomes: dict[str, dict[str, Any]] = {}
    candidate_outcomes: dict[str, dict[str, Any]] = {}
    for scene in config.get("scene_realizations", []):
        scene_id = str(scene["realization_id"])
        scene_config_path, _, scene_config_valid = _frozen_json(scene["config"])
        audit_path, compiled, compiled_valid = _frozen_json(scene["compiled_audit"])
        nominal_path, nominal, nominal_valid = _frozen_json(scene["nominal_route_preflight"])
        checks[f"{scene_id}_config_hash"] = scene_config_valid
        checks[f"{scene_id}_compiled_audit_hash"] = compiled_valid
        checks[f"{scene_id}_compiled_scene_passed"] = compiled.get("passed") is True
        checks[f"{scene_id}_compiled_development_only"] = (
            compiled.get("scene_registry_eligible") is False
            and compiled.get("counts_as_a0_a7_evidence") is False
        )
        checks[f"{scene_id}_nominal_preflight_hash"] = nominal_valid
        checks[f"{scene_id}_nominal_preflight_passed"] = nominal.get("passed") is True
        checks[f"{scene_id}_nominal_route_completed"] = (
            nominal.get("checks", {}).get("nominal_route_completed") is True
            and nominal.get("measurements", {}).get("fallen") is False
        )
        checks[f"{scene_id}_o1_seed_matches_frozen_nominal"] = nominal.get("seed") == config[
            "runtime_contracts"
        ][scene_id]["o1"]["seed"]

        compiled_dir = audit_path.parent
        hashes: dict[str, str] = {}
        for layer in (*PHYSICAL_LAYERS, *VISUAL_LAYERS):
            layer_path = compiled_dir / layer
            actual = _sha256(layer_path) if layer_path.is_file() else ""
            hashes[layer] = actual
            checks[f"{scene_id}_{layer}_bound"] = (
                bool(actual) and compiled.get("files", {}).get(layer) == actual
            )
        fingerprints[scene_id] = hashes

        planned = {name: _resolve(path) for name, path in scene["planned_outputs"].items()}
        if args.phase == "preflight":
            checks[f"{scene_id}_o1_anomaly_output_unused"] = not planned["o1_anomaly"].exists()
            checks[f"{scene_id}_o2_output_unused"] = not planned["o2_pair"].exists()
            checks[f"{scene_id}_o3_output_unused"] = not planned["o3_pair"].exists()
        else:
            strict_scene: dict[str, Any] = {}
            candidate_scene: dict[str, Any] = {}
            for operator in ("o1", "o2", "o3"):
                checks[f"{scene_id}_{operator}_pair_complete"] = _pair_complete(scene, operator)
                pair_path = _strict_pair_path(scene, operator)
                if not pair_path.is_file():
                    continue
                pair = _json(pair_path)
                strict_scene[operator.upper()] = {
                    "passed": pair.get("passed") is True,
                    "failed_checks": sorted(
                        key for key, value in pair.get("checks", {}).items() if value is not True
                    ),
                    "path": str(pair_path),
                    "measurements": pair.get("measurements", {}),
                }
                if operator == "o1":
                    candidate = audit_o1_phase_robust_candidate(
                        pair, evidence_role="unseen_scene_confirmation"
                    )
                elif operator == "o2":
                    anomaly_path = planned["o2_pair"] / "anomaly/lane_manifest.json"
                    candidate = audit_o2_sinkage_normalized_visual_candidate(
                        pair,
                        _json(anomaly_path),
                        evidence_role="unseen_scene_confirmation",
                    )
                else:
                    candidate = {
                        "operator": "O3",
                        "evidence_role": "unseen_scene_confirmation",
                        "passed": pair.get("passed") is True,
                        "contract": "unchanged_v2_event_trigger_contract",
                        "counts_as_a0_a7_evidence": False,
                    }
                candidate_scene[operator.upper()] = candidate
            strict_outcomes[scene_id] = strict_scene
            candidate_outcomes[scene_id] = candidate_scene

        scene_results.append(
            {
                "realization_id": scene_id,
                "appearance_family": scene["appearance_family"],
                "config": str(scene_config_path),
                "compiled_audit": str(audit_path),
                "nominal_route_preflight": str(nominal_path),
                "fingerprint": hashes,
                "planned_outputs": {key: str(value) for key, value in planned.items()},
            }
        )

    for layer in PHYSICAL_LAYERS:
        values = [row[layer] for row in fingerprints.values()]
        checks[f"all_g01_g07_{layer}_hashes_nonempty"] = all(values)
        checks[f"all_g01_g07_{layer}_hashes_distinct"] = len(values) == len(set(values))
    for layer in VISUAL_LAYERS:
        confirmation_values = [fingerprints[scene_id][layer] for scene_id in ("g06", "g07")]
        prior_values = {fingerprints[scene_id][layer] for scene_id in ("g01", "g02", "g03", "g04", "g05")}
        checks[f"g06_g07_{layer}_distinct_from_each_other"] = len(set(confirmation_values)) == 2
        checks[f"g06_g07_{layer}_unseen_by_g01_g05"] = not any(
            value in prior_values for value in confirmation_values
        )

    strict_passes = sum(
        outcome.get("passed") is True
        for scene in strict_outcomes.values()
        for outcome in scene.values()
    )
    candidate_passes = sum(
        outcome.get("passed") is True
        for scene in candidate_outcomes.values()
        for outcome in scene.values()
    )
    total_pairs = sum(len(scene) for scene in candidate_outcomes.values())
    candidate_targeted = [
        outcome
        for scene in candidate_outcomes.values()
        for operator, outcome in scene.items()
        if operator in {"O1", "O2"}
    ]
    unchanged_o3 = [
        scene["O3"] for scene in candidate_outcomes.values() if "O3" in scene
    ]
    result = {
        "schema_version": "kinofail.forest-unseen-scene-operator-v3-confirmation-audit.v1-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": args.phase,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "passed": all(checks.values()),
        "checks": checks,
        "scene_results": scene_results,
        "fingerprints": fingerprints,
        "strict_v2_outcomes": strict_outcomes,
        "candidate_v3_outcomes": candidate_outcomes,
        "endpoint_summary": {
            "strict_v2_passed_pairs": strict_passes,
            "candidate_v3_passed_pairs": candidate_passes,
            "total_pairs": total_pairs,
            "candidate_v3_all_six_passed": total_pairs == 6 and candidate_passes == 6,
            "candidate_targeted_o1_o2_passed_pairs": sum(
                outcome.get("passed") is True for outcome in candidate_targeted
            ),
            "candidate_targeted_o1_o2_total_pairs": len(candidate_targeted),
            "candidate_targeted_o1_o2_all_passed": len(candidate_targeted) == 4
            and all(outcome.get("passed") is True for outcome in candidate_targeted),
            "unchanged_o3_passed_pairs": sum(
                outcome.get("passed") is True for outcome in unchanged_o3
            ),
            "unchanged_o3_total_pairs": len(unchanged_o3),
            "interpretation": (
                "Scientific endpoint outcomes are reported separately from protocol integrity. "
                "A failed frozen endpoint remains failed and is never hidden by passed=true."
            ),
        },
        "frozen_inputs": frozen_inputs,
        "evidence_boundary": {
            "unseen_photographic_appearance_families": True,
            "new_metric_geometries": True,
            "scene_registry_eligible": False,
            "counts_as_a0_a7_evidence": False,
            "realistic_a0_a7_still_requires_registry_and_full_rerun": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "phase": args.phase,
                "passed": result["passed"],
                **result["endpoint_summary"],
            },
            indent=2,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
