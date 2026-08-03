#!/usr/bin/env python3
"""Fail-closed pre/post audit for the G04/G05 crossed-factor O1--O3 replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_LAYERS = ("collision.usda", "prop_collision.usda", "route.usda")


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


def _pair_complete(path: Path, operator: str) -> bool:
    pair = path / "pair_audit.json"
    nominal = path / "nominal" / "lane_manifest.json"
    anomaly = path / "anomaly" / "lane_manifest.json"
    if operator == "o1":
        # O1 reuses the separately frozen nominal-route preflight and stores only its anomaly
        # under the planned output path.  Its pair audit is written beside the anomaly directory.
        return False
    return pair.is_file() and nominal.is_file() and anomaly.is_file()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_forest_crossed_appearance_geometry_o1_o3_replication_dev_v1.json",
    )
    parser.add_argument("--phase", choices=("preflight", "postrun"), required=True)
    parser.add_argument(
        "--preflight-audit",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_crossed_appearance_geometry_o1_o3_replication_dev_v1/preflight_audit.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_crossed_appearance_geometry_o1_o3_replication_dev_v1/audit.json",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _json(config_path)
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.forest-crossed-appearance-geometry-o1-o3-replication.v1-development",
        "development_only": config.get("evidence_boundary", {}).get("development_only")
        is True,
        "not_registry_evidence": config.get("evidence_boundary", {}).get(
            "scene_registry_eligible"
        )
        is False,
        "not_a0_a7_evidence": config.get("evidence_boundary", {}).get(
            "counts_as_a0_a7_evidence"
        )
        is False,
        "visual_family_reuse_disclosed": config.get("evidence_boundary", {}).get(
            "appearance_families_were_used_in_prior_same_geometry_development"
        )
        is True,
    }
    if args.phase == "postrun":
        preflight = _json(args.preflight_audit.resolve()) if args.preflight_audit.is_file() else {}
        checks["preflight_audit_passed"] = preflight.get("passed") is True
        checks["postrun_uses_preflight_frozen_config"] = (
            preflight.get("config", {}).get("sha256") == _sha256(config_path)
        )
    else:
        preflight = {}

    frozen_inputs: list[dict[str, Any]] = []
    for name, spec in config.get("frozen_sources", {}).items():
        path = _resolve(spec["path"])
        valid = path.is_file() and _sha256(path) == spec["sha256"]
        checks[f"frozen_source_{name}"] = valid
        frozen_inputs.append({"name": name, "path": str(path), "sha256": spec["sha256"]})

    physical_fingerprints: dict[str, dict[str, str]] = {}
    reference_rows = config.get("independence_reference_compiled_audits", [])
    for row in reference_rows:
        audit_path, audit, valid = _frozen_json(row)
        realization_id = str(row["realization_id"])
        checks[f"{realization_id}_compiled_audit_hash"] = valid
        checks[f"{realization_id}_compiled_scene_passed"] = audit.get("passed") is True
        physical_fingerprints[realization_id] = {
            layer: str(audit.get("files", {}).get(layer, "")) for layer in PHYSICAL_LAYERS
        }
        frozen_inputs.append(
            {"name": f"{realization_id}_compiled_audit", "path": str(audit_path), "sha256": row["sha256"]}
        )

    scene_results: list[dict[str, Any]] = []
    operator_outcomes: dict[str, dict[str, Any]] = {}
    for scene in config.get("scene_realizations", []):
        realization_id = str(scene["realization_id"])
        config_path_scene, _, config_valid = _frozen_json(scene["config"])
        audit_path, compiled, audit_valid = _frozen_json(scene["compiled_audit"])
        nominal_path, nominal, nominal_valid = _frozen_json(scene["nominal_route_preflight"])
        checks[f"{realization_id}_config_hash"] = config_valid
        checks[f"{realization_id}_compiled_audit_hash"] = audit_valid
        checks[f"{realization_id}_compiled_scene_passed"] = compiled.get("passed") is True
        checks[f"{realization_id}_compiled_development_only"] = (
            compiled.get("scene_registry_eligible") is False
            and compiled.get("counts_as_a0_a7_evidence") is False
        )
        checks[f"{realization_id}_nominal_preflight_hash"] = nominal_valid
        checks[f"{realization_id}_nominal_preflight_passed"] = nominal.get("passed") is True
        measurements = nominal.get("measurements", {})
        checks[f"{realization_id}_nominal_route_completed"] = (
            nominal.get("checks", {}).get("nominal_route_completed") is True
            and measurements.get("fallen") is False
        )
        compiled_dir = audit_path.parent
        layer_hashes: dict[str, str] = {}
        for layer in PHYSICAL_LAYERS:
            layer_path = compiled_dir / layer
            actual = _sha256(layer_path) if layer_path.is_file() else ""
            layer_hashes[layer] = actual
            checks[f"{realization_id}_{layer}_bound"] = (
                bool(actual) and compiled.get("files", {}).get(layer) == actual
            )
        physical_fingerprints[realization_id] = layer_hashes

        planned = {name: _resolve(path) for name, path in scene["planned_outputs"].items()}
        if args.phase == "preflight":
            checks[f"{realization_id}_o1_anomaly_output_unused"] = not planned[
                "o1_anomaly"
            ].exists()
            checks[f"{realization_id}_o2_output_unused"] = not planned["o2_pair"].exists()
            checks[f"{realization_id}_o3_output_unused"] = not planned["o3_pair"].exists()
        else:
            o1_pair = planned["o1_anomaly"].parent / (
                planned["o1_anomaly"].name.replace("_anomaly", "_pair_audit.json")
            )
            checks[f"{realization_id}_o1_anomaly_present"] = (
                planned["o1_anomaly"] / "lane_manifest.json"
            ).is_file()
            checks[f"{realization_id}_o1_pair_audit_present"] = o1_pair.is_file()
            scene_outcomes: dict[str, Any] = {}
            if o1_pair.is_file():
                pair = _json(o1_pair)
                scene_outcomes["O1"] = {
                    "passed": pair.get("passed") is True,
                    "path": str(o1_pair),
                    "failed_checks": sorted(
                        key for key, value in pair.get("checks", {}).items() if value is not True
                    ),
                    "measurements": pair.get("measurements", {}),
                }
            for operator in ("o2", "o3"):
                checks[f"{realization_id}_{operator}_pair_complete"] = _pair_complete(
                    planned[f"{operator}_pair"], operator
                )
                pair_path = planned[f"{operator}_pair"] / "pair_audit.json"
                if pair_path.is_file():
                    pair = _json(pair_path)
                    scene_outcomes[operator.upper()] = {
                        "passed": pair.get("passed") is True,
                        "path": str(pair_path),
                        "failed_checks": sorted(
                            key
                            for key, value in pair.get("checks", {}).items()
                            if value is not True
                        ),
                        "measurements": pair.get("measurements", {}),
                    }
            operator_outcomes[realization_id] = scene_outcomes

        scene_results.append(
            {
                "realization_id": realization_id,
                "appearance_family": scene["appearance_family"],
                "config": str(config_path_scene),
                "compiled_audit": str(audit_path),
                "nominal_route_preflight": str(nominal_path),
                "physical_fingerprint": layer_hashes,
                "planned_outputs": {key: str(value) for key, value in planned.items()},
            }
        )

    for layer in PHYSICAL_LAYERS:
        hashes = [fingerprint[layer] for fingerprint in physical_fingerprints.values()]
        checks[f"all_g01_g05_{layer}_hashes_nonempty"] = all(bool(value) for value in hashes)
        checks[f"all_g01_g05_{layer}_hashes_distinct"] = len(hashes) == len(set(hashes))
    appearance_families = [row["appearance_family"] for row in scene_results]
    checks["g04_g05_appearance_families_distinct"] = len(appearance_families) == len(
        set(appearance_families)
    )

    result = {
        "schema_version": "kinofail.forest-crossed-appearance-geometry-protocol-audit.v1-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": args.phase,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "passed": all(checks.values()),
        "checks": checks,
        "scene_results": scene_results,
        "physical_fingerprints": physical_fingerprints,
        "operator_outcomes": operator_outcomes,
        "primary_endpoint_summary": {
            "passed_pairs": sum(
                outcome.get("passed") is True
                for scene in operator_outcomes.values()
                for outcome in scene.values()
            ),
            "total_pairs": sum(len(scene) for scene in operator_outcomes.values()),
            "all_six_pairs_passed": bool(operator_outcomes)
            and all(
                outcome.get("passed") is True
                for scene in operator_outcomes.values()
                for outcome in scene.values()
            ),
            "interpretation": (
                "Pair outcomes are scientific results and are not folded into protocol-integrity "
                "passed; failed preregistered endpoints remain failed and are retained."
            ),
        },
        "frozen_inputs": frozen_inputs,
        "evidence_boundary": {
            "crosses_two_appearance_families_with_two_new_metric_geometries": True,
            "fully_heldout_visual_family_test": False,
            "scene_registry_eligible": False,
            "counts_as_a0_a7_evidence": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "phase": args.phase, "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
