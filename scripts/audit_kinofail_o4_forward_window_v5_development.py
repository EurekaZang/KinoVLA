#!/usr/bin/env python3
"""Audit the one-shot O4 forward-window v5 development decision."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/data/kinofail_o4_forward_window_v5_development_plan.json"
PLAN_SHA256 = "97f62666c34fe937262edee398d815d24350ee462dbfab0e01f22dd33b48fc46"
OUT = (
    ROOT
    / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2"
    / "o4_forward_window_v5_development_audit.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _same_number(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9)
    except (TypeError, ValueError):
        return False


def _coordinate_free_operator(parameters: dict[str, Any]) -> dict[str, Any]:
    """Remove the two fields that are deterministically derived from each scene frame."""

    return {
        key: value
        for key, value in parameters.items()
        if key not in ("surface_z_m", "progress_axis_xy")
    }


def main() -> None:
    plan = _json(PLAN)
    plan_hash_ok = _sha256(PLAN) == PLAN_SHA256
    sealed_specs = plan["sealed_negative_evidence"]
    implementation_specs = plan["frozen_implementation"]
    run_contract = plan["frozen_run_contract"]

    frozen_checks: dict[str, bool] = {
        "plan_hash": plan_hash_ok,
        "plan_frozen_before_execution": plan.get("status") == "frozen_before_execution",
        "no_iteration_rule": plan["decision_rule"].get("no_iteration") is True,
        "prior_formal_results_must_remain_failed": plan["decision_rule"].get(
            "prior_formal_results_remain_failed"
        )
        is True,
    }
    for name, spec in sealed_specs.items():
        path = _resolve(spec["path"])
        frozen_checks[f"sealed::{name}"] = path.is_file() and _sha256(path) == spec["sha256"]
    for name, spec in implementation_specs.items():
        path = _resolve(spec["path"])
        frozen_checks[f"implementation::{name}"] = (
            path.is_file() and _sha256(path) == spec["sha256"]
        )

    prior_formals = [
        _json(_resolve(sealed_specs["office_formal_failure"]["path"])),
        _json(_resolve(sealed_specs["bedroom_formal_failure"]["path"])),
    ]
    prior_thresholds = prior_formals[0]["paired_consequence"]["thresholds"]
    prior_operator = _coordinate_free_operator(prior_formals[0]["operator"]["parameters"])
    frozen_checks["prior_formals_still_failed"] = all(
        value.get("passed") is False for value in prior_formals
    )
    frozen_checks["prior_thresholds_match"] = all(
        value["paired_consequence"]["thresholds"] == prior_thresholds
        for value in prior_formals
    )
    frozen_checks["prior_operator_parameters_match"] = all(
        _coordinate_free_operator(value["operator"]["parameters"]) == prior_operator
        for value in prior_formals
    )

    expected_paths = {
        _resolve(row["output"]) / "pair_manifest.json"
        for row in plan["development_runs"]
    }
    observed_paths: set[Path] = set()
    for row in plan["development_runs"]:
        scene_root = _resolve(row["output"]).parent
        observed_paths.update(
            path.resolve()
            for path in scene_root.glob("development_forward*_v5/pair_manifest.json")
        )
    frozen_checks["exactly_predeclared_v5_development_outputs"] = observed_paths == expected_paths

    runs: list[dict[str, Any]] = []
    for row in plan["development_runs"]:
        manifest_path = _resolve(row["output"]) / "pair_manifest.json"
        checks: dict[str, bool] = {"manifest_present": manifest_path.is_file()}
        manifest = _json(manifest_path) if manifest_path.is_file() else {}
        protocol = manifest.get("protocol", {})
        consequence = manifest.get("paired_consequence", {})
        provenance = manifest.get("provenance", {})
        formal_protocol = manifest.get("formal_protocol")
        checks.update(
            {
                "scene_id": manifest.get("scene_id") == row["scene_id"],
                "source_scene_seed": int(manifest.get("source_scene_seed", -1))
                == int(row["source_scene_seed"]),
                "episode_seed": int(manifest.get("seed", -1)) == int(row["episode_seed"]),
                "calibration_role": manifest.get("protocol_role") == "calibration",
                "no_formal_protocol": formal_protocol is None,
                "forward_s": _same_number(protocol.get("forward_s"), run_contract["forward_s"]),
                "peel_pulse_s": _same_number(
                    protocol.get("peel_pulse_s"), run_contract["peel_pulse_s"]
                ),
                "recovery_s": _same_number(
                    protocol.get("post_peel_recovery_s"), run_contract["recovery_s"]
                ),
                "reverse_s": _same_number(
                    protocol.get("reverse_retreat_s"), run_contract["reverse_s"]
                ),
                "stop_s": _same_number(protocol.get("stop_s"), run_contract["stop_s"]),
                "rgb_fps": _same_number(protocol.get("rgb_fps"), run_contract["rgb_fps"]),
                "thresholds_unchanged": consequence.get("thresholds") == prior_thresholds,
                "operator_unchanged": _coordinate_free_operator(
                    manifest.get("operator", {}).get("parameters", {})
                )
                == prior_operator,
                "collector_hash": provenance.get("collector_sha256")
                == implementation_specs["collector"]["sha256"],
                "backend_hash": provenance.get("backend_sha256")
                == implementation_specs["backend"]["sha256"],
                "adhesion_hash": provenance.get("adhesion_model_sha256")
                == implementation_specs["adhesion_model"]["sha256"],
                "all_recorded_checks_pass": bool(manifest.get("checks"))
                and all(manifest["checks"].values()),
                "manifest_passed": manifest.get("passed") is True,
                "calibration_not_formal_evidence": manifest.get("dataset_status")
                == "calibration_pair_passed_not_formal_benchmark_evidence",
            }
        )
        metrics = consequence.get("metrics", {})
        runs.append(
            {
                "scene_id": row["scene_id"],
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
                },
                "passed": bool(checks) and all(checks.values()),
                "checks": checks,
                "metrics": {
                    "window_duration_s": metrics.get("window_duration_s"),
                    "progress_gain_lag_m": metrics.get("progress_gain_lag_m"),
                    "relative_forward_speed_suppression": metrics.get(
                        "relative_forward_speed_suppression"
                    ),
                    "max_tilt_increase_rad": metrics.get("max_tilt_increase_rad"),
                    "min_base_height_drop_m": metrics.get("min_base_height_drop_m"),
                },
            }
        )

    passed = all(frozen_checks.values()) and len(runs) == 2 and all(row["passed"] for row in runs)
    payload = {
        "schema_version": "kinofail.o4-forward-window-development-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "plan": {"path": str(PLAN), "sha256": _sha256(PLAN)},
        "passed": passed,
        "decision": "admit_for_new_heldout_batch" if passed else "reject",
        "frozen_checks": frozen_checks,
        "runs": runs,
        "interpretation": (
            "The two sealed formal failures remain failures. These post-seal runs are development "
            "evidence only and may support freezing a separate, newly held-out batch."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
