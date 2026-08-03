#!/usr/bin/env python3
"""Summarize O1--O3 calibration and held-out metric-geometry confirmations."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


EXPECTED_SCHEMAS = {
    "O1": "kinofail.realistic-forest-o1-pair-audit.v2-development",
    "O2": "kinofail.realistic-forest-o2-pair-audit.v2-development",
    "O3": "kinofail.realistic-forest-o3-pair-audit.v2-development",
}


def _load(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operator",
        nargs=5,
        action="append",
        metavar=("ID", "CALIBRATION_PAIR", "HELDOUT_PAIR", "PREFLIGHT", "POSTRUN"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for operator_id, calibration_spec, heldout_spec, preflight_spec, postrun_spec in args.operator:
        calibration_path = Path(calibration_spec).resolve()
        heldout_path = Path(heldout_spec).resolve()
        preflight_path = Path(preflight_spec).resolve()
        postrun_path = Path(postrun_spec).resolve()
        calibration = _load(calibration_path)
        heldout = _load(heldout_path)
        heldout_nominal = _load(Path(heldout["nominal_manifest"]))
        heldout_anomaly = _load(Path(heldout["anomaly_manifest"]))
        rows.append(
            {
                "operator_id": operator_id,
                "calibration_path": str(calibration_path),
                "calibration": calibration,
                "heldout_path": str(heldout_path),
                "heldout": heldout,
                "heldout_nominal": heldout_nominal,
                "heldout_anomaly": heldout_anomaly,
                "preflight_path": str(preflight_path),
                "preflight": _load(preflight_path),
                "postrun_path": str(postrun_path),
                "postrun": _load(postrun_path),
            }
        )

    heldout_controller_contracts = [
        json.dumps(row["heldout_anomaly"]["route_controller"], sort_keys=True)
        for row in rows
    ]
    checks = {
        "exactly_o1_o2_o3": {row["operator_id"] for row in rows} == set(EXPECTED_SCHEMAS),
        "all_calibration_pairs_pass_v2_as_calibration_only": all(
            row["calibration"].get("schema_version") == EXPECTED_SCHEMAS[row["operator_id"]]
            and row["calibration"].get("evidence_role") == "development_calibration"
            and row["calibration"].get("passed") is True
            for row in rows
        ),
        "all_heldout_pairs_pass_v2_as_geometry_confirmation": all(
            row["heldout"].get("schema_version") == EXPECTED_SCHEMAS[row["operator_id"]]
            and row["heldout"].get("evidence_role") == "heldout_geometry_confirmation"
            and row["heldout"].get("passed") is True
            for row in rows
        ),
        "every_operator_changes_resolved_compiled_geometry_from_calibration_to_heldout": all(
            _load(Path(row["calibration"]["nominal_manifest"]))[
                "compiled_audit_sha256"
            ]
            != row["heldout_nominal"]["compiled_audit_sha256"]
            for row in rows
        ),
        "three_heldout_trajectory_contracts_are_distinct": len(
            set(heldout_controller_contracts)
        )
        == 3,
        "three_heldout_seeds_are_distinct": len(
            {row["heldout_anomaly"]["seed"] for row in rows}
        )
        == 3,
        "all_protocol_preflights_pass": all(row["preflight"].get("passed") is True for row in rows),
        "all_protocol_postrun_seals_pass": all(row["postrun"].get("passed") is True for row in rows),
        "all_artifacts_remain_development_only": all(
            row["calibration"].get("counts_as_a0_a7_evidence") is False
            and row["heldout"].get("counts_as_a0_a7_evidence") is False
            and row["preflight"].get("counts_as_a0_a7_evidence") is False
            and row["postrun"].get("counts_as_a0_a7_evidence") is False
            for row in rows
        ),
    }
    result = {
        "schema_version": "kinofail.forest-metric-geometry-operator-summary.v1-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "operators": [
            {
                "operator_id": row["operator_id"],
                "calibration_pair": row["calibration_path"],
                "calibration_measurements": row["calibration"]["measurements"],
                "heldout_pair": row["heldout_path"],
                "heldout_measurements": row["heldout"]["measurements"],
                "preflight_audit": row["preflight_path"],
                "postrun_audit": row["postrun_path"],
                "heldout_seed": row["heldout_anomaly"]["seed"],
                "heldout_scene_id": row["heldout_anomaly"]["scene_id"],
                "heldout_route_controller": row["heldout_anomaly"]["route_controller"],
            }
            for row in rows
        ],
        "passed": all(checks.values()),
        "independent_metric_geometry_development_gate_closed_for_o1_o2_o3": all(
            checks.values()
        ),
        "interpretation": (
            "Corrected O1, O2, and O3 each passed a calibration geometry and a separately "
            "frozen held-out metric-geometry/trajectory confirmation. This is stronger than "
            "appearance-only replication, but remains component validation: it is not a "
            "registry-bound corpus, dose-response matrix, or realistic A0-A7 result."
        ),
        "provenance_note": (
            "The top-level episode.usda is only a relative sublayer container and can have the "
            "same byte hash across output directories. Geometry identity is therefore established "
            "by the frozen compiled-audit hash and resolved collision.usda, prop_collision.usda, "
            "and route.usda hashes, never by the container hash alone."
        ),
        "remaining_gates": [
            "multiple_additional_independent_scene_families_per_split",
            "operator_specific_dose_response_across_independent_geometries",
            "physical_fixture_calibration",
            "registry_bound_formal_episode_collection",
            "realistic_a0_a7_retraining_prediction_and_statistics",
        ],
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
