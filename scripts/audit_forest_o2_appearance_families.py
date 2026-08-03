#!/usr/bin/env python3
"""Audit three physics-identical realistic forest O2 appearance families."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _capture(manifest: dict, label: str) -> dict:
    return next(row for row in manifest["camera"]["captures"] if row["label"] == label)


def _material_split(contract: dict, role: str) -> str:
    explicit = contract.get(f"{role}_material_split")
    if explicit is not None:
        return str(explicit)
    material_id = str(contract[f"{role}_material_id"])
    prefix = material_id.split("_", 1)[0]
    if prefix not in {"train", "val", "test"}:
        raise ValueError(f"cannot recover frozen split from material id: {material_id}")
    return prefix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        nargs=3,
        action="append",
        metavar=("FAMILY_ID", "ROLE", "PAIR_AUDIT"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.family) != 3:
        raise ValueError("exactly three synchronized appearance families are required")

    families = []
    for family_id, role, pair_spec in args.family:
        pair_path = Path(pair_spec).resolve()
        pair = _load(pair_path)
        nominal = _load(Path(pair["nominal_manifest"]))
        anomaly = _load(Path(pair["anomaly_manifest"]))
        compiled_path = Path(nominal["compiled_audit"]).resolve()
        compiled = _load(compiled_path)
        families.append(
            {
                "family_id": family_id,
                "role": role,
                "pair_audit_path": str(pair_path),
                "pair_audit_passed": pair.get("passed") is True,
                "nominal": nominal,
                "anomaly": anomaly,
                "compiled_path": str(compiled_path),
                "compiled": compiled,
                "summary": {
                    "panorama_sha256": compiled["background_contract"]["panorama_sha256"],
                    "route_material_id": compiled["near_field_contract"][
                        "route_material_id"
                    ],
                    "route_material_split": _material_split(
                        compiled["near_field_contract"], "route"
                    ),
                    "surrounding_material_id": compiled["near_field_contract"][
                        "surrounding_material_id"
                    ],
                    "surrounding_material_split": _material_split(
                        compiled["near_field_contract"], "surrounding"
                    ),
                    "pre_entry_image_sha256": _capture(
                        nominal, "pre_entry"
                    )["image_sha256"],
                    "pair_measurements": pair["measurements"],
                },
            }
        )

    reference = families[0]
    ref_nominal = reference["nominal"]
    ref_anomaly = reference["anomaly"]
    ref_compiled = reference["compiled"]
    physical_files = ("collision.usda", "prop_collision.usda", "route.usda")
    checks = {
        "exactly_three_families": len(families) == 3,
        "family_ids_unique": len({row["family_id"] for row in families}) == 3,
        "two_train_one_heldout_roles": [row["role"] for row in families].count("train")
        == 2
        and [row["role"] for row in families].count("heldout_validation") == 1,
        "all_pair_audits_passed": all(row["pair_audit_passed"] for row in families),
        "three_distinct_frozen_panorama_hashes": len(
            {row["summary"]["panorama_sha256"] for row in families}
        )
        == 3,
        "rendered_pre_entry_frames_are_distinct": len(
            {row["summary"]["pre_entry_image_sha256"] for row in families}
        )
        == 3,
        "at_least_two_route_materials": len(
            {row["summary"]["route_material_id"] for row in families}
        )
        >= 2,
        "at_least_two_surrounding_materials": len(
            {row["summary"]["surrounding_material_id"] for row in families}
        )
        >= 2,
        "heldout_validation_material_present": any(
            row["role"] == "heldout_validation"
            and "val"
            in {
                row["summary"]["route_material_split"],
                row["summary"]["surrounding_material_split"],
            }
            for row in families
        ),
        "same_seed": all(row["anomaly"]["seed"] == ref_anomaly["seed"] for row in families),
        "same_runtime_source_hashes": all(
            all(
                row["anomaly"].get("provenance", {}).get(key)
                == ref_anomaly.get("provenance", {}).get(key)
                for key in ("backend_sha256", "operator_sha256", "script_sha256")
            )
            for row in families
        ),
        "same_operator_parameters": all(
            row["anomaly"]["operator"]["parameters"]
            == ref_anomaly["operator"]["parameters"]
            for row in families
        ),
        "same_camera_profile": all(
            row["anomaly"]["camera"]["profile"] == ref_anomaly["camera"]["profile"]
            for row in families
        ),
        "same_camera_intrinsics": all(
            _capture(row["anomaly"], "pre_entry")["intrinsic_matrix"]
            == _capture(ref_anomaly, "pre_entry")["intrinsic_matrix"]
            for row in families
        ),
        "same_metric_route_and_collision_layers": all(
            all(
                row["compiled"]["files"][name] == ref_compiled["files"][name]
                for name in physical_files
            )
            for row in families
        ),
        "same_nominal_outcome": all(
            row["nominal"]["measurements"] == ref_nominal["measurements"]
            for row in families
        ),
        "same_anomaly_outcome": all(
            row["anomaly"]["measurements"] == ref_anomaly["measurements"]
            for row in families
        ),
        "appearance_is_physics_independent": all(
            row["anomaly"]["appearance"]["physics_independent"] is True
            for row in families
        ),
        "contact_local_visual_contract_identical": all(
            row["anomaly"]["operator"]["telemetry"]["continuous_visual_surface"][
                "mode"
            ]
            == "contact_local_load_triggered_imprints"
            and row["anomaly"]["operator"]["telemetry"]["continuous_visual_surface"][
                "surface_predeformed"
            ]
            is False
            and row["anomaly"]["operator"]["telemetry"]["continuous_visual_surface"][
                "operator_region_has_no_visual_boundary"
            ]
            is True
            for row in families
        ),
        "still_not_a0_a7_evidence": all(
            row["pair_audit_path"]
            and row["nominal"].get("counts_as_a0_a7_evidence") is False
            and row["anomaly"].get("counts_as_a0_a7_evidence") is False
            for row in families
        ),
    }
    result = {
        "schema_version": "kinofail.realistic-forest-o2-appearance-family-audit.v1-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "development_only": True,
        "operator_id": "O2_compliance",
        "checks": checks,
        "families": [
            {
                "family_id": row["family_id"],
                "role": row["role"],
                "pair_audit": row["pair_audit_path"],
                "compiled_audit": row["compiled_path"],
                **row["summary"],
            }
            for row in families
        ],
        "invariant_physics": {
            "seed": ref_anomaly["seed"],
            "operator_parameters": ref_anomaly["operator"]["parameters"],
            "camera_profile": ref_anomaly["camera"]["profile"],
            "physical_file_hashes": {
                name: ref_compiled["files"][name] for name in physical_files
            },
            "runtime_source_hashes": {
                key: ref_anomaly.get("provenance", {}).get(key)
                for key in ("backend_sha256", "operator_sha256", "script_sha256")
            },
            "nominal_measurements": ref_nominal["measurements"],
            "anomaly_measurements": ref_anomaly["measurements"],
        },
        "passed": all(checks.values()),
        "appearance_family_gate_closed": all(checks.values()),
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "interpretation": (
            "The unchanged O2 physical outcome replicated across three frozen photographic/PBR "
            "appearance families, including a held-out validation material. This closes only the "
            "three-family development gate; it is not a realistic A0-A7 result."
        ),
        "remaining_gates": [
            "independent_metric_geometry_and_soil_realizations",
            "multi_seed_dose_response_with_nonidentical_conditions",
            "physical_go2_fixture_camera_calibration",
            "independent_human_scene_review",
            "formal_schedule_binding",
        ],
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
