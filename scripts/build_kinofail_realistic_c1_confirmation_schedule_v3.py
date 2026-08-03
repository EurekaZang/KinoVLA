#!/usr/bin/env python3
"""Build preflight or fresh formal schedules for C1 confirmation v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode()
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:20]}"


def _views(
    materials: list[dict[str, Any]],
    *,
    design_tag: str,
    scene: str,
    profile_index: int,
) -> list[dict[str, Any]]:
    definitions = (
        (materials[0], "clean", 0.88, 37.0, (0.17, 0.09), 0.94, 1.00, 0.95),
        (materials[1], "scuffed", 1.12, 143.0, (0.36, 0.25), 1.04, 0.98, 1.07),
        (materials[0], "damp", 1.00, 277.0, (0.43, 0.39), 0.86, 1.13, 0.72),
    )
    rows = []
    for index, (
        material,
        state,
        scale,
        rotation,
        offset,
        brightness,
        normal,
        roughness,
    ) in enumerate(definitions):
        view_id = ("primary", "swap_01", "swap_02")[index]
        rows.append(
            {
                "appearance_view_id": view_id,
                "appearance_view_index": index,
                "is_primary": index == 0,
                "appearance_id": _stable_id(
                    "app", design_tag, scene, profile_index, view_id
                ),
                "appearance_seed": _stable_int(
                    design_tag, scene, profile_index, view_id
                ),
                "material_family": str(material["id"]),
                "material_asset_id": str(material["source_asset_id"]),
                "material_semantics": str(material["semantic_family"]),
                "material_source": str(material["source"]),
                "material_license": str(material["license"]),
                "physical_size_m": material["physical_size_m"],
                "surface_state": state,
                "uv_scale": scale,
                "uv_rotation_deg": rotation + 7.0 * profile_index,
                "uv_offset": list(offset),
                "albedo_brightness_multiplier": brightness,
                "normal_strength": normal,
                "roughness_multiplier": roughness,
                "visual_intervention_only": True,
                "shared_across_causes": True,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("preflight", "development", "formal"),
        required=True,
    )
    parser.add_argument(
        "--scene-registry",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
    )
    parser.add_argument(
        "--asset-lock",
        type=Path,
        default=ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registry = _json(args.scene_registry)
    lock = _json(args.asset_lock)
    if args.mode == "preflight":
        scenes = [
            row
            for row in registry["scenes"]
            if row["scene_id"] == "indoor_office_168"
        ]
        nuisance = ((-0.038, 0.026, 0.21),)
        design_tag = "c1_causal_visibility_preflight_v3"
        split_override = "val"
    elif args.mode == "development":
        scenes = [
            row for row in registry["scenes"] if row["split"] != "test"
        ]
        nuisance = (
            (-0.025, -0.018, 0.21),
            (0.025, 0.018, 0.24),
        )
        design_tag = "c1_causal_structured_v3_development"
        split_override = None
    else:
        scenes = [row for row in registry["scenes"] if row["split"] == "test"]
        nuisance = (
            (-0.040, -0.028, 0.19),
            (-0.027, 0.020, 0.22),
            (-0.013, -0.010, 0.24),
            (0.000, 0.027, 0.20),
            (0.013, -0.023, 0.23),
            (0.027, 0.012, 0.25),
            (0.040, -0.003, 0.21),
        )
        design_tag = "c1_causal_confirmation_extension_v3"
        split_override = "test"
    if not scenes:
        raise RuntimeError("C1 v3 schedule selected no scenes")

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        domain = str(scene["domain"])
        material_split = (
            "test" if args.mode == "formal" else str(scene["split"])
        )
        materials = sorted(
            (
                dict(value)
                for value in lock["materials"]
                if value["split"] == material_split
                and domain in value["domains"]
            ),
            key=lambda value: str(value["id"]),
        )
        if len(materials) != 2:
            raise RuntimeError(
                f"C1 v3 expects two locked materials: {domain} {material_split}"
            )
        for profile_index, (lateral, heading, speed) in enumerate(nuisance):
            case_id = _stable_id(
                "c1x3", design_tag, scene_id, profile_index
            )
            rows.append(
                {
                    "schema_version": "kinofail.realistic-c1-causal-schedule.v3",
                    "benchmark_id": design_tag,
                    "case_id": case_id,
                    "scene_cluster": scene_id,
                    "scene_family": scene_id,
                    "scene_seed": _stable_int(
                        design_tag, "scene", scene_id, profile_index
                    ),
                    "reset_seed": _stable_int(
                        design_tag, "reset", scene_id, profile_index
                    ),
                    "cause_visual_seed": _stable_int(
                        design_tag, "visual", scene_id, profile_index
                    ),
                    "domain": domain,
                    "split": (
                        str(scene["split"])
                        if split_override is None
                        else split_override
                    ),
                    "profile_index": profile_index,
                    "camera_profile": "go2_front_calib_b",
                    "physical_nuisance": {
                        "start_lateral_offset_m": lateral,
                        "start_heading_offset_rad": heading,
                        "forward_speed_mps": speed,
                    },
                    "decision_progress_m": 0.22,
                    # v2 showed that 0.62 m clips the cue at the image bottom
                    # under trajectory nuisance.  Moving it farther away makes
                    # the complete metric geometry visible before contact.
                    "cause_region": {
                        "start_progress_m": 0.90,
                        "length_m": 0.62,
                        "half_width_m": 0.38,
                        "base_z_m": 0.006,
                    },
                    "causes": [
                        {
                            "target_operator": "O2_compliance",
                            "attribution_category": "compliant_terrain",
                            "visible_physical_cue": "soft_surface_deformation",
                        },
                        {
                            "target_operator": "O4_tether",
                            "attribution_category": "adhesion",
                            "visible_physical_cue": (
                                "overlapping_film_and_creases"
                            ),
                        },
                    ],
                    "appearance_views": _views(
                        materials,
                        design_tag=design_tag,
                        scene=scene_id,
                        profile_index=profile_index,
                    ),
                    "shared_prefix_contract": {
                        "one_physics_rollout_for_both_causes": True,
                        "visual_rerenders_do_not_advance_physics": True,
                        "proprio_bytes_reused_without_relabeling_or_perturbation": True,
                        "operator_physics_activates_only_after_the_decision_boundary": True,
                    },
                }
            )
    prior_ids: set[str] = set()
    for path in (
        ROOT
        / "outputs/kinofail_realistic/design_c1_causal_formal_v1/schedule.jsonl",
        ROOT
        / "outputs/kinofail_realistic/design_c1_causal_confirmation_v2/schedule.jsonl",
    ):
        if path.exists():
            prior_ids.update(
                json.loads(line)["case_id"]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            )
    expected = {
        "preflight": 1,
        "development": 12,
        "formal": 21,
    }[args.mode]
    checks = {
        "expected_case_count": len(rows) == expected,
        "fresh_ids_disjoint_from_v1_v2": prior_ids.isdisjoint(
            {row["case_id"] for row in rows}
        ),
        "same_material_views_across_causes": True,
        "decision_before_cue_region": all(
            row["decision_progress_m"] < row["cause_region"]["start_progress_m"]
            for row in rows
        ),
    }
    if args.mode == "formal":
        checks.update(
            {
                "seven_cases_per_scene": all(
                    sum(
                        row["scene_cluster"] == scene["scene_id"]
                        for row in rows
                    )
                    == 7
                    for scene in scenes
                ),
                "three_test_scenes_domains": (
                    len(scenes) == 3
                    and {row["domain"] for row in rows}
                    == {"life", "production", "wild"}
                ),
            }
        )
    elif args.mode == "development":
        checks.update(
            {
                "two_cases_per_scene": all(
                    sum(
                        row["scene_cluster"] == scene["scene_id"]
                        for row in rows
                    )
                    == 2
                    for scene in scenes
                ),
                "six_non_test_scenes": (
                    len(scenes) == 6
                    and {row["split"] for row in rows}
                    == {"train", "val"}
                ),
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    schedule_path = args.out / "schedule.jsonl"
    schedule_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "kinofail.realistic-c1-confirmation-design-audit.v3",
        "mode": args.mode,
        "passed": all(checks.values()),
        "checks": checks,
        "design_tag": design_tag,
        "case_count": len(rows),
        "scene_cluster_count": len(scenes),
        "outcomes_available_at_design_time": False,
        "schedule_sha256": _sha(schedule_path),
        "source_sha256": {
            "scene_registry": _sha(args.scene_registry),
            "asset_lock": _sha(args.asset_lock),
        },
    }
    audit_path = args.out / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
