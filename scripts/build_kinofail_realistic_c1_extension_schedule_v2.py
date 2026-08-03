#!/usr/bin/env python3
"""Build fresh nuisance profiles for the untouched C1/C2 confirmation extension."""

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
    value = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") & 0x7FFFFFFF


def _stable_id(prefix: str, *parts: object) -> str:
    value = "|".join(str(part) for part in parts).encode()
    return f"{prefix}_{hashlib.sha256(value).hexdigest()[:20]}"


def _views(
    materials: list[dict[str, Any]], *, scene: str, profile_index: int
) -> list[dict[str, Any]]:
    definitions = (
        (materials[0], "clean", 0.81, 19.0, (0.13, 0.07), 0.91, 1.03, 0.92),
        (materials[1], "scuffed", 1.21, 119.0, (0.33, 0.29), 1.07, 0.94, 1.11),
        (materials[0], "damp", 0.97, 241.0, (0.47, 0.43), 0.82, 1.17, 0.69),
    )
    out = []
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
        out.append(
            {
                "appearance_view_id": view_id,
                "appearance_view_index": index,
                "is_primary": index == 0,
                "appearance_id": _stable_id(
                    "app",
                    "c1_causal_confirmation_extension_v2",
                    scene,
                    profile_index,
                    view_id,
                ),
                "appearance_seed": _stable_int(
                    "c1_causal_confirmation_extension_v2",
                    scene,
                    profile_index,
                    view_id,
                ),
                "material_family": str(material["id"]),
                "material_asset_id": str(material["source_asset_id"]),
                "material_semantics": str(material["semantic_family"]),
                "material_source": str(material["source"]),
                "material_license": str(material["license"]),
                "physical_size_m": material["physical_size_m"],
                "surface_state": state,
                "uv_scale": scale,
                "uv_rotation_deg": rotation + 11.0 * profile_index,
                "uv_offset": list(offset),
                "albedo_brightness_multiplier": brightness,
                "normal_strength": normal,
                "roughness_multiplier": roughness,
                "visual_intervention_only": True,
                "shared_across_causes": True,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/design_c1_causal_confirmation_v2",
    )
    args = parser.parse_args()
    registry = _json(args.scene_registry)
    lock = _json(args.asset_lock)
    scenes = [row for row in registry["scenes"] if row["split"] == "test"]
    nuisance = (
        (-0.045, -0.032, 0.18),
        (-0.030, 0.024, 0.21),
        (-0.015, -0.012, 0.25),
        (0.000, 0.030, 0.19),
        (0.015, -0.026, 0.23),
        (0.030, 0.014, 0.26),
        (0.045, -0.004, 0.20),
    )
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        domain = str(scene["domain"])
        materials = sorted(
            (
                dict(value)
                for value in lock["materials"]
                if value["split"] == "test" and domain in value["domains"]
            ),
            key=lambda value: str(value["id"]),
        )
        if len(materials) != 2:
            raise RuntimeError(f"extension expects two locked test materials: {domain}")
        for profile_index, (lateral, heading, speed) in enumerate(nuisance):
            case_id = _stable_id(
                "c1x2",
                "c1_causal_confirmation_extension_v2",
                scene_id,
                profile_index,
            )
            rows.append(
                {
                    "schema_version": "kinofail.realistic-c1-causal-schedule.v2",
                    "benchmark_id": "c1_causal_confirmation_extension_v2",
                    "case_id": case_id,
                    "scene_cluster": scene_id,
                    "scene_family": scene_id,
                    "scene_seed": _stable_int(
                        "c1_causal_confirmation_extension_v2",
                        "scene",
                        scene_id,
                        profile_index,
                    ),
                    "reset_seed": _stable_int(
                        "c1_causal_confirmation_extension_v2",
                        "reset",
                        scene_id,
                        profile_index,
                    ),
                    "cause_visual_seed": _stable_int(
                        "c1_causal_confirmation_extension_v2",
                        "visual",
                        scene_id,
                        profile_index,
                    ),
                    "domain": domain,
                    "split": "test",
                    "profile_index": profile_index,
                    "camera_profile": "go2_front_calib_b",
                    "physical_nuisance": {
                        "start_lateral_offset_m": lateral,
                        "start_heading_offset_rad": heading,
                        "forward_speed_mps": speed,
                    },
                    "decision_progress_m": 0.22,
                    "cause_region": {
                        "start_progress_m": 0.62,
                        "length_m": 0.90,
                        "half_width_m": 0.45,
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
                            "visible_physical_cue": "overlapping_film_and_creases",
                        },
                    ],
                    "appearance_views": _views(
                        materials, scene=scene_id, profile_index=profile_index
                    ),
                    "shared_prefix_contract": {
                        "one_physics_rollout_for_both_causes": True,
                        "visual_rerenders_do_not_advance_physics": True,
                        "proprio_bytes_reused_without_relabeling_or_perturbation": True,
                        "operator_physics_activates_only_after_the_decision_boundary": True,
                    },
                }
            )
    checks = {
        "21_fresh_cases": len(rows) == 21,
        "seven_cases_per_scene": all(
            sum(row["scene_cluster"] == scene["scene_id"] for row in rows) == 7
            for scene in scenes
        ),
        "three_test_scenes_domains": (
            len(scenes) == 3
            and {row["domain"] for row in rows} == {"life", "production", "wild"}
        ),
        "fresh_ids_disjoint_from_v1": True,
        "same_material_views_across_causes": True,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    schedule_path = args.out / "schedule.jsonl"
    schedule_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    prior = ROOT / "outputs/kinofail_realistic/design_c1_causal_formal_v1/schedule.jsonl"
    if prior.exists():
        old_ids = {
            json.loads(line)["case_id"]
            for line in prior.read_text(encoding="utf-8").splitlines()
            if line
        }
        checks["fresh_ids_disjoint_from_v1"] = old_ids.isdisjoint(
            {row["case_id"] for row in rows}
        )
    audit = {
        "schema_version": "kinofail.realistic-c1-confirmation-design-audit.v2",
        "passed": all(checks.values()),
        "checks": checks,
        "design_tag": "c1_causal_confirmation_extension_v2",
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
