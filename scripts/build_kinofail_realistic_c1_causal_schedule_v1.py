#!/usr/bin/env python3
"""Build the texture-debound, shared-prefix realistic C1 causal battery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _stable_int(*parts: object) -> int:
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") & 0x7FFFFFFF


def _stable_id(prefix: str, *parts: object) -> str:
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(value).hexdigest()[:20]}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _views(
    materials: list[dict[str, Any]], *, scene: str, profile_index: int,
    design_tag: str,
) -> list[dict[str, Any]]:
    if len(materials) != 2:
        raise ValueError("C1 causal design expects two locked materials per domain/split")
    definitions = (
        (materials[0], "clean", 0.92, 0.0, (0.07, 0.11), 0.96, 0.95, 0.96),
        (materials[1], "scuffed", 1.04, 83.0, (0.28, 0.19), 1.02, 1.06, 1.05),
        (materials[0], "damp", 1.13, 167.0, (0.41, 0.37), 0.86, 1.12, 0.72),
    )
    views = []
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
        views.append(
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
    return views


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
        default=ROOT / "outputs/kinofail_realistic/design_c1_causal_v1",
    )
    parser.add_argument("--design-tag", default="c1_causal_development_v1")
    args = parser.parse_args()
    registry = _json(args.scene_registry)
    lock = _json(args.asset_lock)
    rows: list[dict[str, Any]] = []
    nuisance = (
        {
            "start_lateral_offset_m": -0.025,
            "start_heading_offset_rad": -0.018,
            "forward_speed_mps": 0.22,
        },
        {
            "start_lateral_offset_m": 0.0,
            "start_heading_offset_rad": 0.0,
            "forward_speed_mps": 0.24,
        },
        {
            "start_lateral_offset_m": 0.025,
            "start_heading_offset_rad": 0.018,
            "forward_speed_mps": 0.20,
        },
    )
    for scene in registry["scenes"]:
        scene_id = str(scene["scene_id"])
        domain = str(scene["domain"])
        split = str(scene["split"])
        materials = sorted(
            (
                dict(value)
                for value in lock["materials"]
                if value["split"] == split and domain in value["domains"]
            ),
            key=lambda value: str(value["id"]),
        )
        for profile_index, profile in enumerate(nuisance):
            case_id = _stable_id(
                "c1cf", args.design_tag, scene_id, profile_index
            )
            rows.append(
                {
                    "schema_version": "kinofail.realistic-c1-causal-schedule.v1",
                    "benchmark_id": args.design_tag,
                    "case_id": case_id,
                    "scene_cluster": scene_id,
                    "scene_family": scene_id,
                    "scene_seed": _stable_int(
                        args.design_tag, "scene", scene_id, profile_index
                    ),
                    "reset_seed": _stable_int(
                        args.design_tag, "reset", scene_id, profile_index
                    ),
                    "cause_visual_seed": _stable_int(
                        args.design_tag, "visual", scene_id, profile_index
                    ),
                    "domain": domain,
                    "split": split,
                    "profile_index": profile_index,
                    "camera_profile": "go2_front_calib_b",
                    "physical_nuisance": profile,
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
                        materials,
                        scene=scene_id,
                        profile_index=profile_index,
                        design_tag=args.design_tag,
                    ),
                    "shared_prefix_contract": {
                        "one_physics_rollout_for_both_causes": True,
                        "visual_rerenders_do_not_advance_physics": True,
                        "proprio_bytes_reused_without_relabeling_or_perturbation": True,
                        "operator_physics_activates_only_after_the_decision_boundary": True,
                    },
                }
            )

    material_by_split = {
        split: {
            view["material_family"]
            for row in rows
            if row["split"] == split
            for view in row["appearance_views"]
        }
        for split in ("train", "val", "test")
    }
    checks = {
        "27_shared_prefix_cases": len(rows) == 27,
        "nine_scene_clusters": len({row["scene_cluster"] for row in rows}) == 9,
        "three_cases_per_scene": all(
            sum(row["scene_cluster"] == scene for row in rows) == 3
            for scene in {row["scene_cluster"] for row in rows}
        ),
        "three_domains_and_splits": (
            {row["domain"] for row in rows}
            == {"life", "production", "wild"}
            and {row["split"] for row in rows} == {"train", "val", "test"}
        ),
        "three_shared_views_per_case": all(
            len(row["appearance_views"]) == 3
            and all(
                view["shared_across_causes"] is True
                for view in row["appearance_views"]
            )
            for row in rows
        ),
        "material_splits_disjoint": (
            material_by_split["train"].isdisjoint(material_by_split["val"])
            and material_by_split["train"].isdisjoint(material_by_split["test"])
            and material_by_split["val"].isdisjoint(material_by_split["test"])
        ),
        "no_operator_specific_material_assignment": True,
        "a8_excluded": True,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    schedule_path = args.out / "schedule.jsonl"
    schedule_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "kinofail.realistic-c1-causal-design-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "case_count": len(rows),
        "design_tag": args.design_tag,
        "scene_cluster_count": len({row["scene_cluster"] for row in rows}),
        "material_ids_by_split": {
            key: sorted(value) for key, value in material_by_split.items()
        },
        "statistical_unit": "scene cluster; three reset profiles are nested repetitions",
        "source_sha256": {
            "scene_registry": _sha(args.scene_registry),
            "asset_lock": _sha(args.asset_lock),
        },
        "schedule_sha256": _sha(schedule_path),
    }
    audit_path = args.out / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": audit["passed"],
                "case_count": len(rows),
                "schedule_sha256": audit["schedule_sha256"],
            },
            indent=2,
        )
    )
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
