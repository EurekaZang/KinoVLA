#!/usr/bin/env python3
"""Build fresh O7/O8 physical pairs for the C2 v3 T3 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_kinofail_realistic_c1_confirmation_schedule_v3 import _views


ROOT = Path(__file__).resolve().parents[1]
DESIGN_TAG = "c2_bidirectional_confirmation_v3_t3"
SEVERITIES = (
    "moderate",
    "severe",
    "moderate",
    "severe",
    "moderate",
)
PARAMETERS = {
    "O7_visual_remap": {
        "category": "low_friction",
        "physical_realization": "appearance_physics_swap",
        "geometry_profile": "same_surface_counterfactual",
        "nominal": {
            "mu_s": 0.80,
            "mu_d": 0.60,
            "depth_bias_m": 0.0,
        },
        "moderate": {
            "mu_s": 0.18,
            "mu_d": 0.13,
            "depth_bias_m": 0.18,
        },
        "severe": {
            "mu_s": 0.09,
            "mu_d": 0.06,
            "depth_bias_m": 0.32,
        },
    },
    "O8_invisible_collider": {
        "category": "invisible_obstacle",
        "physical_realization": "transparent_acrylic",
        "geometry_profile": "transparent_panel",
        "moderate": {
            "collision_enabled": 1.0,
            "obstacle_height_m": 0.38,
            "optical_transmission": 0.94,
        },
        "severe": {
            "collision_enabled": 1.0,
            "obstacle_height_m": 0.38,
            "optical_transmission": 0.98,
        },
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return (
        int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        & 0x7FFFFFFF
    )


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode()
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:20]}"


def _outputs(
    domain: str, operator: str, episode_id: str
) -> dict[str, Any]:
    root = f"confirmation_v3/{domain}/{operator}/{episode_id}"
    return {
        "episode_manifest": f"{root}/manifest.json",
        "rgb": f"{root}/rgb/",
        "rgb_views": {
            "primary": f"{root}/rgb/",
            "swap_01": f"{root}/rgb_views/swap_01/",
            "swap_02": f"{root}/rgb_views/swap_02/",
        },
        "proprio": f"{root}/proprio.npz",
        "telemetry": f"{root}/privileged.jsonl",
        "depth_optional": f"{root}/depth/",
    }


def _record(
    *,
    scene: dict[str, Any],
    profile_index: int,
    severity: str,
    operator: str,
    condition: str,
    group_id: str,
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    spec = PARAMETERS[operator]
    active = condition == "anomaly"
    if operator == "O8_invisible_collider":
        physics = dict(spec[severity])
        if not active:
            physics["collision_enabled"] = 0.0
    else:
        physics = dict(
            spec[severity] if active else spec["nominal"]
        )
    primary = dict(views[0])
    episode_id = f"{group_id}_{'anomaly' if active else 'nominal'}"
    normalized_views = [
        {
            **view,
            "render_tier": "structured_pbr",
            "triplanar": True,
        }
        for view in views
    ]
    return {
        "schema_version": "kinofail.realistic-design.v3",
        "benchmark_id": DESIGN_TAG,
        "design_mode": "formal_confirmation",
        "artifact_state": "planned",
        "evaluation_eligible": True,
        "episode_id": episode_id,
        "counterfactual_group_id": group_id,
        "condition": condition,
        "active_operator": operator if active else None,
        "target_operator": operator,
        "attribution_category": (
            str(spec["category"]) if active else "nominal"
        ),
        "severity_id": severity,
        "severity_rank": 2 if severity == "moderate" else 3,
        "physics_parameters": physics,
        "physical_realization": str(
            spec["physical_realization"]
        ),
        "geometry_profile": str(spec["geometry_profile"]),
        "geometry_id": (
            f"{scene['scene_id']}::{spec['geometry_profile']}"
        ),
        "scene_family": str(scene["scene_id"]),
        "scene_source": str(scene["source"]),
        "scene_seed": _stable_int(
            DESIGN_TAG, "scene", scene["scene_id"], profile_index
        ),
        "operator_seed": _stable_int(
            DESIGN_TAG,
            "operator",
            scene["scene_id"],
            profile_index,
            operator,
        ),
        "domain": str(scene["domain"]),
        "split": "test",
        "camera_profile": "go2_front_calib_b",
        "appearance_view_count": 3,
        "appearance_views": normalized_views,
        "appearance_id": primary["appearance_id"],
        "appearance_seed": primary["appearance_seed"],
        "material_family": primary["material_family"],
        "material_asset_id": primary["material_asset_id"],
        "material_semantics": primary["material_semantics"],
        "material_source": primary["material_source"],
        "material_license": primary["material_license"],
        "material_physical_size_m": primary["physical_size_m"],
        "surface_state": primary["surface_state"],
        "uv_scale": primary["uv_scale"],
        "uv_rotation_deg": primary["uv_rotation_deg"],
        "uv_offset": primary["uv_offset"],
        "albedo_brightness_multiplier": primary[
            "albedo_brightness_multiplier"
        ],
        "normal_strength": primary["normal_strength"],
        "roughness_multiplier": primary["roughness_multiplier"],
        "render_tier": "structured_pbr",
        "triplanar": True,
        "texture_swap_group_id": _stable_id(
            "ts",
            DESIGN_TAG,
            scene["scene_id"],
            profile_index,
            operator,
            condition,
        ),
        "required_outputs": _outputs(
            str(scene["domain"]), operator, episode_id
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-registry",
        type=Path,
        default=ROOT
        / "configs/data"
        / "kinofail_realistic_c2_v3_confirmation_scene_registry.json",
    )
    parser.add_argument(
        "--asset-lock",
        type=Path,
        default=ROOT
        / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registry = _json(args.scene_registry)
    lock = _json(args.asset_lock)
    scenes = [dict(row) for row in registry["scenes"]]
    if (
        len(scenes) != 3
        or {str(row["domain"]) for row in scenes}
        != {"life", "production", "wild"}
    ):
        raise RuntimeError(
            "C2 v3 T3 requires three fresh held-out domains"
        )

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        domain = str(scene["domain"])
        materials = sorted(
            (
                dict(value)
                for value in lock["materials"]
                if value["split"] == "test"
                and domain in value["domains"]
            ),
            key=lambda value: str(value["id"]),
        )
        if len(materials) != 2:
            raise RuntimeError(
                f"C2 v3 T3 expects two test materials: {domain}"
            )
        for profile_index, severity in enumerate(SEVERITIES):
            case_id = _stable_id(
                "c2t3x3", DESIGN_TAG, scene_id, profile_index
            )
            views = _views(
                materials,
                design_tag=DESIGN_TAG,
                scene=scene_id,
                profile_index=profile_index,
            )
            groups: dict[str, str] = {}
            for operator in (
                "O7_visual_remap",
                "O8_invisible_collider",
            ):
                group_id = _stable_id(
                    "cf",
                    DESIGN_TAG,
                    scene_id,
                    profile_index,
                    operator,
                )
                groups[operator] = group_id
                for condition in (
                    "nominal_counterfactual",
                    "anomaly",
                ):
                    rows.append(
                        _record(
                            scene=scene,
                            profile_index=profile_index,
                            severity=severity,
                            operator=operator,
                            condition=condition,
                            group_id=group_id,
                            views=views,
                        )
                    )
            cases.append(
                {
                    "case_id": case_id,
                    "scene_cluster": scene_id,
                    "domain": domain,
                    "profile_index": profile_index,
                    "severity": severity,
                    "o7_source_physics_group_id": groups[
                        "O7_visual_remap"
                    ],
                    "o8_source_physics_group_id": groups[
                        "O8_invisible_collider"
                    ],
                    "shared_visual_source": (
                        "O7 anomaly RTX sequence is copied byte-for-byte "
                        "to the O8 attribution candidate at extraction"
                    ),
                    "visual_intervention_advances_physics": False,
                }
            )

    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(
            str(row["counterfactual_group_id"]), []
        ).append(row)
    development_registry = _json(
        ROOT
        / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
    )
    development_scenes = {
        str(row["scene_id"])
        for row in development_registry["scenes"]
    }
    confirmation_scenes = {
        str(row["scene_cluster"]) for row in cases
    }
    checks = {
        "fifteen_fresh_matched_cases": (
            len(cases) == 15
            and len({str(row["case_id"]) for row in cases}) == 15
        ),
        "thirty_complete_physics_pairs": (
            len(by_group) == 30
            and len(rows) == 60
            and all(
                len(value) == 2
                and {
                    str(row["condition"]) for row in value
                }
                == {"nominal_counterfactual", "anomaly"}
                for value in by_group.values()
            )
        ),
        "five_cases_per_confirmation_scene": all(
            sum(
                row["scene_cluster"] == scene["scene_id"]
                for row in cases
            )
            == 5
            for scene in scenes
        ),
        "three_fresh_scene_instances_and_domains": (
            len(confirmation_scenes) == 3
            and confirmation_scenes.isdisjoint(development_scenes)
            and {str(row["domain"]) for row in cases}
            == {"life", "production", "wild"}
        ),
        "same_scheduled_views_across_o7_o8_within_case": True,
        "moderate_and_severe_physical_effects": (
            {str(row["severity"]) for row in cases}
            == {"moderate", "severe"}
        ),
        "outcomes_unavailable_at_design_time": True,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    schedule_path = args.out / "schedule.jsonl"
    schedule_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in sorted(
                rows,
                key=lambda row: (
                    str(row["scene_family"]),
                    str(row["target_operator"]),
                    str(row["counterfactual_group_id"]),
                    str(row["condition"]),
                ),
            )
        ),
        encoding="utf-8",
    )
    case_path = args.out / "case_schedule.jsonl"
    case_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in cases
        ),
        encoding="utf-8",
    )
    audit = {
        "schema_version": (
            "kinofail.realistic-c2-t3-design-audit.v3"
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "design_tag": DESIGN_TAG,
        "case_count": len(cases),
        "counterfactual_pair_count": len(by_group),
        "physical_episode_count": len(rows),
        "scene_cluster_count": len(scenes),
        "schedule_sha256": _sha(schedule_path),
        "case_schedule_sha256": _sha(case_path),
        "source_sha256": {
            "scene_registry": _sha(args.scene_registry),
            "asset_lock": _sha(args.asset_lock),
        },
    }
    (args.out / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
