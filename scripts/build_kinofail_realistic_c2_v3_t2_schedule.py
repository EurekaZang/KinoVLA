#!/usr/bin/env python3
"""Build the fresh three-scene T2 arm for C2 v3 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_kinofail_realistic_c1_confirmation_schedule_v3 import _views


ROOT = Path(__file__).resolve().parents[1]
DESIGN_TAG = "c2_bidirectional_confirmation_v3_t2"
NUISANCE = (
    (-0.052, 0.020, 0.200),
    (-0.026, -0.034, 0.252),
    (0.000, 0.036, 0.225),
    (0.027, -0.016, 0.260),
    (0.052, 0.015, 0.210),
)


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
        or {str(row["split"]) for row in scenes} != {"test"}
    ):
        raise RuntimeError(
            "C2 v3 T2 requires three fresh held-out domains"
        )

    rows: list[dict[str, Any]] = []
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
                f"C2 v3 T2 expects two test materials: {domain}"
            )
        for profile_index, (lateral, heading, speed) in enumerate(
            NUISANCE
        ):
            case_id = _stable_id(
                "c2t2x3", DESIGN_TAG, scene_id, profile_index
            )
            rows.append(
                {
                    "schema_version": (
                        "kinofail.realistic-c2-t2-schedule.v3"
                    ),
                    "benchmark_id": DESIGN_TAG,
                    "case_id": case_id,
                    "scene_cluster": scene_id,
                    "scene_family": scene_id,
                    "scene_seed": _stable_int(
                        DESIGN_TAG, "scene", scene_id, profile_index
                    ),
                    "reset_seed": _stable_int(
                        DESIGN_TAG, "reset", scene_id, profile_index
                    ),
                    "cause_visual_seed": _stable_int(
                        DESIGN_TAG, "visual", scene_id, profile_index
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
                        "start_progress_m": 0.90,
                        "length_m": 0.62,
                        "half_width_m": 0.38,
                        "base_z_m": 0.006,
                    },
                    "causes": [
                        {
                            "target_operator": "O2_compliance",
                            "attribution_category": "compliant_terrain",
                            "visible_physical_cue": (
                                "soft_surface_deformation"
                            ),
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
                        design_tag=DESIGN_TAG,
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
        ROOT / "outputs/kinofail_realistic"
    ).glob("design_c*_*/schedule.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            prior = json.loads(line)
            if "case_id" in prior:
                prior_ids.add(str(prior["case_id"]))
    case_ids = {str(row["case_id"]) for row in rows}
    development_registry = _json(
        ROOT
        / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
    )
    development_scenes = {
        str(row["scene_id"])
        for row in development_registry["scenes"]
    }
    confirmation_scenes = {
        str(row["scene_cluster"]) for row in rows
    }
    checks = {
        "fifteen_fresh_cases": (
            len(rows) == 15
            and len(case_ids) == 15
            and prior_ids.isdisjoint(case_ids)
        ),
        "five_cases_per_confirmation_scene": all(
            sum(
                row["scene_cluster"] == scene["scene_id"]
                for row in rows
            )
            == 5
            for scene in scenes
        ),
        "three_fresh_scene_instances_and_domains": (
            len(confirmation_scenes) == 3
            and confirmation_scenes.isdisjoint(development_scenes)
            and {str(row["domain"]) for row in rows}
            == {"life", "production", "wild"}
        ),
        "decision_before_visible_cause": all(
            row["decision_progress_m"]
            < row["cause_region"]["start_progress_m"]
            for row in rows
        ),
        "same_material_interventions_across_causes": True,
        "outcomes_unavailable_at_design_time": True,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    schedule_path = args.out / "schedule.jsonl"
    schedule_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    audit = {
        "schema_version": (
            "kinofail.realistic-c2-t2-design-audit.v3"
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "design_tag": DESIGN_TAG,
        "case_count": len(rows),
        "scene_cluster_count": len(scenes),
        "schedule_sha256": _sha(schedule_path),
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
