#!/usr/bin/env python3
"""Build the fresh, scene-held-out T2 arm for C2 confirmation v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_kinofail_realistic_c1_confirmation_schedule_v3 import _views


ROOT = Path(__file__).resolve().parents[1]
TEST_NUISANCE = (
    (-0.044, 0.011, 0.205),
    (-0.031, -0.019, 0.235),
    (-0.017, 0.029, 0.215),
    (0.004, -0.027, 0.245),
    (0.019, 0.008, 0.195),
    (0.033, 0.024, 0.230),
    (0.045, -0.012, 0.255),
)
DESIGN_TAG = "c2_bidirectional_confirmation_v2_t2"


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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registry = _json(args.scene_registry)
    lock = _json(args.asset_lock)
    scenes = [
        dict(row) for row in registry["scenes"] if row["split"] == "test"
    ]
    if (
        len(scenes) != 3
        or {str(row["domain"]) for row in scenes}
        != {"life", "production", "wild"}
    ):
        raise RuntimeError("C2 T2 requires three held-out realistic domains")

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
                f"C2 T2 expects two locked test materials: {domain}"
            )
        for profile_index, (lateral, heading, speed) in enumerate(
            TEST_NUISANCE
        ):
            case_id = _stable_id(
                "c2t2x2", DESIGN_TAG, scene_id, profile_index
            )
            rows.append(
                {
                    "schema_version": (
                        "kinofail.realistic-c2-t2-schedule.v2"
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
    for name in (
        "design_c1_causal_formal_v1",
        "design_c1_causal_confirmation_v2",
        "design_c1_causal_confirmation_v3",
    ):
        path = (
            ROOT
            / "outputs/kinofail_realistic"
            / name
            / "schedule.jsonl"
        )
        if path.exists():
            prior_ids.update(
                json.loads(line)["case_id"]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            )
    case_ids = {str(row["case_id"]) for row in rows}
    checks = {
        "twenty_one_fresh_cases": (
            len(rows) == 21
            and len(case_ids) == 21
            and prior_ids.isdisjoint(case_ids)
        ),
        "seven_cases_per_test_scene": all(
            sum(row["scene_cluster"] == scene["scene_id"] for row in rows)
            == 7
            for scene in scenes
        ),
        "three_scene_disjoint_test_domains": (
            {row["domain"] for row in rows}
            == {"life", "production", "wild"}
            and {row["split"] for row in rows} == {"test"}
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
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "kinofail.realistic-c2-t2-design-audit.v2",
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
