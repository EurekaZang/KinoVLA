#!/usr/bin/env python3
"""Compile and nominally admit one frozen confirmatory wild scene.

The five photographic HDRIs are distant illumination/background assets only.
Two deterministic, seed-separated metric near-field geometries are authored
per HDRI.  Admission is model blind and uses only asset integrity, compiled USD
checks, and a nominal articulated-Go2 rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT / "configs/data/kinofail_forest_hybrid_composition_dev_v24.json"
)
HDRI_AUDITOR = ROOT / "scripts/audit_forest_hdri.py"
COMPILER = ROOT / "scripts/compile_kinofail_forest_hybrid_scene.py"
NOMINAL_COLLECTOR = (
    ROOT / "scripts/isaac_collect_realistic_route_operator_lane_v6.py"
)
POLICY = ROOT / "outputs/locomotion/realistic_route_v1/policy.pt"
ISAAC_PYTHON = Path("/home/eureka/nvidia/isaacsim/python.sh")
ISAAC_SETUP = Path("/home/eureka/nvidia/isaacsim/setup_conda_env.sh")
ISAACLAB_PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _run(command: list[str]) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(ROOT)
        if not prior_pythonpath
        else str(ROOT) + os.pathsep + prior_pythonpath
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    return {
        "command": command,
        "started_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "returncode": int(completed.returncode),
        "passed": completed.returncode == 0,
    }


def _candidate(config: dict[str, Any], scene_id: str) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in config["wild_scenes"]
        if str(row["scene_id"]) == scene_id
    ]
    if len(matches) != 1:
        raise KeyError(f"scene_id must resolve exactly once: {scene_id}")
    return matches[0]


def _material(lock: dict[str, Any], material_id: str) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in lock.get("materials", [])
        if str(row["id"]) == material_id
    ]
    if len(matches) != 1:
        raise KeyError(f"material must resolve exactly once: {material_id}")
    return matches[0]


def _expected_qa_materials(
    config: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[str, str]:
    stream = [
        row for row in config["wild_scenes"] if str(row["domain"]) == "wild"
    ]
    index = next(
        position
        for position, row in enumerate(stream)
        if str(row["scene_id"]) == str(candidate["scene_id"])
    )
    return (
        f"confirm_v1_wild_pbr_{index:02d}",
        f"confirm_v1_wild_pbr_{(index + 5) % 10:02d}",
    )


def _prop_transforms(seed: int, route_y: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    identifiers = (
        "roots_left_00",
        "roots_right_00",
        "roots_left_01",
        "rocks_left_00",
        "rocks_right_00",
        "rocks_right_01",
        "stump_left_00",
        "stump_right_00",
    )
    values: dict[str, Any] = {}
    for index, name in enumerate(identifiers):
        left = "left" in name
        x = -0.8 + 0.95 * index + float(rng.uniform(-0.22, 0.22))
        y = route_y + (3.0 if left else -3.0) + float(
            rng.uniform(-0.38, 0.38)
        )
        z = 0.10 if "stump" in name else 0.08 if "rocks" in name else 0.0
        values[name] = {
            "xyz_m": [round(x, 6), round(y, 6), z],
            "rotation_xyz_deg": [
                0.0,
                0.0,
                round(float(rng.uniform(0.0, 360.0)), 6),
            ],
        }
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_unified_confirmatory_scene_candidates_v1.json",
    )
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--hdri-lock", type=Path, required=True)
    parser.add_argument("--material-lock", type=Path, required=True)
    parser.add_argument("--route-material-id", required=True)
    parser.add_argument("--surrounding-material-id", required=True)
    args = parser.parse_args()

    scene_config_path = args.scene_config.resolve()
    hdri_lock_path = args.hdri_lock.resolve()
    material_lock_path = args.material_lock.resolve()
    for runtime_path in (ISAAC_PYTHON, ISAAC_SETUP, ISAACLAB_PYTHON):
        if not runtime_path.is_file():
            raise FileNotFoundError(runtime_path)
    scene_config = _json(scene_config_path)
    candidate = _candidate(scene_config, args.scene_id)
    expected_route, expected_surrounding = _expected_qa_materials(
        scene_config, candidate
    )
    if (
        args.route_material_id != expected_route
        or args.surrounding_material_id != expected_surrounding
    ):
        raise RuntimeError(
            "wild QA materials differ from the F0 rule: "
            f"expected {expected_route}/{expected_surrounding}"
        )
    hdri_lock = _json(hdri_lock_path)
    material_lock = _json(material_lock_path)
    route_material = _material(material_lock, args.route_material_id)
    surrounding_material = _material(
        material_lock, args.surrounding_material_id
    )
    for material in (route_material, surrounding_material):
        if "wild" not in material.get("domains", []):
            raise ValueError(f"non-wild material assigned to wild scene: {material['id']}")
    hdri_matches = [
        dict(row)
        for row in hdri_lock.get("assets", [])
        if str(row["id"]) == str(candidate["hdri_id"])
    ]
    if len(hdri_matches) != 1:
        raise KeyError(f"HDRI absent from lock: {candidate['hdri_id']}")
    hdri = hdri_matches[0]
    panorama = Path(hdri["file"]["path"]).resolve()
    if _sha256(panorama) != str(hdri["file"]["sha256"]):
        raise ValueError("HDRI content differs from lock")

    candidate_root = (
        ROOT
        / "outputs/kinofail_confirmatory_v1/scenes/wild"
        / args.scene_id
    ).resolve()
    if candidate_root.exists():
        raise FileExistsError(
            f"refusing to overwrite confirmatory wild scene: {candidate_root}"
        )
    candidate_root.mkdir(parents=True, exist_ok=False)
    preflight = panorama.parent / "preflight.json"
    ledger: list[dict[str, Any]] = []
    if not preflight.exists():
        ledger.append(
            {
                "stage": "hdri_preflight",
                **_run(
                    [
                        sys.executable,
                        str(HDRI_AUDITOR),
                        "--lock",
                        str(hdri_lock_path),
                        "--asset-id",
                        str(candidate["hdri_id"]),
                        "--output",
                        str(preflight),
                    ]
                ),
            }
        )
    else:
        audit = _json(preflight)
        ledger.append(
            {
                "stage": "hdri_preflight_reuse",
                "command": [],
                "returncode": 0 if audit.get("passed") is True else 2,
                "passed": audit.get("passed") is True,
                "started_utc": datetime.now(UTC).isoformat(),
                "completed_utc": datetime.now(UTC).isoformat(),
            }
        )
    failure_stage = next(
        (str(row["stage"]) for row in ledger if row["passed"] is not True),
        None,
    )

    seed = int(candidate["metric_geometry_seed"])
    variant = int(candidate["route_variant"])
    rng = np.random.default_rng(seed)
    route_y = round(float(rng.uniform(-0.38, 0.38)), 6)
    route_length = 5.4
    phase = round(float(rng.uniform(0.0, 2.0 * math.pi)), 6)
    rotation = round(float(rng.uniform(0.0, 360.0)), 6)
    output_dir = candidate_root / "composition"
    composition_path = candidate_root / "composition_config.json"
    composition = {
        "schema_version": "kinofail.forest-hybrid-isaac-composition.v7-development",
        "iteration_id": f"{args.scene_id}-confirmatory-v1",
        "status": "confirmatory_f1_candidate_model_blind",
        "extends": {
            "path": str(BASE_CONFIG.relative_to(ROOT)),
            "sha256": _sha256(BASE_CONFIG),
        },
        "overrides": {
            "scene_id": args.scene_id,
            "source_scene_id": (
                f"{candidate['hdri_id']}_metric_geometry_seed_{seed}"
            ),
            "background": {
                "panorama": {
                    "path": str(panorama),
                    "sha256": _sha256(panorama),
                },
                "preflight": {
                    "path": str(preflight),
                    "sha256": _sha256(preflight),
                },
                "intensity": 325.0,
                "rotation_xyz_deg": [0.0, 0.0, rotation],
            },
            "appearance": {
                "route_material_lock": {
                    "path": str(material_lock_path),
                    "sha256": _sha256(material_lock_path),
                },
                "route_material_id": args.route_material_id,
                "route_expected_split": "confirmatory",
                "surrounding_material_lock": {
                    "path": str(material_lock_path),
                    "sha256": _sha256(material_lock_path),
                },
                "surrounding_material_id": args.surrounding_material_id,
                "surrounding_expected_split": "confirmatory",
            },
            "route": {
                "waypoints_xy_m": [
                    [round(route_length * index / 4.0, 6), route_y]
                    for index in range(5)
                ],
                "surface_width_m": 1.5,
                "endpoint_margin_m": 0.88,
            },
            "near_field": {
                "prop_transform_overrides": _prop_transforms(seed, route_y),
                "opaque_composited_ground": {
                    "seed": seed,
                    "centerline_wander_m": round(
                        float(rng.uniform(0.08, 0.16)), 6
                    ),
                    "phase_rad": phase,
                },
                "visual_height_variation": {
                    "phase_rad": round(
                        float(rng.uniform(0.0, 2.0 * math.pi)), 6
                    )
                },
            },
            "output": {"directory": str(output_dir)},
        },
        "confirmation_contract": {
            "new_hdri_asset": True,
            "new_metric_geometry_seed": True,
            "new_route_and_surrounding_materials": True,
            "selected_before_model_inference": True,
            "appearance_views_are_not_independent_units": True,
            "counts_as_prior_a0_a7_evidence": False,
        },
    }
    composition_path.write_text(
        json.dumps(composition, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if failure_stage is None:
        compile_result = _run(
            [
                str(ISAAC_PYTHON),
                str(COMPILER),
                "--config",
                str(composition_path),
            ]
        )
        ledger.append({"stage": "metric_scene_compile", **compile_result})
        if not compile_result["passed"]:
            failure_stage = "metric_scene_compile"

    compiled_audit = output_dir / "compiled_scene_audit.json"
    episode_usd = output_dir / "episode.usda"
    nominal_root = candidate_root / "nominal_qa"
    if failure_stage is None:
        nominal_result = _run(
            [
                "/bin/bash",
                "-lc",
                'source "$1"; shift; exec "$@"',
                "kinofail-confirmatory-wild",
                str(ISAAC_SETUP),
                str(ISAACLAB_PYTHON),
                str(NOMINAL_COLLECTOR),
                "--operator",
                "o1",
                "--lane",
                "nominal",
                "--episode-usd",
                str(episode_usd),
                "--compiled-audit",
                str(compiled_audit),
                "--out",
                str(nominal_root),
                "--camera-profile",
                "go2_front_calib_b",
                "--material-id",
                args.route_material_id,
                "--material-lock",
                str(material_lock_path),
                "--policy-path",
                str(POLICY),
                "--seed",
                str(int(candidate["runtime_seed"])),
                "--steps",
                "650",
                "--region-progress-m",
                "0.86",
                "--region-half-length-m",
                "0.45",
                "--region-half-width-m",
                "0.60",
                "--forward-speed-mps",
                "0.32",
                "--max-route-deviation-m",
                "0.30",
                "--min-region-samples",
                "40",
                "--headless",
            ]
        )
        ledger.append({"stage": "nominal_go2_qa", **nominal_result})
        if not nominal_result["passed"]:
            failure_stage = "nominal_go2_qa"

    evidence: dict[str, Any] = {}
    for name, path in (
        ("composition_config", composition_path),
        ("compiled_scene", compiled_audit),
        ("nominal_go2", nominal_root / "lane_manifest.json"),
    ):
        if path.is_file():
            payload = _json(path)
            evidence[name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "passed": payload.get("passed", True if name == "composition_config" else None),
            }
    admitted = (
        failure_stage is None
        and set(evidence)
        == {"composition_config", "compiled_scene", "nominal_go2"}
        and evidence["compiled_scene"]["passed"] is True
        and evidence["nominal_go2"]["passed"] is True
    )
    terminal = {
        "schema_version": "kinofail.confirmatory-wild-scene-admission.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": args.scene_id,
        "candidate": candidate,
        "model_blind": True,
        "model_or_endpoint_files_read": False,
        "material_ids": [
            args.route_material_id,
            args.surrounding_material_id,
        ],
        "scene_config": {
            "path": str(scene_config_path),
            "sha256": _sha256(scene_config_path),
        },
        "hdri_lock": {
            "path": str(hdri_lock_path),
            "sha256": _sha256(hdri_lock_path),
        },
        "material_lock": {
            "path": str(material_lock_path),
            "sha256": _sha256(material_lock_path),
        },
        "ledger": ledger,
        "evidence": evidence,
        "failure_stage": failure_stage,
        "admitted": admitted,
        "episode_usd": str(episode_usd) if admitted else None,
        "confirmatory_model_inference_authorized": False,
    }
    terminal_path = candidate_root / "terminal_scene_admission.json"
    terminal_path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"scene_id": args.scene_id, "admitted": admitted}))
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
