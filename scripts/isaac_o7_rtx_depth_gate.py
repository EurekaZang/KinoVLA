#!/usr/bin/env python3
"""Paired Go2-front RTX/PBR/depth-corruption gate for realistic O7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEEDS = (11, 23, 37, 42, 59)
DOSES = (
    ("nominal", 0.80, 0.60, 0.00),
    ("mild", 0.36, 0.30, 0.08),
    ("moderate", 0.18, 0.13, 0.18),
    ("severe", 0.09, 0.06, 0.32),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rgb_u8(value: np.ndarray) -> np.ndarray:
    rgb = np.asarray(value)[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(rgb)) <= 1.5 else 1.0
        rgb = np.clip(rgb * scale, 0.0, 255.0)
    return rgb.astype(np.uint8)


def _select_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates = [
        row
        for row in records
        if row["target_operator"] == "O7_visual_remap"
        and row["condition"] == "anomaly"
        and row["severity_id"] == "severe"
    ]
    selected = []
    seen_materials = set()
    for row in candidates:
        material = str(row["material_family"])
        if material in seen_materials:
            continue
        selected.append(row)
        seen_materials.add(material)
        if len(selected) == len(SEEDS):
            break
    if len(selected) != len(SEEDS):
        raise RuntimeError("O7 gate requires five distinct PBR material families")
    return selected


def _semantic_mask(capture: dict[str, object]) -> np.ndarray:
    telemetry = capture["depth_fault_telemetry"]
    semantic_ids = telemetry["faults"][0]["semantic_ids"]
    return np.isin(np.asarray(capture["seg"]), semantic_ids)


def _backproject_xy(capture: dict[str, object], depth: np.ndarray) -> np.ndarray:
    eye = np.asarray(capture["eye"], dtype=np.float64)
    target = np.asarray(capture["target"], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack([right, down, forward])
    height, width = depth.shape
    k = np.asarray(capture["K"], dtype=np.float64)
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
    )
    ray_camera = np.stack(
        [(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)],
        axis=-1,
    )
    ray_world = ray_camera @ rotation.T
    return eye[:2] + depth[..., None] * ray_world[..., :2]


def _capture_metrics(capture: dict[str, object], bias_m: float) -> dict[str, object]:
    depth = np.asarray(capture["depth"], dtype=np.float64)
    mask = _semantic_mask(capture) & np.isfinite(depth)
    raw_depth = depth.copy()
    raw_depth[mask] -= bias_m
    raw_xy = _backproject_xy(capture, raw_depth)
    output_xy = _backproject_xy(capture, depth)
    raw_center = np.median(raw_xy[mask], axis=0)
    output_center = np.median(output_xy[mask], axis=0)
    rgb = _rgb_u8(np.asarray(capture["rgb"]))
    telemetry = capture["depth_fault_telemetry"]
    return {
        "rgb": rgb,
        "depth": depth,
        "raw_depth": raw_depth,
        "mask": mask,
        "affected_pixels": int(mask.sum()),
        "raw_center_xy_m": raw_center,
        "output_center_xy_m": output_center,
        "centroid_shift_m": float(np.linalg.norm(output_center - raw_center)),
        "camera_pose_readback_error_m": float(capture["camera_pose_sync_error_m"]),
        "camera_orientation_readback_error_rad": float(
            capture["camera_orientation_sync_error_rad"]
        ),
        "body_fixed": bool(capture["body_fixed"]),
        "pose_sync_method": str(capture["pose_sync_method"]),
        "camera_timestamp_s": float(capture["camera_timestamp_s"]),
        "depth_source_timestamp_s": float(capture["depth_source_timestamp_s"]),
        "sensor_read_timestamp_s": float(capture["sensor_read_timestamp_s"]),
        "effective_depth_age_s": float(telemetry["effective_depth_age_s"]),
        "frame_id": int(capture["camera_frame_id"]),
        "telemetry": telemetry,
    }


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--out", default="outputs/kinofail_realistic/o7_rtx_depth_gate")
    preliminary.add_argument(
        "--asset-lock", default="outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
    )
    pre_args, _ = preliminary.parse_known_args()
    parser = argparse.ArgumentParser(description="Kino-Fail O7 Go2-front RTX depth gate")
    parser.add_argument("--out", default=pre_args.out)
    parser.add_argument("--asset-lock", default=pre_args.asset_lock)
    parser.add_argument("--max-rgb-mae", type=float, default=10.0)
    parser.add_argument("--max-raw-depth-mae-m", type=float, default=0.01)
    parser.add_argument("--min-region-pixels", type=int, default=100)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import isaaclab
    from PIL import Image

    from kino_vla.data.realistic_benchmark import build_realistic_corpus
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators.o7_visual_remap import VisualPhysicsRemap
    from kino_vla.sim.terrain_materials import (
        appearance_binding_from_record,
        load_terrain_asset_lock,
    )
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    output = (REPO_ROOT / args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    asset_lock_path = (REPO_ROOT / args.asset_lock).resolve()
    lock = load_terrain_asset_lock(asset_lock_path)
    design = build_realistic_corpus(mode="full")
    selected = _select_records(design.records)
    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.zeros(2), 0.0, perception_cam=True)

    records: list[dict[str, object]] = []
    in_memory: dict[tuple[int, str], dict[str, object]] = {}
    for dose, mu_s, mu_d, depth_bias_m in DOSES:
        for index, (seed, design_row) in enumerate(zip(SEEDS, selected, strict=True)):
            # Five PBR/camera nuisance cells are fixed across all four doses.
            start_y = 0.03 * (index - 2)
            heading = 0.025 * (index - 2)
            backend._start_pos = np.array([0.0, start_y], dtype=np.float64)
            backend._start_heading = heading
            backend.deep_reset(seed)
            binding = appearance_binding_from_record(
                design_row,
                lock=lock,
                asset_root=REPO_ROOT / str(lock["asset_root"]),
            )
            backend.set_terrain_appearance(binding)
            direction = np.array([np.cos(heading), np.sin(heading)])
            center = backend._start_pos + 1.2 * direction
            region = Rect(float(center[0]), float(center[1]), 0.50, 0.62)
            operator = VisualPhysicsRemap(
                region,
                mu_s=mu_s,
                mu_d=mu_d,
                depth_bias_m=depth_bias_m,
                appearance_class="solid_ground",
            )
            operator.on_reset(backend)
            for _ in range(4):
                backend.step(np.zeros(3, dtype=np.float64))
            capture = None
            for _ in range(4):
                capture = backend.capture_perception()
            if capture is None:
                raise RuntimeError("perception camera returned no frame")
            metrics = _capture_metrics(capture, depth_bias_m)
            image_path = output / f"seed_{seed}_{binding.material_id}_{dose}.png"
            Image.fromarray(metrics["rgb"]).save(image_path)
            in_memory[(seed, dose)] = metrics
            fault = metrics["telemetry"]["faults"][0]
            row = {
                "seed": seed,
                "dose": dose,
                "domain": design_row["domain"],
                "scene_family": design_row["scene_family"],
                "material_family": binding.material_id,
                "appearance_id": binding.appearance_id,
                "surface_state": binding.surface_state,
                "omnipbr": binding.omnipbr_parameters(),
                "start_y_m": start_y,
                "start_heading_rad": heading,
                "region": {
                    "cx": region.cx,
                    "cy": region.cy,
                    "hx": region.hx,
                    "hy": region.hy,
                },
                "requested_mu_s": mu_s,
                "requested_mu_d": mu_d,
                "friction_readback_mu_d": backend.friction_at(center),
                "requested_depth_bias_m": depth_bias_m,
                "mean_applied_depth_bias_m": fault["mean_applied_bias_m"],
                "affected_pixels": metrics["affected_pixels"],
                "centroid_shift_m": metrics["centroid_shift_m"],
                "camera_pose_readback_error_m": metrics["camera_pose_readback_error_m"],
                "camera_orientation_readback_error_rad": metrics[
                    "camera_orientation_readback_error_rad"
                ],
                "body_fixed": metrics["body_fixed"],
                "pose_sync_method": metrics["pose_sync_method"],
                "camera_timestamp_s": metrics["camera_timestamp_s"],
                "depth_source_timestamp_s": metrics["depth_source_timestamp_s"],
                "sensor_read_timestamp_s": metrics["sensor_read_timestamp_s"],
                "effective_depth_age_s": metrics["effective_depth_age_s"],
                "frame_id": metrics["frame_id"],
                "raw_depth_sha256": metrics["telemetry"]["raw_depth_sha256"],
                "output_depth_sha256": metrics["telemetry"]["output_depth_sha256"],
                "input_rgb_sha256": metrics["telemetry"]["input_rgb_sha256"],
                "output_rgb_sha256": metrics["telemetry"]["output_rgb_sha256"],
                "image": str(image_path),
            }
            records.append(row)
            print(
                f"[o7] seed={seed} {binding.material_id} {dose}: "
                f"mu={row['friction_readback_mu_d']:.2f} "
                f"bias={row['mean_applied_depth_bias_m']:.3f} m px={row['affected_pixels']} "
                f"map_shift={row['centroid_shift_m']:.3f} m"
            )

    pair_audits = []
    for seed in SEEDS:
        nominal = in_memory[(seed, "nominal")]
        dose_rows = []
        for dose, _mu_s, _mu_d, _bias in DOSES[1:]:
            anomaly = in_memory[(seed, dose)]
            rgb_mae = float(
                np.abs(anomaly["rgb"].astype(np.float32) - nominal["rgb"].astype(np.float32)).mean()
            )
            common = anomaly["mask"] & nominal["mask"]
            mask_iou = float(
                common.sum() / max(1, np.logical_or(anomaly["mask"], nominal["mask"]).sum())
            )
            raw_depth_mae = float(
                np.mean(np.abs(anomaly["raw_depth"][common] - nominal["raw_depth"][common]))
            )
            dose_rows.append(
                {
                    "dose": dose,
                    "rgb_mae": rgb_mae,
                    "semantic_mask_iou": mask_iou,
                    "raw_depth_mae_m": raw_depth_mae,
                }
            )
        shifts = [float(in_memory[(seed, dose)]["centroid_shift_m"]) for dose, *_ in DOSES]
        frictions = [
            next(
                float(row["friction_readback_mu_d"])
                for row in records
                if row["seed"] == seed and row["dose"] == dose
            )
            for dose, *_ in DOSES
        ]
        pair_audits.append(
            {
                "seed": seed,
                "dose_comparisons": dose_rows,
                "centroid_shifts_m": shifts,
                "shift_strictly_ordered": all(
                    a < b for a, b in zip(shifts, shifts[1:], strict=False)
                ),
                "friction_readbacks": frictions,
                "friction_strictly_ordered": all(
                    a > b for a, b in zip(frictions, frictions[1:], strict=False)
                ),
            }
        )

    checks = {
        "complete_paired_design": len(records) == len(SEEDS) * len(DOSES),
        "five_distinct_pbr_materials": len({row["material_family"] for row in records}) == 5,
        "go2_front_camera_body_fixed": all(row["body_fixed"] for row in records),
        "rigid_pose_readback": (
            max(row["camera_pose_readback_error_m"] for row in records) < 1e-4
            and max(row["camera_orientation_readback_error_rad"] for row in records) < 1e-3
        ),
        "timestamps_are_synchronous": all(
            row["camera_timestamp_s"] == row["depth_source_timestamp_s"] for row in records
        ),
        "camera_latency_is_timestamped_and_bounded": all(
            -1e-12 <= row["effective_depth_age_s"] <= backend.dt + 1e-9 for row in records
        ),
        "region_visible_in_every_frame": min(row["affected_pixels"] for row in records)
        >= args.min_region_pixels,
        "requested_depth_bias_is_realized": all(
            abs(row["mean_applied_depth_bias_m"] - row["requested_depth_bias_m"]) < 1e-9
            for row in records
        ),
        "nominal_depth_pipeline_is_identity": all(
            row["raw_depth_sha256"] == row["output_depth_sha256"]
            for row in records
            if row["dose"] == "nominal"
        ),
        "depth_pipeline_preserves_rgb_bitwise": all(
            row["input_rgb_sha256"] == row["output_rgb_sha256"] for row in records
        ),
        "depth_induced_map_shift_strictly_orders_each_pair": all(
            row["shift_strictly_ordered"] for row in pair_audits
        ),
        "physics_friction_strictly_orders_each_pair": all(
            row["friction_strictly_ordered"] for row in pair_audits
        ),
        "counterfactual_rgb_is_matched": all(
            comp["rgb_mae"] <= args.max_rgb_mae
            for row in pair_audits
            for comp in row["dose_comparisons"]
        ),
        "counterfactual_raw_depth_is_matched": all(
            comp["raw_depth_mae_m"] <= args.max_raw_depth_mae_m
            for row in pair_audits
            for comp in row["dose_comparisons"]
        ),
        "counterfactual_semantic_mask_is_matched": all(
            comp["semantic_mask_iou"] >= 0.95
            for row in pair_audits
            for comp in row["dose_comparisons"]
        ),
    }
    manifest = {
        "schema_version": "kinofail.o7-go2-front-rtx-depth-gate.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "sensor_and_visual_physics_contract_not_corpus_evidence",
        "checks": checks,
        "thresholds": {
            "max_rgb_mae": args.max_rgb_mae,
            "max_raw_depth_mae_m": args.max_raw_depth_mae_m,
            "min_region_pixels": args.min_region_pixels,
            "min_semantic_mask_iou": 0.95,
        },
        "records": records,
        "pair_audits": pair_audits,
        "camera_contract": {
            "mount": "Go2_base_rigid_front",
            "base_to_camera_xyz_m": backend._perception_mount_offset_m.tolist(),
            "pitch_down_rad": backend._perception_pitch_down_rad,
            "calibration_status": "engineering_nominal_requires_real_fixture_calibration",
            "depth_type": "Isaac_RTX_distance_to_image_plane",
        },
        "software": {
            "isaac_lab_version": str(isaaclab.__version__),
            "isaac_sim_version": (
                Path(sys.executable).resolve().parents[3] / "VERSION"
            ).read_text(encoding="utf-8").strip(),
            "physics_engine": "PhysX GPU",
            "physics_dt_s": float(cfg.physics_dt),
            "control_dt_s": float(backend.dt),
        },
        "input_sha256": {
            "policy": _sha256(REPO_ROOT / str(cfg.policy_path)),
            "backend": _sha256(REPO_ROOT / "kino_vla/sim/isaac_policy_backend.py"),
            "operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o7_visual_remap.py"),
            "depth_pipeline": _sha256(REPO_ROOT / "kino_vla/sim/depth_pipeline.py"),
            "live_segmenter": _sha256(REPO_ROOT / "kino_vla/map/live_rtx_segmenter.py"),
            "terrain_materials": _sha256(REPO_ROOT / "kino_vla/sim/terrain_materials.py"),
            "benchmark_design": _sha256(REPO_ROOT / "kino_vla/data/realistic_benchmark.py"),
            "asset_lock": _sha256(asset_lock_path),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
    }
    manifest_path = output / "gate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    ok = bool(manifest["passed"])
    print("PASS: O7 Go2-front RTX depth gate" if ok else "FAIL: O7 RTX depth gate")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
