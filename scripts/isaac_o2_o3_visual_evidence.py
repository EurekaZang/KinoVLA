#!/usr/bin/env python3
"""Create an auditable RTX review bundle for O2/O3 terrain-state visual synchronization.

The structural gates remain the publication evidence.  This script renders the same live Isaac
state from three fixed viewpoints so a human reviewer can inspect whether measured O2 contacts
produce persistent soil imprints and whether the O3 impulse trigger produces a broken surface.
"""

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _select_record(records: list[dict[str, object]], **required: object) -> dict[str, object]:
    matches = [
        record
        for record in records
        if all(record.get(key) == value for key, value in required.items())
    ]
    if not matches:
        raise RuntimeError(f"no benchmark record matched {required}")
    return matches[0]


def _drive_o2(backend, region) -> object:
    command = np.asarray([0.55, 0.0, 0.0], dtype=np.float64)
    obs = None
    for _ in range(320):
        obs = backend.step(command)
        if obs.pos[0] > region.cx + region.hx + 0.4 or obs.fallen:
            break
    if obs is None:
        raise RuntimeError("O2 rollout produced no observation")
    return obs


def _drive_o3(backend) -> tuple[object, int]:
    command = np.asarray([0.55, 0.0, 0.0], dtype=np.float64)
    obs = None
    post_trigger_steps = 0
    for _ in range(320):
        collapsed = bool(backend.collapse_telemetry()["regions"][0]["collapsed"])
        obs = backend.step(command if not collapsed else np.zeros(3, dtype=np.float64))
        if bool(backend.collapse_telemetry()["regions"][0]["collapsed"]):
            post_trigger_steps += 1
        if post_trigger_steps >= 35:
            break
    if obs is None:
        raise RuntimeError("O3 rollout produced no observation")
    return obs, post_trigger_steps


def _capture_views(backend, output: Path, case_id: str, views, image_module):
    frames: list[dict[str, object]] = []
    for view_id, eye, target in views:
        backend.aim_record_camera(np.asarray(eye), np.asarray(target))
        # Flush the camera transform and permit RTX/MDL temporal state to settle.  A fallen
        # rollout intentionally freezes policy stepping, so render explicitly as well: otherwise
        # all post-fall viewpoints would reuse the last sensor product despite new USD poses.
        for _ in range(2):
            backend.step(np.zeros(3, dtype=np.float64))
            backend._env.sim.render()
        frame = None
        for _ in range(10):
            frame = backend.capture_rgb(force_recompute=True)
        if frame is None:
            raise RuntimeError(f"record camera produced no RGB for {case_id}/{view_id}")
        path = output / f"{case_id}_{view_id}.png"
        image_module.fromarray(frame).save(path)
        frames.append(
            {
                "view_id": view_id,
                "eye_xyz_m": list(eye),
                "target_xyz_m": list(target),
                "path": str(path),
                "sha256": _sha256(path),
                "rgb_shape": list(frame.shape),
                "rgb_mean": float(np.mean(frame)),
                "rgb_std": float(np.std(frame)),
            }
        )
    return frames


def _make_contact_sheet(output: Path, cases, image_module, draw_module, font_module) -> Path:
    thumb_w, thumb_h = 512, 288
    margin, title_h, label_h = 22, 58, 32
    canvas = image_module.new(
        "RGB",
        (
            3 * thumb_w + 4 * margin,
            2 * (thumb_h + label_h) + title_h + 3 * margin,
        ),
        "white",
    )
    draw = draw_module.Draw(canvas)
    font = font_module.load_default(size=18)
    draw.text(
        (margin, margin),
        "Kino-Fail terrain state / visual synchronization (live Isaac RTX)",
        fill=(20, 26, 30),
        font=font,
    )
    for row, case in enumerate(cases):
        for column, frame in enumerate(case["frames"]):
            image = image_module.open(frame["path"]).convert("RGB")
            image = image.resize((thumb_w, thumb_h), image_module.Resampling.LANCZOS)
            x = margin + column * (thumb_w + margin)
            y = title_h + margin + row * (thumb_h + label_h + margin)
            canvas.paste(image, (x, y + label_h))
            draw.text(
                (x, y + 5),
                f"{case['case_id']} / {frame['view_id']}",
                fill=(20, 26, 30),
                font=font,
            )
    path = output / "o2_o3_visual_evidence_contact_sheet.jpg"
    canvas.save(path, quality=95)
    return path


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument(
        "--asset-lock", default="outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
    )
    preliminary.add_argument("--out", default="outputs/kinofail_realistic/o2_o3_visual_evidence")
    pre_args, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser(description="O2/O3 live RTX visual evidence bundle")
    parser.add_argument("--asset-lock", default=pre_args.asset_lock)
    parser.add_argument("--out", default=pre_args.out)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import isaaclab
    from PIL import Image, ImageDraw, ImageFont

    from kino_vla.data.realistic_benchmark import build_realistic_corpus
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import Collapse, ComplianceField
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
    records = build_realistic_corpus(mode="full").records
    o2_record = _select_record(
        records,
        condition="anomaly",
        target_operator="O2_compliance",
        severity_id="severe",
        split="train",
        domain="wild",
        material_family="train_ground037",
        surface_state="damp",
    )
    o3_record = _select_record(
        records,
        condition="anomaly",
        target_operator="O3_collapse",
        severity_id="severe",
        split="train",
        domain="production",
        material_family="train_concrete046",
        surface_state="wet",
    )

    cfg = load_config("sim/go2_skeleton.yaml", {"cam_width": 1280, "cam_height": 720})
    backend = IsaacPolicyBackend(
        cfg,
        np.asarray([0.0, 0.0], dtype=np.float64),
        0.0,
        record_cam=True,
    )
    hidden_default_grid_prims = backend.hide_default_ground_visual()
    if hidden_default_grid_prims == 0:
        raise RuntimeError("default Isaac ground was not found; cannot prove grid suppression")
    views = (
        ("overview", (3.8, -3.0, 2.35), (1.8, 0.0, -0.04)),
        ("side_close", (2.15, -2.1, 0.78), (1.8, 0.0, -0.07)),
        ("top_oblique", (1.9, -0.55, 3.85), (1.9, 0.0, -0.10)),
    )
    cases: list[dict[str, object]] = []

    # O2: the persistent imprints are created only by named, measured load-bearing contacts.
    o2_region = Rect(1.9, 0.0, 0.9, 0.8)
    o2_binding = appearance_binding_from_record(
        o2_record,
        lock=lock,
        asset_root=REPO_ROOT / str(lock["asset_root"]),
    )
    backend.deep_reset(seed=int(o2_record["operator_seed"]))
    backend.set_terrain_appearance(o2_binding)
    ComplianceField(
        region=o2_region,
        k_c=240.0,
        c_c=0.32,
        d_sink=0.11,
        realistic_foot_model=True,
    ).on_reset(backend)
    backend.reset(int(o2_record["operator_seed"]))
    o2_obs = _drive_o2(backend, o2_region)
    o2_telemetry = backend.foot_compliance_telemetry()
    o2_frames = _capture_views(backend, output, "O2_disturbed_soil", views, Image)
    o2_checks = {
        "measured_footprints_exist": int(o2_telemetry["visual_footprint_count"]) >= 4,
        "multiple_named_feet": len(
            {str(row["foot_name"]) for row in o2_telemetry["visual_footprints"]}
        )
        >= 2,
        "images_nonblank": all(float(frame["rgb_std"]) >= 5.0 for frame in o2_frames),
        "view_hashes_unique": len({str(frame["sha256"]) for frame in o2_frames}) == len(views),
    }
    cases.append(
        {
            "case_id": "O2_disturbed_soil",
            "operator": "O2_compliance",
            "record_id": o2_record["episode_id"],
            "domain": o2_record["domain"],
            "scene_family": o2_record["scene_family"],
            "material_family": o2_binding.material_id,
            "appearance_id": o2_binding.appearance_id,
            "surface_state": o2_binding.surface_state,
            "omnipbr": o2_binding.omnipbr_parameters(),
            "final_robot_position_xy_m": [float(value) for value in o2_obs.pos],
            "fallen": bool(o2_obs.fallen),
            "visual_footprint_count": int(o2_telemetry["visual_footprint_count"]),
            "visual_footprint_named_feet": sorted(
                {str(row["foot_name"]) for row in o2_telemetry["visual_footprints"]}
            ),
            "checks": o2_checks,
            "frames": o2_frames,
        }
    )

    # O3: the same measured impulse event disables support and authors the fractured facets.
    o3_region = Rect(1.8, 0.0, 0.9, 0.8)
    o3_binding = appearance_binding_from_record(
        o3_record,
        lock=lock,
        asset_root=REPO_ROOT / str(lock["asset_root"]),
    )
    backend.deep_reset(seed=int(o3_record["operator_seed"]))
    backend.set_terrain_appearance(o3_binding)
    Collapse(
        region=o3_region,
        mu_intact=0.8,
        mu_collapsed=0.07,
        damage_threshold_ns=25.0,
        drop_m=0.12,
        residual_support=0.18,
    ).on_reset(backend)
    backend.reset(int(o3_record["operator_seed"]))
    o3_obs, post_trigger_steps = _drive_o3(backend)
    o3_telemetry = backend.collapse_telemetry()["regions"][0]
    o3_frames = _capture_views(backend, output, "O3_fractured_surface", views, Image)
    o3_checks = {
        "measured_impulse_triggered": bool(o3_telemetry["collapsed"]),
        "failed_support_cells_exist": int(o3_telemetry["failed_support_cells"]) > 0,
        "fracture_facets_cover_failed_cells": int(o3_telemetry["fracture_visual_count"])
        == 2 * int(o3_telemetry["failed_support_cells"]),
        "visual_and_physics_share_trigger": o3_telemetry["visual_sync_trigger_step"]
        == o3_telemetry["last_damage_update"]["trigger_step"],
        "images_nonblank": all(float(frame["rgb_std"]) >= 5.0 for frame in o3_frames),
        "view_hashes_unique": len({str(frame["sha256"]) for frame in o3_frames}) == len(views),
    }
    cases.append(
        {
            "case_id": "O3_fractured_surface",
            "operator": "O3_collapse",
            "record_id": o3_record["episode_id"],
            "domain": o3_record["domain"],
            "scene_family": o3_record["scene_family"],
            "material_family": o3_binding.material_id,
            "appearance_id": o3_binding.appearance_id,
            "surface_state": o3_binding.surface_state,
            "omnipbr": o3_binding.omnipbr_parameters(),
            "final_robot_position_xy_m": [float(value) for value in o3_obs.pos],
            "fallen": bool(o3_obs.fallen),
            "post_trigger_steps": post_trigger_steps,
            "failed_support_cells": int(o3_telemetry["failed_support_cells"]),
            "fracture_visual_count": int(o3_telemetry["fracture_visual_count"]),
            "trigger_step": o3_telemetry["visual_sync_trigger_step"],
            "normal_impulse_ns": o3_telemetry["last_damage_update"]["normal_impulse_ns"],
            "checks": o3_checks,
            "frames": o3_frames,
        }
    )

    sheet = _make_contact_sheet(output, cases, Image, ImageDraw, ImageFont)
    checks = {
        "default_grid_suppressed": hidden_default_grid_prims > 0,
        "all_case_checks_pass": all(
            all(bool(value) for value in case["checks"].values()) for case in cases
        ),
        "six_rtx_views_created": sum(len(case["frames"]) for case in cases) == 6,
        "contact_sheet_created": sheet.exists() and sheet.stat().st_size > 0,
    }
    manifest = {
        "schema_version": "kinofail.o2-o3-visual-evidence.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "review_bundle_not_corpus_evidence",
        "claim_boundary": (
            "Live RTX images support visual review only; structural USD/PhysX gates and paired "
            "corpus statistics carry the publication claim."
        ),
        "checks": checks,
        "hidden_default_grid_prims": hidden_default_grid_prims,
        "asset_lock": str(asset_lock_path),
        "asset_lock_sha256": _sha256(asset_lock_path),
        "cases": cases,
        "contact_sheet": str(sheet),
        "contact_sheet_sha256": _sha256(sheet),
        "software": {
            "isaac_lab_version": str(isaaclab.__version__),
            "isaac_sim_version": (Path(sys.executable).resolve().parents[3] / "VERSION")
            .read_text(encoding="utf-8")
            .strip(),
            "physics_engine": "PhysX GPU",
            "physics_dt_s": float(cfg.physics_dt),
            "control_dt_s": float(backend.dt),
        },
        "input_sha256": {
            "policy": _sha256(REPO_ROOT / str(cfg.policy_path)),
            "backend": _sha256(REPO_ROOT / "kino_vla/sim/isaac_policy_backend.py"),
            "o2_operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o2_compliance.py"),
            "o3_operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o3_collapse.py"),
            "terramechanics": _sha256(REPO_ROOT / "kino_vla/sim/terramechanics.py"),
            "collapse": _sha256(REPO_ROOT / "kino_vla/sim/collapse.py"),
            "terrain_materials": _sha256(REPO_ROOT / "kino_vla/sim/terrain_materials.py"),
            "benchmark_design": _sha256(REPO_ROOT / "kino_vla/data/realistic_benchmark.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
    }
    _write_json(output / "gate_manifest.json", manifest)
    ok = bool(manifest["passed"])
    print("PASS: O2/O3 terrain visual evidence bundle" if ok else "FAIL: O2/O3 visual bundle")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
