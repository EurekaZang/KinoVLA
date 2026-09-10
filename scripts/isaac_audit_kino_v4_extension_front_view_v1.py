#!/usr/bin/env python3
"""Model-blind registered-material visibility audit for extension scenes.

The audit places the frozen Go2 front camera at the route start, changes only
the visual material bound to the audited route surface, and requires both
registered swaps to produce the same minimum aligned RGB change used by the
runtime validator.  No operator, checkpoint, feature extractor, label, or
prediction is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--scene-id", required=True)
    preliminary.add_argument("--episode-usd", type=Path, required=True)
    preliminary.add_argument("--compiled-audit", type=Path, required=True)
    preliminary.add_argument("--material-lock", type=Path, required=True)
    preliminary.add_argument("--material-id", action="append", required=True)
    preliminary.add_argument("--camera-profile", default="go2_front_calib_b")
    preliminary.add_argument("--seed", type=int, required=True)
    preliminary.add_argument("--output", type=Path, required=True)
    preliminary.add_argument("--minimum-swap-l1", type=float, default=0.015)
    pre, _ = preliminary.parse_known_args()
    parser = argparse.ArgumentParser(parents=[preliminary])
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    if len(args.material_id) != 3 or len(set(args.material_id)) != 3:
        raise ValueError("exactly three distinct material IDs are required")
    if not math.isclose(args.minimum_swap_l1, 0.015, abs_tol=0.0, rel_tol=0.0):
        raise ValueError("the F4h visibility threshold is fixed at 0.015")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    episode = args.episode_usd.resolve()
    compiled_path = args.compiled_audit.resolve()
    lock_path = args.material_lock.resolve()
    for path in (episode, compiled_path, lock_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    app = AppLauncher(args).app
    passed = False
    backend = None
    try:
        import omni.usd
        import yaml

        from kino_vla.sim.isaac_o4_v3_backend import IsaacPolicyBackendO4V3
        from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding
        from kino_vla.sim.terrain_materials import (
            appearance_binding_from_record,
            bind_omnipbr_material,
            load_terrain_asset_lock,
        )
        from kino_vla.utils.config import CONFIGS_DIR, load_config
        from scripts.isaac_collect_realistic_smoke_pair import _rgb_uint8

        compiled = load(compiled_path)
        route = scene_route_binding(compiled)
        benchmark = yaml.safe_load(
            (CONFIGS_DIR / "data/kinofail_realistic.yaml").read_text(encoding="utf-8")
        )
        camera = dict(benchmark["camera_profile_specs"][args.camera_profile])
        mount = camera["mount_xyz_m"]
        config = load_config(
            "sim/go2_skeleton.yaml",
            {
                "perception_cam_width": camera["width"],
                "perception_cam_height": camera["height"],
                "perception_focal_mm": camera["focal_length_mm"],
                "perception_cam_x_m": mount[0],
                "perception_cam_y_m": mount[1],
                "perception_cam_z_m": mount[2],
                "perception_cam_pitch_down_rad": camera["pitch_down_rad"],
            },
        )
        backend = IsaacPolicyBackendO4V3(
            config,
            route.frame.point(0.0, 0.0),
            route.frame.heading_rad,
            perception_cam=True,
        )
        scene_prim = backend.load_realistic_scene(str(episode))
        route_surface = route.route_surface_prim_path(scene_prim)
        backend.deep_reset(int(args.seed))
        backend.reset(int(args.seed))
        lock = load_terrain_asset_lock(lock_path)
        asset_root = Path(str(lock["asset_root"]))
        if not asset_root.is_absolute():
            asset_root = ROOT / asset_root
        material_rows = {str(row["id"]): dict(row) for row in lock["materials"]}
        bindings = []
        for index, material_id in enumerate(args.material_id):
            if material_id not in material_rows:
                raise KeyError(material_id)
            record = {
                "material_family": material_id,
                "appearance_id": f"f4h_front_gate_{args.scene_id}_{index}",
                "uv_scale": 1.0,
                "uv_rotation_deg": float(index * 90),
                "surface_state": ("clean", "dusty", "wet")[index],
                "uv_offset": [0.137 * index, 0.211 * index],
                "albedo_brightness_multiplier": 1.0,
                "normal_strength": 1.0,
                "roughness_multiplier": 1.0,
            }
            bindings.append(
                appearance_binding_from_record(
                    record, lock=lock, asset_root=asset_root
                )
            )

        output.mkdir(parents=True, exist_ok=False)
        frames = []
        stage = omni.usd.get_context().get_stage()
        for index, binding in enumerate(bindings):
            bind_omnipbr_material(stage, route_surface, binding)
            for _ in range(12):
                backend._env.sim.render()
            capture = backend.capture_perception()
            if capture is None:
                raise RuntimeError("Go2 front camera returned no frame")
            rgb = _rgb_uint8(capture["rgb"])
            frame_path = output / f"view_{index}.png"
            Image.fromarray(rgb).save(frame_path)
            frames.append(rgb.astype(np.float32))

        primary = frames[0]
        swap_l1 = [
            float(np.mean(np.abs(primary - frame)) / 255.0)
            for frame in frames[1:]
        ]
        luminance_std = [
            float(np.std(0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]))
            for frame in frames
        ]
        checks = {
            "three_distinct_material_ids": len(set(args.material_id)) == 3,
            "both_registered_swaps_visible": all(
                value >= args.minimum_swap_l1 for value in swap_l1
            ),
            "all_frames_nondegenerate": all(value >= 12.0 for value in luminance_std),
            "model_or_prediction_loaded": False,
        }
        # The final entry is a negative contamination sentinel: ``False`` is
        # the required clean value.  Treating every check as a positive
        # predicate would reject every otherwise valid scene.
        passed = (
            checks["three_distinct_material_ids"] is True
            and checks["both_registered_swaps_visible"] is True
            and checks["all_frames_nondegenerate"] is True
            and checks["model_or_prediction_loaded"] is False
        )
        audit = {
            "schema_version": "kinofail.kino-v4-extension-front-view-audit.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "passed": passed,
            "scene_id": args.scene_id,
            "model_prediction_truth_key_or_score_read": False,
            "checks": checks,
            "minimum_swap_l1": args.minimum_swap_l1,
            "measured_swap_l1": swap_l1,
            "luminance_std_0_255": luminance_std,
            "camera_profile": args.camera_profile,
            "material_ids": list(args.material_id),
            "source_sha256": {
                "episode_usd": sha256(episode),
                "compiled_audit": sha256(compiled_path),
                "material_lock": sha256(lock_path),
                "audit_script": sha256(Path(__file__).resolve()),
            },
            "frame_sha256": {
                f"view_{index}": sha256(output / f"view_{index}.png")
                for index in range(3)
            },
        }
        atomic_json(output / "audit.json", audit)
        print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Isaac Kit can deadlock while closing after headless RTX use.  This
        # process owns the app and has no in-memory result worth preserving
        # after an exception, so let the OS reclaim it deterministically.
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    # The terminal audit and PNGs are atomically on disk.  A graceful Kit
    # shutdown intermittently deadlocks in this one-shot gate process and
    # blocks the entire unattended admission queue.
    os._exit(0 if passed else 2)


if __name__ == "__main__":
    raise SystemExit(main())
