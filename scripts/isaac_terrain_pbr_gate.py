"""Isaac RTX gate for appearance/physics-separated Kino-Fail terrain pairs.

For one O1 cell in each domain, render a nominal and a low-friction episode with identical scene,
camera, PBR material, UV transform, and seed.  The gate verifies both the PhysX friction readback
and near-identical pre-motion RGB, directly testing that texture does not reveal the condition.
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--out", default="outputs/kinofail_realistic/terrain_pbr_gate")
    preliminary.add_argument(
        "--asset-lock", default="outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
    )
    pre_args, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser(description="Kino-Fail terrain PBR/physics separation gate")
    parser.add_argument("--out", default=pre_args.out)
    parser.add_argument("--asset-lock", default=pre_args.asset_lock)
    parser.add_argument("--max-rgb-mae", type=float, default=10.0)
    parser.add_argument("--min-cross-material-rgb-mae", type=float, default=3.0)
    parser.add_argument("--max-pair-cross-ratio", type=float, default=0.20)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import isaaclab
    from PIL import Image, ImageDraw, ImageFont

    from kino_vla.data.realistic_benchmark import build_realistic_corpus
    from kino_vla.sim.isaac_policy_backend import (
        IsaacPolicyBackend,
        _read_material_friction,
        _set_material_friction,
    )
    from kino_vla.sim.operators import MuField
    from kino_vla.sim.terrain_materials import (
        appearance_binding_from_record,
        load_terrain_asset_lock,
    )
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.geometry import Rect

    output = REPO_ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    lock = load_terrain_asset_lock(REPO_ROOT / args.asset_lock)
    build = build_realistic_corpus(mode="full")
    selected = []
    for domain in ("life", "production", "wild"):
        rows = [
            row
            for row in build.records
            if row["condition"] == "anomaly"
            and row["target_operator"] == "O1_mu_field"
            and row["domain"] == domain
            and row["split"] == "train"
            and row["severity_id"] == "severe"
        ]
        selected.append(rows[0])

    sim_cfg = load_config(
        "sim/go2_skeleton.yaml",
        {"cam_width": 960, "cam_height": 540},
    )
    backend = IsaacPolicyBackend(
        sim_cfg,
        np.asarray([0.0, 0.0], dtype=np.float64),
        0.0,
        record_cam=True,
    )
    backend.aim_record_camera(np.asarray([2.2, -4.8, 4.3]), np.asarray([0.8, 0.0, 0.0]))
    hidden_grid_prims = backend.hide_default_ground_visual()
    if hidden_grid_prims == 0:
        raise RuntimeError("default Isaac ground was not found; cannot prove grid suppression")
    region = Rect(cx=0.0, cy=0.0, hx=4.0, hy=2.6)
    results = []
    rendered: list[tuple[str, Path]] = []
    for index, record in enumerate(selected):
        binding = appearance_binding_from_record(
            record,
            lock=lock,
            asset_root=REPO_ROOT / lock["asset_root"],
        )
        pair_frames: dict[str, np.ndarray] = {}
        friction: dict[str, float] = {}
        paths: dict[str, str] = {}

        backend.deep_reset(seed=int(record["operator_seed"]))
        backend.set_terrain_appearance(binding)
        MuField(region, mu_s=0.80, mu_d=0.60).on_reset(backend)
        patch_path = backend._friction_prim_paths[-1][1]
        friction["nominal"] = _read_material_friction(patch_path)[1]
        # Settle and compile the runtime-authored geometry/material exactly once.  The physical
        # state is frozen from here onward: the counterfactual changes only the PhysX material
        # attributes and refreshes Hydra, without advancing a simulation step.
        for _ in range(4):
            backend.step(np.zeros(3, dtype=np.float64))

        def state_vector() -> np.ndarray:
            tensors = (
                backend._robot.data.root_state_w[0],
                backend._robot.data.joint_pos[0],
                backend._robot.data.joint_vel[0],
            )
            return np.concatenate([tensor.detach().cpu().numpy().reshape(-1) for tensor in tensors])

        frozen_state = state_vector()
        for condition in ("nominal", "anomaly"):
            if condition == "anomaly":
                _set_material_friction(patch_path, 0.10, 0.07)
                friction[condition] = _read_material_friction(patch_path)[1]
            # Render-only refresh prevents both physics-state drift and an RTX annotator cache hit.
            for _ in range(8):
                backend._env.sim.render()
                frame = backend.capture_rgb(force_recompute=True)
                if frame is None:
                    raise RuntimeError("record camera produced no RGB")
            pair_frames[condition] = frame
            filename = f"{index + 1:02d}_{record['domain']}_{binding.material_id}_{condition}.png"
            path = output / filename
            Image.fromarray(frame).save(path)
            paths[condition] = str(path)
            rendered.append((f"{record['domain']} / {condition}", path))
        state_after_counterfactual = state_vector()
        state_max_abs_delta = float(np.max(np.abs(frozen_state - state_after_counterfactual)))
        difference = np.abs(
            pair_frames["nominal"].astype(np.float32) - pair_frames["anomaly"].astype(np.float32)
        )
        mae = float(difference.mean())
        result = {
            "domain": record["domain"],
            "scene_family": record["scene_family"],
            "counterfactual_group_id": record["counterfactual_group_id"],
            "material_family": binding.material_id,
            "appearance_id": binding.appearance_id,
            "surface_state": binding.surface_state,
            "omnipbr": binding.omnipbr_parameters(),
            "friction_readback": friction,
            "physics_state_max_abs_delta": state_max_abs_delta,
            "physics_state_sha256": hashlib.sha256(frozen_state.tobytes()).hexdigest(),
            "rgb_mae": mae,
            "rgb_max_abs": float(difference.max()),
            "frames": paths,
            "frame_sha256": {key: _sha256(Path(value)) for key, value in paths.items()},
            "passed": (
                abs(friction["nominal"] - 0.60) < 1.0e-6
                and abs(friction["anomaly"] - 0.07) < 1.0e-6
                and state_max_abs_delta == 0.0
                and mae <= float(args.max_rgb_mae)
            ),
        }
        results.append(result)
        print(
            f"[terrain-gate] {record['domain']} {binding.material_id}: "
            f"mu {friction['nominal']:.2f}->{friction['anomaly']:.2f}, RGB MAE={mae:.4f}"
        )

    nominal_frames = [Image.open(result["frames"]["nominal"]).convert("RGB") for result in results]
    cross_material_mae = []
    for left in range(len(nominal_frames)):
        for right in range(left + 1, len(nominal_frames)):
            a = np.asarray(nominal_frames[left], dtype=np.float32)
            b = np.asarray(nominal_frames[right], dtype=np.float32)
            cross_material_mae.append(float(np.abs(a - b).mean()))
    min_cross_material_mae = min(cross_material_mae)
    max_counterfactual_mae = max(float(result["rgb_mae"]) for result in results)
    pair_cross_ratio = max_counterfactual_mae / max(min_cross_material_mae, 1.0e-9)
    pbr_pixel_gate = min_cross_material_mae >= float(
        args.min_cross_material_rgb_mae
    ) and pair_cross_ratio <= float(args.max_pair_cross_ratio)

    # White-background review sheet with explicit pair labels.
    images = [(label, Image.open(path).convert("RGB")) for label, path in rendered]
    thumb_w, thumb_h, margin, label_h = 480, 270, 16, 30
    canvas_size = (2 * thumb_w + 3 * margin, 3 * (thumb_h + label_h) + 4 * margin)
    canvas = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for item, (label, image) in enumerate(images):
        row, column = divmod(item, 2)
        image = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = margin + column * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)
        canvas.paste(image, (x, y + label_h))
        draw.text((x, y + 5), label, fill=(20, 26, 30), font=font)
    sheet = output / "counterfactual_pbr_contact_sheet.jpg"
    canvas.save(sheet, quality=94)
    manifest = {
        "schema_version": "kinofail.terrain-pbr-gate.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(result["passed"] for result in results) and pbr_pixel_gate,
        "publication_status": "appearance_physics_vertical_slice_not_corpus_evidence",
        "asset_lock": str(REPO_ROOT / args.asset_lock),
        "asset_lock_sha256": _sha256(REPO_ROOT / args.asset_lock),
        "max_rgb_mae": float(args.max_rgb_mae),
        "hidden_default_grid_prims": hidden_grid_prims,
        "min_cross_material_rgb_mae": min_cross_material_mae,
        "required_cross_material_rgb_mae": float(args.min_cross_material_rgb_mae),
        "max_counterfactual_rgb_mae": max_counterfactual_mae,
        "counterfactual_to_cross_material_ratio": pair_cross_ratio,
        "required_max_pair_cross_ratio": float(args.max_pair_cross_ratio),
        "pbr_pixel_gate": pbr_pixel_gate,
        "results": results,
        "contact_sheet": str(sheet),
        "software": {
            "isaac_lab_version": str(isaaclab.__version__),
            "isaac_sim_version": (Path(sys.executable).resolve().parents[3] / "VERSION")
            .read_text(encoding="utf-8")
            .strip(),
            "physics_engine": "PhysX GPU",
            "physics_dt_s": float(sim_cfg.physics_dt),
            "control_dt_s": float(backend.dt),
        },
        "input_sha256": {
            "policy": _sha256(REPO_ROOT / str(sim_cfg.policy_path)),
            "backend": _sha256(REPO_ROOT / "kino_vla/sim/isaac_policy_backend.py"),
            "operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o1_mu_field.py"),
            "terrain_materials": _sha256(REPO_ROOT / "kino_vla/sim/terrain_materials.py"),
            "benchmark_design": _sha256(REPO_ROOT / "kino_vla/data/realistic_benchmark.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
    }
    manifest_path = output / "gate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    ok = bool(manifest["passed"])
    print("PASS: terrain PBR/physics separation gate" if ok else "FAIL: terrain gate")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
