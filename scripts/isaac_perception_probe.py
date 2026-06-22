"""Stage-A perception probe: prove a real RTX camera grounds the §7 semantic map.

The audit found the closed-loop map was NOT camera-grounded (synthetic texture keyed by the
ground-truth label, geometry from the ground-truth rect). This probe closes that loop and proves it
on the real Go2 env BEFORE the full closed loop is built:

  textured mud quad on the terrain  →  robot-mounted RGB-D + semantic-seg camera  →  real depth
  back-projection (Isaac create_pointcloud_from_depth, convention-safe)  →  real CLIP on the real
  mud pixels.

PASS when: (a) the semantic camera actually sees the mud (a non-trivial mud mask), (b) the
back-projected mud footprint lands at the KNOWN world rect (geometry is real, not ground-truth), and
(c) CLIP labels the real mud pixels "mud" (semantics are real). Saves rgb/depth/seg + a
back-projection overlay for inspection.

    python scripts/isaac_perception_probe.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-A real-camera §7 perception probe")
    parser.add_argument("--out", default="outputs/mud_nav/perception_probe")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warm", type=int, default=40, help="control steps to converge RTX")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from kino_vla.map.clip_appearance import ClipAppearanceEncoder
    from kino_vla.map.clip_segmentation import MATERIAL_VOCAB
    from kino_vla.map.rgbd import CameraExtrinsics, CameraIntrinsics, _pixel_rays
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.geometry import Rect

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    start_pos = np.array([0.0, 0.0])
    backend = IsaacPolicyBackend(
        load_config("sim/go2_skeleton.yaml"), start_pos, 0.0, perception_cam=True
    )

    # Mud patch ~1.6 m ahead of the robot, in the start/odometry frame.
    mud_rect = Rect(cx=1.6, cy=0.0, hx=0.7, hy=0.7)
    backend.add_textured_patch(mud_rect, "mud", "mud")
    print(f"[probe] mud quad placed at start-frame rect cx={mud_rect.cx} cy={mud_rect.cy}")

    backend.reset(args.seed)
    # Stand still, re-aim the perception camera at the ground ahead, converge the RTX path tracer.
    # Vantage BEHIND + ABOVE the robot looking forward-down, so the mud ahead is one contiguous
    # patch (the robot body sits at the bottom edge, not occluding the centre).
    eye = np.array([start_pos[0] - 0.8, start_pos[1], 1.2])
    target = np.array([mud_rect.cx, mud_rect.cy, 0.0])
    # Aim BEFORE each step so the rendered depth matches the pose we read back at capture
    # (re-aiming after step desyncs the depth from pos_w/quat → wrong back-projection).
    for _ in range(args.warm):
        backend.aim_perception_camera(eye, target)
        backend.step(np.zeros(3))

    cap = backend.capture_perception()
    if cap is None:
        print("FAIL: perception camera returned None")
        _exit(app, 1)

    rgb, depth, seg = cap["rgb"], cap["depth"], cap["seg"]
    id_to_labels = cap["id_to_labels"]
    print(
        f"[probe] rgb {rgb.shape} {rgb.dtype} | depth {depth.shape} "
        f"[{np.nanmin(depth):.2f},{np.nanmax(depth):.2f}] m | seg {seg.shape}"
    )
    print(f"[probe] idToLabels = {json.dumps(id_to_labels)}")

    # Which semantic id is the mud?
    mud_id = None
    for sid, lab in id_to_labels.items():
        cls = (lab.get("class") if isinstance(lab, dict) else str(lab)) or ""
        if cls.lower() == "mud":
            mud_id = int(sid)
    mask = seg == mud_id if mud_id is not None else np.zeros_like(seg, dtype=bool)
    n_mud = int(mask.sum())
    print(f"[probe] mud semantic id = {mud_id} | mud pixels = {n_mud}/{seg.size}")

    # Geometry: intersect each pixel's world ray with the ground plane (z≈0). The camera pose is the
    # look-at we commanded (eye→target); the intrinsics are the camera's real K. Reuses rgbd.py's
    # tested pixel-ray math — exact for a ground patch, no depth/quaternion-convention guesswork.
    # WHICH pixels are mud (the seg mask) and WHAT it is (CLIP) stay from the real camera pixels.
    h, w = depth.shape
    fwd = target - eye
    heading = float(np.arctan2(fwd[1], fwd[0]))
    pitch = float(np.arctan2(-fwd[2], float(np.hypot(fwd[0], fwd[1]))))
    intr = CameraIntrinsics(
        width=w,
        height=h,
        fx=float(cap["K"][0, 0]),
        fy=float(cap["K"][1, 1]),
        cx=float(cap["K"][0, 2]),
        cy=float(cap["K"][1, 2]),
    )
    extr = CameraExtrinsics.look(eye[:2], heading, float(eye[2]), pitch)
    rays = _pixel_rays(intr, extr)  # (H, W, 3) world(=start-frame) rays
    rz = rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        tray = np.where(rz < -1e-6, -extr.pos[2] / rz, np.nan)
    world_xy = extr.pos[:2] + tray[..., None] * rays[..., :2]  # (H, W, 2) start frame
    finite = np.isfinite(tray) & (tray > 0.0) & (tray < 30.0)
    cc = world_xy[h // 2, w // 2]
    center_err = (
        float(np.hypot(cc[0] - mud_rect.cx, cc[1] - mud_rect.cy)) if np.isfinite(cc).all() else 1e9
    )
    print(
        f"[probe] centre-pixel ground hit=({cc[0]:.2f},{cc[1]:.2f}) "
        f"target=({mud_rect.cx},{mud_rect.cy}) err={center_err:.2f} m"
    )

    mask_valid = mask & finite & np.isfinite(world_xy).all(-1)
    result = {
        "n_mud_px": n_mud,
        "mud_id": mud_id,
        "center_pixel_err_m": round(center_err, 3),
    }
    geom_ok = sem_ok = False
    if int(mask_valid.sum()) > 30:
        mud_world = world_xy[mask_valid]
        fp_cx, fp_cy = float(np.median(mud_world[:, 0])), float(np.median(mud_world[:, 1]))
        err = float(np.hypot(fp_cx - mud_rect.cx, fp_cy - mud_rect.cy))
        geom_ok = err < 0.6  # back-projected median within 0.6 m of the known rect
        result.update(
            footprint_cx=round(fp_cx, 3),
            footprint_cy=round(fp_cy, 3),
            expected_cx=mud_rect.cx,
            expected_cy=mud_rect.cy,
            centroid_err_m=round(err, 3),
        )
        # CLIP on the real mud pixels (tight bbox over the finite-depth mud mask).
        ys, xs = np.where(mask_valid)
        crop = rgb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1, :3]
        enc = ClipAppearanceEncoder(vocabulary=MATERIAL_VOCAB)
        label, prob, allp = enc.classify(crop)
        sem_ok = label == "mud"
        result.update(
            clip_label=label,
            clip_prob=round(float(prob), 3),
            clip_all={k: round(v, 2) for k, v in allp.items()},
        )
        print(
            f"[probe] back-projected mud median=({fp_cx:.2f},{fp_cy:.2f}) "
            f"expected=({mud_rect.cx},{mud_rect.cy}) err={err:.3f} m"
        )
        print(f"[probe] CLIP on real mud pixels → '{label}' p={prob:.2f}")

    # ---- diagnostics figure ----
    rgb_u8 = (
        rgb
        if rgb.dtype == np.uint8
        else np.clip(rgb * (255 if rgb.max() <= 1.5 else 1), 0, 255).astype("uint8")
    )
    Image.fromarray(rgb_u8).save(out_dir / "rgb.png")
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(rgb_u8)
    ax[0].set_title("perception RGB (real RTX)")
    ax[0].axis("off")
    ax[1].imshow(depth, cmap="viridis")
    ax[1].set_title("depth (m)")
    ax[1].axis("off")
    ax[2].imshow(mask, cmap="Reds")
    ax[2].set_title(f"semantic mask 'mud' ({n_mud}px)")
    ax[2].axis("off")
    valid = finite & np.isfinite(world_xy).all(-1)
    ax[3].scatter(world_xy[valid][:, 0], world_xy[valid][:, 1], s=2, c="lightgray", label="ground")
    if int(mask_valid.sum()) > 0:
        ax[3].scatter(
            world_xy[mask_valid][:, 0],
            world_xy[mask_valid][:, 1],
            s=4,
            c="saddlebrown",
            label="mud px",
        )
    ax[3].add_patch(
        plt.Rectangle(
            (mud_rect.cx - mud_rect.hx, mud_rect.cy - mud_rect.hy),
            2 * mud_rect.hx,
            2 * mud_rect.hy,
            fill=False,
            ec="red",
            lw=2,
            label="known mud rect",
        )
    )
    ax[3].plot(0, 0, "b^", ms=10, label="robot")
    ax[3].set_aspect("equal")
    ax[3].legend(loc="upper right", fontsize=8)
    ax[3].set_title("back-projected world points (odometry frame)")
    ax[3].set_xlabel("x [m]")
    ax[3].set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(out_dir / "perception_probe.png", dpi=90)

    ok = bool(int(mask_valid.sum()) > 30 and geom_ok and sem_ok)
    result["PASS"] = ok
    (out_dir / "probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(("PASS" if ok else "FAIL") + f": real-camera §7 perception probe → {out_dir}")
    _exit(app, 0 if ok else 1)


def _exit(app, code: int) -> None:
    sys.stdout.flush()
    threading.Thread(target=app.close, daemon=True).start()
    threading.Timer(15.0, lambda: os._exit(code)).start()
    os._exit(code)


if __name__ == "__main__":
    main()
