"""REFLECT multi-sensory episode loader for A8b.

REFLECT real_data layout (per episode directory):
  <episode>/
    videos/color/<frame_id>          # RGB frames (raw image bytes or arrays)
    videos/depth/...
    replay_buffer.zarr/
      data/robot_joint/
      data/robot_joint_vel/
      data/robot_eef_pose/
      data/gripper_pos/
      data/gripper_state/
      data/gripper_force/
      data/action/
      data/stage/
      meta/episode_ends/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.eval.a8_leakage import assert_no_leakage, filter_allowed_fields
from kino_vla.eval.a8_strata import label_stratum


STATE_KEYS = (
    "robot_joint",
    "robot_joint_vel",
    "robot_eef_pose",
    "robot_eef_pose_vel",
    "gripper_pos",
    "gripper_state",
    "gripper_force",
    "action",
)


def _find_episode_dirs(root: Path) -> list[Path]:
    """Find episode directories that contain replay_buffer.zarr."""
    if not root.exists():
        return []
    eps = []
    for p in sorted(root.rglob("replay_buffer.zarr")):
        if p.is_dir():
            eps.append(p.parent)
    # de-nest
    keep: list[Path] = []
    for c in sorted(eps, key=lambda x: len(x.parts)):
        if any(str(c).startswith(str(k) + "/") for k in keep):
            continue
        keep.append(c)
    return keep


def _ensure_imagecodecs_registered() -> None:
    """Register imagecodecs JPEG-XL etc. with numcodecs (needed for REFLECT RGB zarr)."""
    try:
        import imagecodecs.numcodecs as icn

        icn.register_codecs()
    except Exception:
        # Best-effort; state arrays only need blosc via numcodecs.
        pass


def _load_zarr_array(path: Path) -> np.ndarray | None:
    """Load a zarr array directory (blosc/zstd compressed REFLECT chunks)."""
    zarray = path / ".zarray"
    if not zarray.exists():
        return None
    _ensure_imagecodecs_registered()
    try:
        import zarr

        arr = zarr.open(str(path), mode="r")
        return np.asarray(arr)
    except Exception:
        return None


def _load_rgb_frames(ep: Path, max_frames: int = 4) -> list[np.ndarray]:
    """Load start/mid/end RGB frames from REFLECT videos/color zarr (JPEG-XL)."""
    color_dir = ep / "videos" / "color"
    if not color_dir.exists():
        alts = list(ep.rglob("color"))
        color_dir = alts[0] if alts else color_dir
    if not color_dir.exists():
        return []

    _ensure_imagecodecs_registered()
    # Preferred: zarr array of shape (T,H,W,3)
    if (color_dir / ".zarray").exists():
        try:
            import zarr

            arr = zarr.open(str(color_dir), mode="r")
            n = int(arr.shape[0]) if hasattr(arr, "shape") else 0
            if n <= 0:
                return []
            idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
            idxs = sorted(set(max(0, min(n - 1, i)) for i in idxs))[:max_frames]
            out = []
            for i in idxs:
                frame = np.asarray(arr[i])
                if frame.ndim == 3 and frame.shape[-1] in (3, 4):
                    out.append(frame[..., :3].astype(np.float32) / 255.0)
            if out:
                return out
        except Exception:
            pass

    # Fallback: individual image files
    files = sorted(
        [p for p in color_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    )
    if not files:
        return []
    idxs = [0, len(files) // 2, len(files) - 1] if len(files) >= 3 else list(range(len(files)))
    idxs = sorted(set(max(0, min(len(files) - 1, i)) for i in idxs))[:max_frames]
    out = []
    from PIL import Image
    import io

    for i in idxs:
        p = files[i]
        data = p.read_bytes()
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            out.append(np.asarray(img, dtype=np.float32) / 255.0)
        except Exception:
            continue
    return out


def _state_summary_from_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "robot_joint" in arrays:
        j = np.asarray(arrays["robot_joint"], dtype=np.float64)
        if j.ndim == 1:
            j = j.reshape(1, -1)
        if len(j) > 1:
            summary["joint_motion_norm"] = float(np.linalg.norm(j[-1] - j[0]))
        else:
            summary["joint_motion_norm"] = 0.0
        summary["joint_mean"] = float(np.nanmean(j))
        summary["state_dim"] = int(j.shape[-1])
    if "gripper_pos" in arrays:
        g = np.asarray(arrays["gripper_pos"], dtype=np.float64).reshape(-1)
        # REFLECT gripper_pos is often ~0-120 (mm-like), not meters.
        summary["gripper_width"] = float(g[-1]) if g.size else 0.0
        summary["gripper_width_norm"] = float(g[-1] / 120.0) if g.size else 0.0
        summary["gripper_delta"] = float(g[-1] - g[0]) if g.size > 1 else 0.0
    if "gripper_state" in arrays:
        gs = np.asarray(arrays["gripper_state"]).reshape(-1)
        # Values observed: 0 open-ish, 4 closed-ish in some episodes.
        last = float(gs[-1]) if gs.size else 0.0
        summary["gripper_cmd_closed"] = 1.0 if last >= 2.0 else 0.0
        summary["gripper_state_raw"] = last
    if "gripper_force" in arrays:
        f = np.asarray(arrays["gripper_force"], dtype=np.float64).reshape(-1)
        summary["gripper_force_mean"] = float(np.nanmean(f)) if f.size else 0.0
        summary["gripper_force_max"] = float(np.nanmax(f)) if f.size else 0.0
    # mismatch heuristic: commanded closed but still wide
    if "gripper_width" in summary and "gripper_cmd_closed" in summary:
        width = summary["gripper_width"]
        closed = summary["gripper_cmd_closed"] >= 0.5
        # width units ~mm in REFLECT; treat >20 as open/wide
        summary["gripper_mismatch"] = bool(closed and width > 20.0)
    return summary


def _stack_state_window(arrays: dict[str, np.ndarray], T: int = 25) -> np.ndarray:
    """Build a fixed (T, F) state window from available arrays (tail of trajectory)."""
    feats = []
    for key in STATE_KEYS:
        if key not in arrays:
            continue
        a = np.asarray(arrays[key], dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        elif a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
        feats.append(a)
    if not feats:
        return np.zeros((T, 8), dtype=np.float32)
    # align lengths
    n = min(f.shape[0] for f in feats)
    feats = [f[-n:] for f in feats]
    mat = np.concatenate(feats, axis=1)  # (n, F)
    if mat.shape[0] >= T:
        return mat[-T:]
    # pad by repeating first row
    pad = np.repeat(mat[:1], T - mat.shape[0], axis=0)
    return np.concatenate([pad, mat], axis=0)


def _episode_meta(ep: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Best-effort labels: REFLECT real episodes are failure demonstrations by construction."""
    name = ep.name
    meta: dict[str, Any] = {
        "task_instruction": f"complete task: {name}",
        "episode_name": name,
        "reward": 0,  # real demos are failure cases
        "failure_mode": "unknown_mode",
    }
    if "stage" in arrays:
        st = np.asarray(arrays["stage"]).reshape(-1)
        meta["n_stages"] = int(len(np.unique(st))) if st.size else 0
    summary = _state_summary_from_arrays(arrays)
    if summary.get("gripper_mismatch"):
        meta["failure_mode"] = "no_close"
    elif float(summary.get("joint_motion_norm") or 0) < 0.05:
        meta["failure_mode"] = "no_progress"
    elif abs(float(summary.get("gripper_delta") or 0.0)) > 5.0:
        meta["failure_mode"] = "slip"
    else:
        # Default proprio-true contact/execution failure for REFLECT real demos.
        meta["failure_mode"] = "no_close"
    return meta


def audit_reflect_root(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    eps = _find_episode_dirs(root)
    n_rgb = 0
    n_state = 0
    n_both = 0
    state_fields: set[str] = set()
    examples = []
    for ep in eps:
        zdata = ep / "replay_buffer.zarr" / "data"
        has_state = False
        if zdata.exists():
            for k in STATE_KEYS:
                if (zdata / k).exists():
                    has_state = True
                    state_fields.add(k)
        has_rgb = (ep / "videos" / "color").exists() or any(ep.rglob("color"))
        n_rgb += int(has_rgb)
        n_state += int(has_state)
        n_both += int(has_rgb and has_state)
        if len(examples) < 5:
            examples.append({"path": str(ep), "rgb": has_rgb, "state": has_state})
    passed = n_both >= 10 and n_state > 0 and n_rgb > 0
    reason = "ok" if passed else (
        "missing root" if not root.exists() else
        "no RGB episodes found" if n_rgb == 0 else
        "no robot-state / proprio files found" if n_state == 0 else
        f"insufficient multi-sensory episodes: {n_both} < 10"
    )
    return {
        "pass": passed,
        "n_episodes": len(eps),
        "n_rgb_episodes": n_rgb,
        "n_state_episodes": n_state,
        "n_multisensory_episodes": n_both,
        "state_fields": sorted(state_fields),
        "examples": examples,
        "reason": reason,
        "root": str(root),
    }


def build_a8b_cards_from_reflect_zarr(
    root: Path | str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    root = Path(root)
    eps = _find_episode_dirs(root)
    cards: list[dict[str, Any]] = []
    for i, ep in enumerate(eps):
        if limit is not None and i >= limit:
            break
        zdata = ep / "replay_buffer.zarr" / "data"
        arrays: dict[str, np.ndarray] = {}
        if zdata.exists():
            for k in STATE_KEYS:
                p = zdata / k
                if p.exists():
                    arr = _load_zarr_array(p)
                    if arr is not None:
                        arrays[k] = arr
        rgbs = _load_rgb_frames(ep, max_frames=2)
        if not rgbs and not arrays:
            continue
        # save temporary RGB paths into card by writing to a cache dir next to cards later;
        # for now embed as arrays via dataset writer using images list of temp files.
        state_summary = _state_summary_from_arrays(arrays)
        meta = _episode_meta(ep, arrays)
        stratum = label_stratum(meta, state_summary)
        state_window = _stack_state_window(arrays, T=25)
        sid = f"a8b_{ep.name}"
        # write rgb to episode-local cache for path-based pipeline
        img_paths = []
        cache = ep / "_a8_rgb_cache"
        cache.mkdir(exist_ok=True)
        from PIL import Image

        for j, rgb in enumerate(rgbs or [np.zeros((224, 224, 3), dtype=np.float32)]):
            pp = cache / f"frame_{j}.png"
            if not pp.exists():
                Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(pp)
            img_paths.append(str(pp))
        model_input = filter_allowed_fields(
            {
                "sample_id": sid,
                "images": img_paths,
                "task_instruction": meta.get("task_instruction") or ep.name,
                "robot_state": state_window.reshape(-1).astype(float).tolist(),
                "state_summary": state_summary,
                "track": "a8b",
                "split": "reflect_real",
            },
            mode="model_input",
        )
        assert_no_leakage(model_input)
        # binary label: REFLECT real demos are failure demonstrations by construction
        binary = "failure"
        cards.append(
            {
                **model_input,
                "binary_label": binary,
                "failure_mode": meta.get("failure_mode"),
                "failure_reason": meta.get("failure_mode"),
                "reward": 0,
                "stratum": stratum,
                "episode_dir": str(ep),
                "n_state_keys": len(arrays),
                "state_window_shape": list(state_window.shape),
            }
        )
    return cards
