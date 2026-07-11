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


def _stack_state_window(
    arrays: dict[str, np.ndarray],
    T: int = 25,
    *,
    end_idx: int | None = None,
) -> np.ndarray:
    """Build a fixed (T, F) state window ending at end_idx (default: trajectory end)."""
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
    n = min(f.shape[0] for f in feats)
    end = n if end_idx is None else max(1, min(int(end_idx), n))
    start = max(0, end - T)
    feats = [f[start:end] for f in feats]
    mat = np.concatenate(feats, axis=1)  # (t, F)
    if mat.shape[0] >= T:
        return mat[-T:]
    pad = np.repeat(mat[:1], T - mat.shape[0], axis=0)
    return np.concatenate([pad, mat], axis=0)


# Pre-registered keyword maps from REFLECT gt_failure_reason → evidence structure.
_VISION_TRUE_CUES = (
    "upside",
    "instead of",
    "wrong",
    "knife on top",
    "blocking",
    "on top of",
    "mistakenly picked",
    "mistakenly",
    "placements are",
    "should be placed",
    "not a fruit",
    "is a fruit",
    "orientation is wrong",
    "already occupied",
    "already inside",
    "door while",
    "never opened",
    "did not open",
    "attempted to open",
    "attempted to toggle",
)
_PROPRIO_TRUE_CUES = (
    "dropped",
    "drop",
    "gripper",
    "failed to",
    "never toggled",
    "never closed",
    "too full",
    "occupied by the pear",
    "slice the carrot",
    "toggle on",
    "toggle off",
)


def load_reflect_task_meta(tasks_json: Path | str | None) -> dict[str, dict[str, Any]]:
    """Map episode folder name → task metadata from tasks_real_world.json."""
    if tasks_json is None:
        return {}
    p = Path(tasks_json)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out: dict[str, dict[str, Any]] = {}
    for _k, v in raw.items():
        if not isinstance(v, dict):
            continue
        name = str(v.get("general_folder_name") or "").strip()
        if name:
            out[name] = v
    return out


def classify_reflect_stratum(failure_reason: str | None) -> str:
    """Map natural-language failure reason to E2/E3/E1 evidence structure."""
    r = (failure_reason or "").lower()
    is_v = any(c in r for c in _VISION_TRUE_CUES)
    is_p = any(c in r for c in _PROPRIO_TRUE_CUES)
    if is_v and not is_p:
        return "E2"
    if is_p and not is_v:
        return "E3"
    if is_v and is_p:
        # Prefer proprio when both fire (drop/fail verbs are decisive).
        return "E3"
    return "E1"


def _episode_meta(
    ep: Path,
    arrays: dict[str, np.ndarray],
    *,
    task_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Labels from REFLECT task metadata when available; else state heuristics."""
    name = ep.name
    tm = task_meta or {}
    task_name = str(tm.get("name") or name)
    reason = str(tm.get("gt_failure_reason") or "")
    meta: dict[str, Any] = {
        "task_instruction": f"complete task: {task_name}",
        "episode_name": name,
        "reward": 0,
        "failure_mode": "unknown_mode",
        "failure_reason": reason,
        "success_condition": tm.get("success_condition"),
    }
    if "stage" in arrays:
        st = np.asarray(arrays["stage"]).reshape(-1)
        meta["n_stages"] = int(len(np.unique(st))) if st.size else 0
    if reason:
        stratum = classify_reflect_stratum(reason)
        if stratum == "E2":
            meta["failure_mode"] = "wrong_object"
        elif stratum == "E3":
            meta["failure_mode"] = "no_close"
        else:
            meta["failure_mode"] = "no_progress"
        meta["stratum_from_meta"] = stratum
        return meta
    summary = _state_summary_from_arrays(arrays)
    if summary.get("gripper_mismatch"):
        meta["failure_mode"] = "no_close"
    elif float(summary.get("joint_motion_norm") or 0) < 0.05:
        meta["failure_mode"] = "no_progress"
    elif abs(float(summary.get("gripper_delta") or 0.0)) > 5.0:
        meta["failure_mode"] = "slip"
    else:
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


def _slice_arrays(
    arrays: dict[str, np.ndarray], end_idx: int
) -> dict[str, np.ndarray]:
    """Truncate all arrays to [0, end_idx)."""
    out: dict[str, np.ndarray] = {}
    for k, a in arrays.items():
        a = np.asarray(a)
        n = a.shape[0] if a.ndim >= 1 else 1
        e = max(1, min(int(end_idx), n))
        out[k] = a[:e]
    return out


def _write_rgb_cache(ep: Path, rgbs: list[np.ndarray], tag: str) -> list[str]:
    from PIL import Image

    cache = ep / f"_a8_rgb_cache_{tag}"
    cache.mkdir(exist_ok=True)
    paths: list[str] = []
    for j, rgb in enumerate(rgbs or [np.zeros((224, 224, 3), dtype=np.float32)]):
        pp = cache / f"frame_{j}.png"
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(pp)
        paths.append(str(pp))
    return paths


def _load_rgb_at_indices(ep: Path, indices: list[int]) -> list[np.ndarray]:
    """Load specific RGB frames by index from color zarr."""
    color_dir = ep / "videos" / "color"
    if not (color_dir / ".zarray").exists():
        return _load_rgb_frames(ep, max_frames=2)
    _ensure_imagecodecs_registered()
    try:
        import zarr

        arr = zarr.open(str(color_dir), mode="r")
        n = int(arr.shape[0])
        out = []
        for i in indices:
            ii = max(0, min(n - 1, int(i)))
            frame = np.asarray(arr[ii])
            if frame.ndim == 3 and frame.shape[-1] in (3, 4):
                out.append(frame[..., :3].astype(np.float32) / 255.0)
        return out
    except Exception:
        return _load_rgb_frames(ep, max_frames=2)


def build_a8b_cards_from_reflect_zarr(
    root: Path | str,
    *,
    limit: int | None = None,
    tasks_json: Path | str | None = None,
    include_early_success: bool = True,
) -> list[dict[str, Any]]:
    """Build multi-window A8b cards with E2/E3/E4 strata from REFLECT real data.

    For each failure episode:
      - failure window (trajectory end) → E2/E3/E1 from gt_failure_reason keywords
      - early window (~20% of trajectory) → E4 nominal/success (pre-failure control)
    """
    root = Path(root)
    task_map = load_reflect_task_meta(tasks_json)
    # default tasks path next to real_data root
    if not task_map:
        for cand in (
            root.parents[2] / "tasks_real_world.json" if len(root.parents) >= 3 else None,
            Path("outputs/eval/a8/raw/reflect/tasks_real_world.json"),
        ):
            if cand is not None and cand.exists():
                task_map = load_reflect_task_meta(cand)
                break

    eps = _find_episode_dirs(root)
    cards: list[dict[str, Any]] = []
    for i, ep in enumerate(eps):
        if limit is not None and i >= limit:
            break
        zdata = ep / "replay_buffer.zarr" / "data"
        arrays: dict[str, np.ndarray] = {}
        if zdata.exists():
            for k in STATE_KEYS + ("stage",):
                p = zdata / k
                if p.exists():
                    arr = _load_zarr_array(p)
                    if arr is not None:
                        arrays[k] = arr
        if not arrays:
            continue
        # trajectory length from joint or any array
        tlen = max(int(np.asarray(a).shape[0]) for a in arrays.values())
        tm = task_map.get(ep.name, {})
        meta = _episode_meta(ep, arrays, task_meta=tm)
        fail_stratum = meta.get("stratum_from_meta") or classify_reflect_stratum(
            meta.get("failure_reason")
        )
        # --- failure window (end) ---
        fail_arrays = arrays
        fail_summary = _state_summary_from_arrays(fail_arrays)
        fail_window = _stack_state_window(fail_arrays, T=25, end_idx=tlen)
        fail_rgbs = _load_rgb_at_indices(ep, [max(0, tlen // 2), max(0, tlen - 1)])
        fail_imgs = _write_rgb_cache(ep, fail_rgbs, "fail")
        fail_input = filter_allowed_fields(
            {
                "sample_id": f"a8b_{ep.name}__fail",
                "images": fail_imgs,
                "task_instruction": meta.get("task_instruction") or ep.name,
                "robot_state": fail_window.reshape(-1).astype(float).tolist(),
                "state_summary": fail_summary,
                "track": "a8b",
                "split": "reflect_real",
            },
            mode="model_input",
        )
        assert_no_leakage(fail_input)
        cards.append(
            {
                **fail_input,
                "binary_label": "failure",
                "failure_mode": meta.get("failure_mode"),
                "failure_reason": meta.get("failure_reason") or meta.get("failure_mode"),
                "reward": 0,
                "stratum": fail_stratum,
                "episode_dir": str(ep),
                "n_state_keys": len(arrays),
                "state_window_shape": list(fail_window.shape),
                "window": "failure_end",
            }
        )

        # --- early success/nominal window (E4) ---
        # Use the first ~50 frames (pre-contact cruise), not 20% of long episodes,
        # so proprio remains near-nominal while vision still shows the scene.
        if include_early_success and tlen >= 50:
            early_end = 50
            if "stage" in arrays:
                st = np.asarray(arrays["stage"]).reshape(-1)
                # if stage0 is short, keep first stage0 chunk capped at 80 frames
                zidx = np.where(st == st[0])[0]
                if len(zidx) >= 25:
                    early_end = int(min(80, max(50, zidx[min(len(zidx) - 1, 49)] + 1)))
            early_arrays = _slice_arrays(arrays, early_end)
            early_summary = _state_summary_from_arrays(early_arrays)
            early_summary = dict(early_summary)
            early_summary["gripper_mismatch"] = False
            early_window = _stack_state_window(arrays, T=25, end_idx=early_end)
            early_rgbs = _load_rgb_at_indices(ep, [0, max(0, early_end - 1)])
            early_imgs = _write_rgb_cache(ep, early_rgbs, "early50")
            early_input = filter_allowed_fields(
                {
                    "sample_id": f"a8b_{ep.name}__early",
                    "images": early_imgs,
                    "task_instruction": meta.get("task_instruction") or ep.name,
                    "robot_state": early_window.reshape(-1).astype(float).tolist(),
                    "state_summary": early_summary,
                    "track": "a8b",
                    "split": "reflect_real",
                },
                mode="model_input",
            )
            assert_no_leakage(early_input)
            cards.append(
                {
                    **early_input,
                    "binary_label": "success",
                    "failure_mode": "ground_truth",
                    "failure_reason": "pre_failure_nominal_window",
                    "reward": 1,
                    "stratum": "E4",
                    "episode_dir": str(ep),
                    "n_state_keys": len(arrays),
                    "state_window_shape": list(early_window.shape),
                    "window": "early_nominal",
                }
            )
    return cards
