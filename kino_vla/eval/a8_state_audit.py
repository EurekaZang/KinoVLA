"""A8b REFLECT multi-sensory state admission gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Filenames / suffixes that indicate raw observable robot state (not labels).
STATE_NAME_HINTS = (
    "proprio",
    "robot_state",
    "joint",
    "gripper",
    "ee_pose",
    "end_effector",
    "qpos",
    "qvel",
    "force",
    "torque",
    "wrench",
    "contact",
    "low_dim",
    "states",
    "obs_dict",
    "action",
)

STATE_SUFFIXES = (".npy", ".npz", ".pkl", ".pickle", ".json", ".csv", ".txt", ".pt", ".pth")
RGB_HINTS = ("rgb", "color", "image", "frame", "cam", "front", "wrist")


def _is_state_file(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in STATE_SUFFIXES and path.suffix.lower() not in {
        ".npz5",
        ".npz5.gz",
    }:
        # allow extensionless state dumps only with strong name hints
        if not any(h in name for h in STATE_NAME_HINTS):
            return False
    return any(h in name for h in STATE_NAME_HINTS)


def _is_rgb_file(path: Path) -> bool:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return False
    name = path.name.lower()
    return any(h in name for h in RGB_HINTS) or True  # any image counts as visual evidence


def _episode_dirs(root: Path) -> list[Path]:
    """Heuristic episode roots: directories that contain images or nested records."""
    if not root.exists():
        return []
    # Prefer directories one level under common containers.
    candidates: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_dir():
            continue
        files = list(p.iterdir()) if p.exists() else []
        has_img = any(f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"} for f in files)
        has_state = any(f.is_file() and _is_state_file(f) for f in files)
        if has_img or has_state:
            candidates.append(p)
    # Dedup nested: keep shallowest paths when possible.
    keep: list[Path] = []
    for c in sorted(candidates, key=lambda x: len(x.parts)):
        if any(str(c).startswith(str(k) + "/") for k in keep):
            continue
        keep.append(c)
    return keep


def audit_reflect_state(root: Path | str) -> dict[str, Any]:
    """Inspect extracted REFLECT data for usable multi-sensory robot state.

    Pass criteria:
    - root exists
    - ≥1 episode with RGB evidence
    - ≥1 episode with raw state-like files
    - ≥10 episodes with BOTH (enough for strata); softer note if fewer
    """
    root = Path(root)
    if not root.exists():
        return {
            "pass": False,
            "n_episodes": 0,
            "n_rgb_episodes": 0,
            "n_state_episodes": 0,
            "n_multisensory_episodes": 0,
            "state_fields": [],
            "reason": f"reflect root missing: {root}",
        }

    episodes = _episode_dirs(root)
    state_fields: set[str] = set()
    n_rgb = 0
    n_state = 0
    n_both = 0
    examples: list[dict[str, Any]] = []

    for ep in episodes:
        files = [f for f in ep.rglob("*") if f.is_file()]
        rgb_files = [f for f in files if _is_rgb_file(f)]
        state_files = [f for f in files if _is_state_file(f)]
        # Also scan JSON for state keys.
        json_state_keys: list[str] = []
        for jf in files:
            if jf.suffix.lower() != ".json":
                continue
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            if isinstance(data, dict):
                for k in data:
                    kl = str(k).lower()
                    if any(h in kl for h in STATE_NAME_HINTS):
                        json_state_keys.append(str(k))
                        state_fields.add(str(k))
        if state_files:
            for sf in state_files:
                state_fields.add(sf.name)
        has_rgb = bool(rgb_files)
        has_state = bool(state_files) or bool(json_state_keys)
        n_rgb += int(has_rgb)
        n_state += int(has_state)
        n_both += int(has_rgb and has_state)
        if len(examples) < 5 and (has_rgb or has_state):
            examples.append(
                {
                    "path": str(ep),
                    "n_rgb": len(rgb_files),
                    "n_state_files": len(state_files),
                    "json_state_keys": json_state_keys[:10],
                }
            )

    min_both = 10
    passed = n_both >= min_both and n_state > 0 and n_rgb > 0
    if not root.exists():
        reason = "missing root"
    elif n_rgb == 0:
        reason = "no RGB episodes found"
    elif n_state == 0:
        reason = "no robot-state / proprio files found"
    elif n_both < min_both:
        reason = f"insufficient multi-sensory episodes: {n_both} < {min_both}"
    else:
        reason = "ok"

    return {
        "pass": passed,
        "n_episodes": len(episodes),
        "n_rgb_episodes": n_rgb,
        "n_state_episodes": n_state,
        "n_multisensory_episodes": n_both,
        "n_state_episodes_alias": n_state,  # for tests
        "state_fields": sorted(state_fields)[:50],
        "examples": examples,
        "reason": reason,
        "root": str(root),
    }
