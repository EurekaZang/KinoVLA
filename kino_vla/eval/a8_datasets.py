"""A8 dataset builders: Guardian VQA cards and REFLECT multi-sensory cards."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from kino_vla.eval.a8_leakage import assert_no_leakage, filter_allowed_fields
from kino_vla.eval.a8_strata import label_stratum

SUCCESS_TOKENS = frozenset({"success", "yes", "true", "1", "pass", "nominal"})
FAIL_TOKENS = frozenset({"failure", "fail", "no", "false", "0", "error"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_rgb(path: Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def parse_binary_label(text: str | None, reward: float | int | None = None) -> str:
    """Map Guardian-style answers to success|failure.

    Guardian InternVL answers look like:
      ``<answer> False </answer> <category> ...``
      ``<answer> True </answer> <category> success </category>``
    where True=success and False=failure.
    """
    t = (text or "").strip()
    # Prefer explicit <answer> tags over reward when both exist (reward can be missing).
    m = re.search(r"<answer>\s*(true|false|success|failure|yes|no|1|0)\s*</answer>", t, flags=re.I)
    if m:
        tok = m.group(1).lower()
        if tok in {"true", "success", "yes", "1"}:
            return "success"
        if tok in {"false", "failure", "no", "0"}:
            return "failure"
    if reward is not None:
        try:
            return "success" if float(reward) >= 0.5 else "failure"
        except (TypeError, ValueError):
            pass
    tl = t.lower()
    # category success
    if re.search(r"<category>\s*success\s*</category>", tl):
        return "success"
    if any(tok in tl for tok in ("failure", "failed", "fail", "error", "wrong object", "no close", "no progress")):
        # avoid matching "successfully failed" nonsense; still treat explicit fail words as failure
        if "success" in tl and "fail" not in tl:
            return "success"
        return "failure"
    if any(tok in tl for tok in ("success", "succeeded", "completed successfully")):
        return "success"
    head = re.split(r"[\s:,.]", tl)[0] if tl else ""
    if head in SUCCESS_TOKENS:
        return "success"
    if head in FAIL_TOKENS:
        return "failure"
    return "unknown"


def extract_gt_from_internvl_row(row: dict[str, Any]) -> dict[str, Any]:
    """Pull ground-truth answer/category from InternVL conversation or metadata fields."""
    gt_answer = row.get("ground_truth_answer") or row.get("label")
    category = None
    if gt_answer is None:
        conv = row.get("conversations") or []
        # last gpt turn
        for turn in reversed(conv):
            if turn.get("from") in {"gpt", "assistant"}:
                gt_answer = turn.get("value")
                break
    if isinstance(gt_answer, str):
        mcat = re.search(r"<category>\s*(.*?)\s*</category>", gt_answer, flags=re.I | re.S)
        if mcat:
            category = mcat.group(1).strip()
    reward = row.get("reward", row.get("execution_reward", row.get("planning_reward")))
    binary = parse_binary_label(str(gt_answer) if gt_answer is not None else None, reward)
    failure_mode = row.get("failure_mode") or category
    return {
        "ground_truth_answer": gt_answer,
        "binary_label": binary,
        "failure_mode": failure_mode,
        "failure_reason": row.get("failure_reason") or category,
        "reward": reward,
        "task_instruction": row.get("task_instruction")
        or row.get("instruction")
        or _instruction_from_conversations(row),
    }


def _instruction_from_conversations(row: dict[str, Any]) -> str:
    conv = row.get("conversations") or []
    for turn in conv:
        if turn.get("from") in {"human", "user"}:
            val = str(turn.get("value", ""))
            # strip image placeholders
            val = re.sub(r"<image>|\n+", " ", val).strip()
            return val[:500]
    return "Verify whether the robot execution succeeded or failed."


def resolve_image_paths(row: dict[str, Any], split_root: Path) -> list[Path]:
    images = row.get("image") or row.get("images") or []
    if isinstance(images, str):
        images = [images]
    paths: list[Path] = []
    for rel in images:
        rel_s = str(rel)
        p = Path(rel_s)
        if p.is_absolute() and p.exists():
            paths.append(p)
            continue
        # Common layouts after tar extract:
        #   split/records/<rel>
        #   split/records/records/<rel>   (double-nested extract)
        #   split/<rel>
        #   split/records/<rel without records/ prefix>
        rel_noprefix = rel_s.removeprefix("records/").removeprefix("data/failure_forge/data/")
        candidates = [
            split_root / rel_s,
            split_root / "records" / rel_s,
            split_root / "records" / "records" / rel_s,
            split_root / "records" / rel_noprefix,
            split_root / "records" / "records" / rel_noprefix,
            split_root / rel_noprefix,
            # OOD bundle paths like data/failure_forge/data/robofail_dataset/records/...
            split_root / "records" / Path(rel_noprefix).name,
        ]
        # also try stripping leading dataset folders after /records/
        if "/records/" in rel_s:
            tail = rel_s.split("/records/", 1)[1]
            candidates.extend(
                [
                    split_root / "records" / tail,
                    split_root / "records" / "records" / tail,
                    split_root / tail,
                ]
            )
        found = next((c for c in candidates if c.exists()), None)
        if found is not None:
            paths.append(found)
    return paths


def build_a8a_cards_from_split(
    split_root: Path | str,
    *,
    split_name: str,
    stage: str = "execution",  # execution|planning
    prompt_policy: str = "vanilla",
    max_images: int = 2,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert one Guardian split directory into leakage-safe A8a cards."""
    split_root = Path(split_root)
    # Prefer internVL jsonl; fall back to metadata.
    candidates = [
        split_root / f"internVL_dataset_{stage}_{prompt_policy}.jsonl",
        split_root / f"internVL_dataset_{stage}_vanilla.jsonl",
        split_root / f"internVL_dataset_{stage}_thinking.jsonl",
        split_root / f"metadata_{stage}.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    src = None
    for c in candidates:
        rows = _read_jsonl(c)
        if rows:
            src = c
            break
    if not rows:
        return []

    cards: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        gt = extract_gt_from_internvl_row(row)
        img_paths = resolve_image_paths(row, split_root)
        if not img_paths:
            # metadata-only rows may list images under records/
            for key in ("images", "image"):
                if key in row:
                    img_paths = resolve_image_paths({key: row[key]}, split_root)
                    break
        sid = str(row.get("id") or row.get("episode_id") or f"{split_name}_{stage}_{i}")
        model_input = filter_allowed_fields(
            {
                "sample_id": sid,
                "images": [str(p) for p in img_paths[:max_images]],
                "task_instruction": gt["task_instruction"],
                "split": split_name,
                "track": "a8a",
                "prompt_policy": f"{stage}_{prompt_policy}",
            },
            mode="model_input",
        )
        assert_no_leakage(model_input)
        card = {
            **model_input,
            # eval-only sidecar fields (not for model prompt builder)
            "binary_label": gt["binary_label"],
            "failure_mode": gt["failure_mode"],
            "failure_reason": gt["failure_reason"],
            "reward": gt["reward"],
            "ground_truth_answer": gt["ground_truth_answer"],
            "source_jsonl": str(src) if src else None,
            "stage": stage,
        }
        cards.append(card)
    return cards


def write_a8a_dataset(out_dir: Path | str, cards: list[dict[str, Any]], card_meta: dict[str, Any]) -> Path:
    """Write A8a cards + optional RGB frames npz for Kino-SFT-compatible loading."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    frames: dict[str, np.ndarray] = {}
    for c in cards:
        sid = c["sample_id"]
        # Build a recovery-like record so existing loaders can be adapted; A8 eval uses cards directly.
        rgb_list = []
        for p in c.get("images") or []:
            pp = Path(p)
            if pp.exists():
                try:
                    rgb_list.append(_load_rgb(pp))
                except Exception:
                    continue
        if not rgb_list:
            # placeholder black frame keeps shape stable for smoke tests; real eval skips missing
            rgb_list = [np.zeros((224, 224, 3), dtype=np.float32)]
        rgb = np.stack(rgb_list, axis=0)
        # dummy proprio zeros (vision-only A8a); width must match Go2 FEATURE_SCHEMA (11)
        proprio = np.zeros((25, 11), dtype=np.float32)
        depth = np.zeros(rgb.shape[:-1], dtype=np.float32)
        frames[f"{sid}__rgb"] = rgb
        frames[f"{sid}__depth"] = depth
        frames[f"{sid}__proprio"] = proprio
        label = c.get("binary_label", "unknown")
        # Map Guardian success/failure onto the KiNO SFT schema:
        # attribution carries the verification label; primitive must be from §5 library.
        if label == "success":
            thought = (
                "The robot completed the instructed subtask successfully. "
                "Answer: <answer> True </answer> <category> success </category>."
            )
            reason = "execution_success"
        else:
            thought = (
                "The robot failed the instructed subtask. "
                "Answer: <answer> False </answer> <category> failure </category>."
            )
            reason = "execution_failure"
        rec = {
            "sample_id": sid,
            "snapshot": {
                "operator_name": "a8a_failure_verify",
                "appearance_class": str(c.get("failure_mode") or "unknown"),
                "t": 0.0,
                "pose_xy": [0.0, 0.0],
                "heading": 0.0,
                "prior_outputs": [],
                "privileged_theta": {},
                "monitor_channel": "a8a",
                "rgb_shape": list(rgb.shape),
                "depth_shape": list(depth.shape),
                "proprio_shape": list(proprio.shape),
            },
            "ground_truth": {
                "category": label,
                "ab_class": "A",
                "theta": {},
                "feasible": ["Hold_and_Request"],
                "canonical_primitive": "Hold_and_Request",
                "is_sudden_trap": False,
            },
            "annotation": {
                "thought": thought,
                "attribution": label,
                "attribution_raw": label,
                "action": {
                    "primitive": "Hold_and_Request",
                    "params": {"reason": reason},
                },
            },
            "verdict": {"keep": True, "reason": "a8a"},
            "target_theta": [0.0, 0.0, 0.0, 0.0],
            "ambiguity_pair": None,
            "a8": {
                "track": "a8a",
                "task_instruction": c.get("task_instruction"),
                "binary_label": label,
                "stage": c.get("stage"),
                "prompt_policy": c.get("prompt_policy"),
            },
            # keep eval-only in record but training prompt builder must not read them into user text beyond instruction
            "failure_mode": c.get("failure_mode"),
            "reward": c.get("reward"),
        }
        records.append(rec)

    (out / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    # cards.jsonl is the leakage-audited view
    safe_cards = []
    for c in cards:
        mi = filter_allowed_fields(c, mode="model_input")
        assert_no_leakage(mi)
        safe_cards.append({**mi, "binary_label": c.get("binary_label"), "stage": c.get("stage")})
    (out / "cards.jsonl").write_text("\n".join(json.dumps(r) for r in safe_cards) + "\n")
    np.savez_compressed(out / "frames.npz", **frames)
    meta = {
        **card_meta,
        "n_records": len(records),
        "n_cards": len(cards),
        "track": "a8a",
    }
    (out / "dataset_card.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def summarize_state_window(state: np.ndarray | list[float] | dict[str, Any] | None) -> dict[str, Any]:
    """Build a small observable state summary for strata + text injection."""
    if state is None:
        return {}
    if isinstance(state, dict):
        return {k: state[k] for k in list(state)[:20]}
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {}
    return {
        "state_dim": int(arr.size),
        "joint_motion_norm": float(np.linalg.norm(arr - arr.mean())),
        "state_mean": float(arr.mean()),
        "state_std": float(arr.std()),
        "gripper_delta": float(arr[-1]) if arr.size else 0.0,
    }


def build_a8b_cards_from_reflect(
    reflect_root: Path | str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Scan REFLECT extract for multi-sensory episodes and emit A8b cards.

    This is intentionally defensive: REFLECT layout varies; we discover RGB + state files
    per episode directory and attach strata from nearby metadata when present.
    """
    root = Path(reflect_root)
    if not root.exists():
        return []

    # episode heuristic: directories containing images
    ep_dirs: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_dir():
            continue
        imgs = list(p.glob("*.png")) + list(p.glob("*.jpg"))
        if imgs:
            ep_dirs.append(p)
    # de-nest
    keep: list[Path] = []
    for c in sorted(ep_dirs, key=lambda x: len(x.parts)):
        if any(str(c).startswith(str(k) + "/") for k in keep):
            continue
        keep.append(c)

    cards: list[dict[str, Any]] = []
    for i, ep in enumerate(keep):
        if limit is not None and i >= limit:
            break
        imgs = sorted(list(ep.glob("*.png")) + list(ep.glob("*.jpg")))[:4]
        state_files = [
            f
            for f in ep.rglob("*")
            if f.is_file()
            and any(
                h in f.name.lower()
                for h in (
                    "proprio",
                    "robot_state",
                    "joint",
                    "gripper",
                    "qpos",
                    "state",
                    "low_dim",
                    "force",
                )
            )
        ]
        state_arr = None
        state_summary: dict[str, Any] = {}
        if state_files:
            sf = state_files[0]
            try:
                if sf.suffix == ".npy":
                    state_arr = np.load(sf)
                elif sf.suffix == ".json":
                    state_summary = json.loads(sf.read_text())
                    if isinstance(state_summary, dict):
                        # flatten numeric values if possible
                        nums = [float(v) for v in state_summary.values() if isinstance(v, (int, float))]
                        if nums:
                            state_arr = np.asarray(nums, dtype=np.float32)
            except Exception:
                state_arr = None
        if state_arr is not None and not state_summary:
            state_summary = summarize_state_window(state_arr)

        # metadata sidecar search
        meta: dict[str, Any] = {"reward": 0}
        for mf in list(ep.glob("*.json")) + list(ep.glob("meta*.json")) + list(ep.glob("*label*")):
            try:
                data = json.loads(mf.read_text())
                if isinstance(data, dict):
                    meta.update(data)
            except Exception:
                continue
        stratum = label_stratum(meta, state_summary)
        binary = parse_binary_label(str(meta.get("failure_mode")), meta.get("reward"))
        sid = f"a8b_{ep.relative_to(root).as_posix().replace('/', '__')}"
        model_input = filter_allowed_fields(
            {
                "sample_id": sid,
                "images": [str(p) for p in imgs],
                "task_instruction": str(meta.get("task_instruction") or meta.get("task") or "failure verification"),
                "robot_state": state_arr.reshape(-1).astype(float).tolist()
                if isinstance(state_arr, np.ndarray)
                else None,
                "state_summary": state_summary,
                "track": "a8b",
                "split": "reflect",
            },
            mode="model_input",
        )
        assert_no_leakage(model_input)
        cards.append(
            {
                **model_input,
                "binary_label": binary,
                "failure_mode": meta.get("failure_mode"),
                "failure_reason": meta.get("failure_reason"),
                "reward": meta.get("reward"),
                "stratum": stratum,
                "episode_dir": str(ep),
                "n_state_files": len(state_files),
            }
        )
    return cards


def write_a8b_dataset(out_dir: Path | str, cards: list[dict[str, Any]], card_meta: dict[str, Any]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    frames: dict[str, np.ndarray] = {}
    for c in cards:
        sid = c["sample_id"]
        rgb_list = []
        for p in c.get("images") or []:
            pp = Path(p)
            if pp.exists():
                try:
                    rgb_list.append(_load_rgb(pp))
                except Exception:
                    continue
        if not rgb_list:
            rgb_list = [np.zeros((224, 224, 3), dtype=np.float32)]
        rgb = np.stack(rgb_list, axis=0)
        state = c.get("robot_state")
        if state is None:
            proprio = np.zeros((25, 8), dtype=np.float32)
        else:
            arr = np.asarray(state, dtype=np.float32).reshape(-1)
            # tile/truncate to (25, F)
            f = min(32, max(1, arr.size))
            vec = np.zeros((f,), dtype=np.float32)
            vec[: min(f, arr.size)] = arr[:f]
            proprio = np.tile(vec, (25, 1))
        depth = np.zeros(rgb.shape[:-1], dtype=np.float32)
        frames[f"{sid}__rgb"] = rgb
        frames[f"{sid}__depth"] = depth
        frames[f"{sid}__proprio"] = proprio
        label = c.get("binary_label", "unknown")
        rec = {
            "sample_id": sid,
            "snapshot": {
                "operator_name": "a8b_failure_verify",
                "appearance_class": str(c.get("failure_mode") or c.get("stratum") or "unknown"),
                "t": 0.0,
                "pose_xy": [0.0, 0.0],
                "heading": 0.0,
                "prior_outputs": [],
                "privileged_theta": {},
                "monitor_channel": "a8b",
                "rgb_shape": list(rgb.shape),
                "depth_shape": list(depth.shape),
                "proprio_shape": list(proprio.shape),
            },
            "ground_truth": {
                "category": label,
                "ab_class": "A",
                "theta": {},
                "feasible": ["Hold_and_Request"],
                "canonical_primitive": "Hold_and_Request",
                "is_sudden_trap": False,
            },
            "annotation": {
                "thought": f"Cross-modal failure attribution on external episode ({c.get('stratum')}).",
                "attribution": label,
                "attribution_raw": label,
                "action": {"primitive": "Hold_and_Request", "params": {"reason": "execution_failure"}},
            },
            "verdict": {"keep": True, "reason": "a8b"},
            "target_theta": [0.0, 0.0, 0.0, 0.0],
            "ambiguity_pair": None,
            "a8": {
                "track": "a8b",
                "stratum": c.get("stratum"),
                "task_instruction": c.get("task_instruction"),
                "binary_label": label,
                "state_summary": c.get("state_summary"),
            },
            "stratum": c.get("stratum"),
            "failure_mode": c.get("failure_mode"),
            "reward": c.get("reward"),
        }
        records.append(rec)
    (out / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (out / "cards.jsonl").write_text("\n".join(json.dumps(c) for c in cards) + "\n")
    np.savez_compressed(out / "frames.npz", **frames)
    # strata histogram
    from collections import Counter

    hist = Counter(str(c.get("stratum")) for c in cards)
    meta = {**card_meta, "n_records": len(records), "stratum_hist": dict(hist), "track": "a8b"}
    (out / "dataset_card.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out
