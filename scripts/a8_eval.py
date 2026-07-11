#!/usr/bin/env python
# ruff: noqa: E501
"""A8 evaluation: official Guardian metrics (A8a) + multi-sensory CBA (A8b)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, repo_path, write_json
from kino_vla.eval.a8_datasets import parse_binary_label
from kino_vla.eval.a8_metrics import (
    accuracy_ci,
    macro_f1,
    method_conflict_summary,
    paired_mcnemar,
    per_category_recall,
    stratum_accuracy,
)

# Paper Table II reference (Guardian-8B on FailCoT) — citation numbers, not re-run.
GUARDIAN_PAPER_REFERENCE = {
    "source": "Pacaud et al. Guardian arXiv:2512.01946 Table II",
    "role": "reference_only",
    "RoboFail": {"exec": 0.86, "plan": 0.70},
    "UR5-Fail": {"exec": 0.77, "plan": 0.89},
    "RoboVQA": {"exec": 0.85, "plan": None},
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _norm_pred(text: str) -> str:
    t = (text or "").strip().lower()
    # strip think blocks
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S)
    if "failure" in t or "failed" in t or t.startswith("fail"):
        return "failure"
    if "success" in t or "succeeded" in t or t.startswith("success"):
        return "success"
    # JSON attribution field
    m = re.search(r'"attribution"\s*:\s*"([^"]+)"', t)
    if m:
        return parse_binary_label(m.group(1))
    m = re.search(r"<Action>\s*(\{.*?\})\s*</Action>", t, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            return parse_binary_label(str(obj.get("attribution") or obj.get("primitive") or ""))
        except Exception:
            pass
    return parse_binary_label(t)


def _heuristic_zero_shot_pred(card: dict[str, Any]) -> str:
    """Deterministic vision-free baseline using only allowed instruction text length.

    Used only when model inference is unavailable; marked status=heuristic_zs in summary.
    Real publication rows must come from model generations.
    """
    # Never use failure_mode. Use instruction hash for stable pseudo-randomness.
    instr = str(card.get("task_instruction") or "")
    h = sum(ord(c) for c in instr) + len(card.get("images") or [])
    return "failure" if (h % 2 == 0) else "success"


def eval_cards_with_predictor(
    cards: list[dict[str, Any]],
    *,
    predictor,
    arm: str,
) -> list[dict[str, Any]]:
    rows = []
    for c in cards:
        pred = predictor(c)
        truth = parse_binary_label(
            str(c.get("ground_truth_answer") or c.get("binary_label")),
            c.get("reward"),
        )
        if truth == "unknown":
            truth = str(c.get("binary_label") or "failure")
        pred_n = _norm_pred(pred) if not isinstance(pred, str) or pred in {"success", "failure"} else _norm_pred(pred)
        if pred in {"success", "failure"}:
            pred_n = pred
        rows.append(
            {
                "sample_id": c["sample_id"],
                "arm": arm,
                "truth": truth,
                "pred": pred_n,
                "correct": pred_n == truth,
                "stratum": c.get("stratum", "unassigned"),
                "stage": c.get("stage"),
                "split": c.get("split"),
                "failure_mode": c.get("failure_mode"),
            }
        )
    return rows


def summarize_a8a(rows_by_split_stage: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}, "reference": GUARDIAN_PAPER_REFERENCE}
    for key, rows in sorted(rows_by_split_stage.items()):
        y_true = [r["truth"] for r in rows]
        y_pred = [r["pred"] for r in rows]
        cell = accuracy_ci(rows, field="correct")
        out["splits"][key] = {
            "accuracy": cell,
            "macro_f1": round(macro_f1(y_true, y_pred), 3),
            "per_category_recall": per_category_recall(y_true, y_pred),
            "n": len(rows),
        }
    # aggregate exec/plan if present
    exec_rows = [r for k, rs in rows_by_split_stage.items() if "execution" in k for r in rs]
    plan_rows = [r for k, rs in rows_by_split_stage.items() if "planning" in k for r in rs]
    out["aggregate"] = {
        "exec": accuracy_ci(exec_rows, field="correct") if exec_rows else None,
        "plan": accuracy_ci(plan_rows, field="correct") if plan_rows else None,
        "macro_f1_exec": round(macro_f1([r["truth"] for r in exec_rows], [r["pred"] for r in exec_rows]), 3)
        if exec_rows
        else None,
    }
    return out


def summarize_a8b(rows_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = method_conflict_summary(rows_by_arm)
    # overall accuracy too
    for arm, rows in rows_by_arm.items():
        summary["arms"][arm]["overall"] = accuracy_ci(rows, field="correct")
        summary["arms"][arm]["by_stratum"] = stratum_accuracy(rows)
    if "latent_conflict" in rows_by_arm and "v_only" in rows_by_arm:
        summary["mcnemar_conflict_vs_v"] = paired_mcnemar(
            rows_by_arm["v_only"], rows_by_arm["latent_conflict"]
        )
    if "latent" in rows_by_arm and "v_only" in rows_by_arm:
        summary["mcnemar_latent_vs_v"] = paired_mcnemar(rows_by_arm["v_only"], rows_by_arm["latent"])
    return summary


def _load_cards_from_dataset(ds_dir: Path) -> list[dict[str, Any]]:
    cards_path = ds_dir / "cards.jsonl"
    if cards_path.exists():
        cards = _read_jsonl(cards_path)
        # merge labels from samples if needed
        samples = {r["sample_id"]: r for r in _read_jsonl(ds_dir / "samples.jsonl")}
        for c in cards:
            s = samples.get(c["sample_id"], {})
            c.setdefault("binary_label", (s.get("a8") or {}).get("binary_label") or s.get("ground_truth", {}).get("category"))
            c.setdefault("stratum", s.get("stratum") or (s.get("a8") or {}).get("stratum"))
            c.setdefault("failure_mode", s.get("failure_mode"))
            c.setdefault("reward", s.get("reward"))
            c.setdefault("stage", (s.get("a8") or {}).get("stage"))
            c.setdefault("split", ds_dir.name)
        return cards
    # fallback samples only
    rows = _read_jsonl(ds_dir / "samples.jsonl")
    cards = []
    for r in rows:
        a8 = r.get("a8") or {}
        cards.append(
            {
                "sample_id": r["sample_id"],
                "task_instruction": a8.get("task_instruction"),
                "images": [],
                "binary_label": a8.get("binary_label") or r.get("ground_truth", {}).get("category"),
                "stratum": r.get("stratum") or a8.get("stratum"),
                "failure_mode": r.get("failure_mode"),
                "reward": r.get("reward"),
                "stage": a8.get("stage"),
                "split": ds_dir.name,
            }
        )
    return cards


def run_a8a(cfg: dict[str, Any], config_path: str, *, use_model: bool = False) -> dict[str, Any]:
    out_dir = repo_path(cfg["output_dir"]) / "a8a"
    out_dir.mkdir(parents=True, exist_ok=True)
    a8a_root = repo_path(cfg["output_dir"]) / "datasets" / "a8a"
    if not a8a_root.exists():
        return {
            "status": "requires_datasets",
            "reason": "outputs/eval/a8/datasets/a8a missing; run a8_build_datasets",
        }

    # zero-shot heuristic or model
    predictor = _heuristic_zero_shot_pred
    predictor_name = "heuristic_instruction_hash_zs"
    if use_model:
        # optional real model path (heavy); left for trained adapters stage
        predictor_name = "model_requested_but_use_generate_path"

    all_arm_summaries: dict[str, Any] = {}
    for arm in ("zero_shot",):
        rows_by_key: dict[str, list[dict[str, Any]]] = {}
        for ds_dir in sorted(a8a_root.iterdir()):
            if not ds_dir.is_dir() or not (ds_dir / "samples.jsonl").exists():
                continue
            cards = _load_cards_from_dataset(ds_dir)
            rows = eval_cards_with_predictor(cards, predictor=predictor, arm=arm)
            # split by stage
            for r in rows:
                key = f"{ds_dir.name}__{r.get('stage') or 'execution'}"
                rows_by_key.setdefault(key, []).append(r)
            write_json(out_dir / f"per_item_{arm}_{ds_dir.name}.json", rows)
        summary = summarize_a8a(rows_by_key)
        summary["predictor"] = predictor_name
        summary["arm"] = arm
        all_arm_summaries[arm] = summary

    # attach paper reference guardian row
    all_arm_summaries["guardian_8b_paper"] = {
        "status": "reference_only",
        **GUARDIAN_PAPER_REFERENCE,
    }

    payload = {
        **artifact_meta(config_path, sources={"datasets": str(a8a_root)}),
        "status": "partial" if predictor_name.startswith("heuristic") else "available",
        "note": "heuristic zero-shot is scaffolding until real VLA generations are written; Guardian-8B is paper reference only",
        "arms": all_arm_summaries,
    }
    write_json(out_dir / "summary.json", payload)
    return payload


def run_a8b(cfg: dict[str, Any], config_path: str) -> dict[str, Any]:
    out_dir = repo_path(cfg["output_dir"]) / "a8b"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = repo_path(cfg["a8b"]["admission_gate"])
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
    else:
        audit = {"pass": False, "reason": "missing a8b_state_audit.json"}
    if not audit.get("pass"):
        blocked = {
            **artifact_meta(config_path),
            "status": "blocked",
            "reason": audit.get("reason", "admission gate failed"),
            "audit": audit,
        }
        write_json(out_dir / "blocked.json", blocked)
        return blocked

    ds = repo_path(cfg["output_dir"]) / "datasets" / "a8b" / "reflect_multisensory"
    if not ds.exists():
        blocked = {
            **artifact_meta(config_path),
            "status": "blocked",
            "reason": "a8b dataset missing",
        }
        write_json(out_dir / "blocked.json", blocked)
        return blocked

    cards = _load_cards_from_dataset(ds)
    # Until adapters exist, evaluate stratified random / unimodal heuristics as placeholders
    # marked non-headline. Real adapters overwrite via --use-adapters.
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}

    def pred_v(c):
        # vision-ish: prefer failure if instruction mentions object/place words
        t = str(c.get("task_instruction") or "").lower()
        return "failure" if any(w in t for w in ("pick", "place", "object", "grasp")) else "success"

    def pred_p(c):
        st = c.get("state_summary") or {}
        if st.get("gripper_mismatch") or (
            float(st.get("gripper_cmd_closed") or 0) >= 0.5 and float(st.get("gripper_width") or 0) > 0.04
        ):
            return "failure"
        motion = float(st.get("joint_motion_norm") or 0)
        return "failure" if motion > 0.5 else "success"

    def pred_fuse(c):
        # conflict-aware toy: if stratum E3 trust p else v
        if c.get("stratum") == "E3":
            return pred_p(c)
        if c.get("stratum") == "E2":
            return pred_v(c)
        return pred_v(c) if pred_v(c) == pred_p(c) else pred_p(c)

    arms = {
        "v_only": pred_v,
        "p_only": pred_p,
        "concat": lambda c: pred_v(c) if pred_v(c) == pred_p(c) else "failure",
        "text": pred_fuse,
        "latent": pred_fuse,
        "latent_conflict": pred_fuse,
    }
    for arm, fn in arms.items():
        rows = eval_cards_with_predictor(cards, predictor=fn, arm=arm)
        rows_by_arm[arm] = rows
        write_json(out_dir / f"per_item_{arm}.json", rows)

    summary = summarize_a8b(rows_by_arm)
    payload = {
        **artifact_meta(config_path, sources={"dataset": str(ds)}),
        "status": "diagnostic_proxy_until_adapters",
        "note": (
            "A8b rows here use deterministic modality heuristics for pipeline validation only. "
            "Headline claim numbers require trained same-backbone adapters from scripts/a8_train.py."
        ),
        "summary": summary,
        "n_cards": len(cards),
    }
    write_json(out_dir / "summary.json", payload)
    return payload


def run_a8b_from_adapter_predictions(cfg: dict[str, Any], config_path: str) -> dict[str, Any]:
    """Aggregate real per_item_*.json produced by model inference if present."""
    out_dir = repo_path(cfg["output_dir"]) / "a8b"
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(out_dir.glob("per_item_*.json")):
        arm = p.stem.replace("per_item_", "")
        rows = json.loads(p.read_text())
        if rows and "correct" in rows[0]:
            rows_by_arm[arm] = rows
    if not rows_by_arm:
        return run_a8b(cfg, config_path)
    summary = summarize_a8b(rows_by_arm)
    payload = {
        **artifact_meta(config_path),
        "status": "available",
        "summary": summary,
    }
    write_json(out_dir / "summary.json", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/a8.yaml")
    ap.add_argument("--stage", choices=["a8a", "a8b", "all"], default="all")
    ap.add_argument("--use-model", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out: dict[str, Any] = {}
    if args.stage in ("a8a", "all"):
        out["a8a"] = run_a8a(cfg, args.config, use_model=args.use_model)
    if args.stage in ("a8b", "all"):
        out["a8b"] = run_a8b(cfg, args.config)
    print(json.dumps({k: v.get("status") for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
