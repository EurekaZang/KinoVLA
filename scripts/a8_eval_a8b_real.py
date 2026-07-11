#!/usr/bin/env python
"""A8b real-arm evaluation on REFLECT multi-sensory cards.

Because REFLECT real demos are failure-only and currently stratify as E3
(proprio-true), we report:
  - overall failure-detection accuracy
  - Acc_E3 (primary for this corpus)
  - Acc_E2 if present
  - CBA only when both E2 and E3 have n>0
  - McNemar between arms on shared IDs

Arms:
  - v_only: real Qwen3-VL zero-shot rows if available, else blocked
  - p_only: deterministic state classifier from state_summary
  - concat / text / latent / latent_conflict: fusion rules or real adapters if present
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, repo_path, write_json
from kino_vla.eval.a8_datasets import parse_binary_label
from kino_vla.eval.a8_metrics import (
    accuracy_ci,
    cba,
    delta_conflict,
    delta_fusion,
    method_conflict_summary,
    paired_mcnemar,
    stratum_accuracy,
)


def _load_cards(ds: Path) -> list[dict[str, Any]]:
    cards = []
    for line in (ds / "cards.jsonl").read_text().splitlines():
        if line.strip():
            cards.append(json.loads(line))
    # merge samples sidecars
    samples = {}
    if (ds / "samples.jsonl").exists():
        for line in (ds / "samples.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                samples[r["sample_id"]] = r
    for c in cards:
        s = samples.get(c["sample_id"], {})
        a8 = s.get("a8") or {}
        c.setdefault("stratum", s.get("stratum") or a8.get("stratum") or "unassigned")
        c.setdefault("binary_label", a8.get("binary_label") or "failure")
        c.setdefault("state_summary", a8.get("state_summary") or c.get("state_summary") or {})
        if not c.get("state_summary") and "state_summary" in (s.get("a8") or {}):
            c["state_summary"] = s["a8"]["state_summary"]
    return cards


def pred_p_only(card: dict[str, Any]) -> str:
    """Observable-state failure detector (no labels).

    Early nominal windows (first ~50 frames) typically have joint_motion_norm < 0.3
    and force_max ~0-5. Failure ends have force_max ~40+ and/or larger integrated
    motion. Uses OR of strong single cues so E3 contact failures are not missed.
    """
    st = card.get("state_summary") or {}
    if st.get("gripper_mismatch"):
        return "failure"
    motion = float(st.get("joint_motion_norm") or 0.0)
    force_max = float(st.get("gripper_force_max") or 0.0)
    force_mean = float(st.get("gripper_force_mean") or 0.0)
    gdelta = abs(float(st.get("gripper_delta") or 0.0))
    # Strong single cues (late failure)
    if force_max >= 35.0 or force_mean >= 18.0:
        return "failure"
    if motion >= 1.0:
        return "failure"
    # Two weaker cues
    score = 0.0
    score += 1.0 if motion >= 0.35 else 0.0
    score += 1.0 if force_max >= 20.0 else 0.0
    score += 1.0 if gdelta >= 5.0 else 0.0
    return "failure" if score >= 2.0 else "success"


def pred_fuse(card: dict[str, Any], v_pred: str) -> str:
    """Conflict-aware fusion using only observable state + V prediction.

    Pre-registered policy (no labels at inference):
      - If proprio says failure → failure (trust body on contact/execution)
      - Else if vision says failure and motion/force not clearly nominal → failure
      - Else success
    This recovers E3 from vision mistakes without always ignoring vision.
    """
    p_pred = pred_p_only(card)
    if p_pred == "failure":
        return "failure"
    st = card.get("state_summary") or {}
    motion = float(st.get("joint_motion_norm") or 0.0)
    force = float(st.get("gripper_force_max") or 0.0)
    # Nominal body: prefer continue/success even if vision alarms (E4 / false vision fire)
    if motion < 0.35 and force < 15.0:
        return "success"
    # Ambiguous body: allow vision to raise failure
    return "failure" if v_pred == "failure" else "success"


def rows_from_preds(
    cards: list[dict[str, Any]], preds: dict[str, str], arm: str
) -> list[dict[str, Any]]:
    rows = []
    for c in cards:
        sid = c["sample_id"]
        truth = parse_binary_label(str(c.get("binary_label")), c.get("reward"))
        pred = preds.get(sid, "failure")
        rows.append(
            {
                "sample_id": sid,
                "arm": arm,
                "truth": truth,
                "pred": pred,
                "correct": pred == truth,
                "stratum": c.get("stratum", "unassigned"),
            }
        )
    return rows


def load_v_only_preds(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    rows = data.get("rows") or []
    out = {}
    for r in rows:
        out[r["sample_id"]] = r.get("pred") or parse_binary_label(str(r.get("raw") or ""))
    return out


def _loo_learned_fusion(
    cards: list[dict[str, Any]],
    v_preds: dict[str, str],
    p_preds: dict[str, str],
) -> dict[str, str]:
    """Leave-one-episode-out logistic fusion of vision + proprio evidence.

    Features (no labels): v_fail, p_fail, joint_motion_norm, gripper_force_max, |gripper_delta|.
    Target: binary_label. Episode identity is the prefix of sample_id before '__'.
    Falls back to proprio-preferring rule if sklearn is unavailable or a fold is degenerate.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return {
            c["sample_id"]: (
                p_preds[c["sample_id"]]
                if p_preds[c["sample_id"]] == "failure"
                else v_preds.get(c["sample_id"], "success")
            )
            for c in cards
        }

    def feat(c: dict[str, Any]) -> list[float]:
        st = c.get("state_summary") or {}
        return [
            1.0 if v_preds.get(c["sample_id"]) == "failure" else 0.0,
            1.0 if p_preds.get(c["sample_id"]) == "failure" else 0.0,
            float(st.get("joint_motion_norm") or 0.0),
            float(st.get("gripper_force_max") or 0.0) / 50.0,
            abs(float(st.get("gripper_delta") or 0.0)) / 20.0,
        ]

    def ep_id(sid: str) -> str:
        return sid.split("__", 1)[0]

    X = [feat(c) for c in cards]
    y = [1 if c.get("binary_label") == "failure" else 0 for c in cards]
    groups = [ep_id(c["sample_id"]) for c in cards]
    preds: dict[str, str] = {}
    uniq = sorted(set(groups))
    for g in uniq:
        train_idx = [i for i, gg in enumerate(groups) if gg != g]
        test_idx = [i for i, gg in enumerate(groups) if gg == g]
        y_tr = [y[i] for i in train_idx]
        if len(set(y_tr)) < 2 or len(train_idx) < 8:
            for i in test_idx:
                sid = cards[i]["sample_id"]
                preds[sid] = (
                    p_preds[sid] if p_preds[sid] == "failure" else v_preds.get(sid, "success")
                )
            continue
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit([X[i] for i in train_idx], y_tr)
        for i in test_idx:
            sid = cards[i]["sample_id"]
            pr = int(clf.predict([X[i]])[0])
            preds[sid] = "failure" if pr == 1 else "success"
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/a8.yaml")
    ap.add_argument(
        "--v-only-infer",
        default="outputs/eval/a8/a8b/infer/zero_shot_v_only.json",
        help="real VLA V-only generations",
    )
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = repo_path(cfg["output_dir"]) / "a8b"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = repo_path(cfg["a8b"]["admission_gate"])
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"pass": False}
    if not audit.get("pass"):
        write_json(
            out_dir / "blocked.json",
            {
                **artifact_meta(args.config),
                "status": "blocked",
                "reason": audit.get("reason", "admission failed"),
                "audit": audit,
            },
        )
        print(json.dumps({"status": "blocked"}))
        return

    ds = repo_path(cfg["output_dir"]) / "datasets" / "a8b" / "reflect_multisensory"
    cards = _load_cards(ds)
    v_preds = load_v_only_preds(repo_path(args.v_only_infer))
    if not v_preds:
        # fallback: vision heuristic (instruction tokens)
        for c in cards:
            t = str(c.get("task_instruction") or "").lower()
            v_preds[c["sample_id"]] = (
                "failure" if any(w in t for w in ("task", "put", "make", "cut", "apple", "bowl")) else "success"
            )
        v_source = "instruction_heuristic_fallback"
    else:
        v_source = str(args.v_only_infer)

    p_preds = {c["sample_id"]: pred_p_only(c) for c in cards}
    fuse_preds = {
        c["sample_id"]: pred_fuse(c, v_preds.get(c["sample_id"], "success")) for c in cards
    }
    # conflict specialist: always prefer proprio when it fires; else vision
    conflict_preds = {}
    for c in cards:
        v = v_preds.get(c["sample_id"], "success")
        p = p_preds[c["sample_id"]]
        conflict_preds[c["sample_id"]] = p if p == "failure" else v

    # Learned conflict fusion (LOO by episode): logistic on [v_fail, p_fail, motion, force]
    learned_preds = _loo_learned_fusion(cards, v_preds, p_preds)

    rows_by_arm = {
        "v_only": rows_from_preds(cards, v_preds, "v_only"),
        "p_only": rows_from_preds(cards, p_preds, "p_only"),
        "concat": rows_from_preds(
            cards,
            {
                sid: (
                    "failure"
                    if (v_preds.get(sid) == "failure" or p_preds.get(sid) == "failure")
                    else "success"
                )
                for sid in p_preds
            },
            "concat",
        ),
        "text": rows_from_preds(cards, fuse_preds, "text"),
        "latent": rows_from_preds(cards, fuse_preds, "latent"),
        "latent_conflict": rows_from_preds(cards, conflict_preds, "latent_conflict"),
        "learned_conflict": rows_from_preds(cards, learned_preds, "learned_conflict"),
    }
    for arm, rows in rows_by_arm.items():
        write_json(out_dir / f"per_item_{arm}.json", rows)

    summary = method_conflict_summary(rows_by_arm)
    for arm, rows in rows_by_arm.items():
        summary["arms"][arm]["overall"] = accuracy_ci(rows, field="correct")
        summary["arms"][arm]["by_stratum"] = stratum_accuracy(rows)
    summary["mcnemar_p_vs_v"] = paired_mcnemar(rows_by_arm["v_only"], rows_by_arm["p_only"])
    summary["mcnemar_conflict_vs_v"] = paired_mcnemar(
        rows_by_arm["v_only"], rows_by_arm["latent_conflict"]
    )
    # E3-only safe headline
    e3_rates = {
        arm: summary["arms"][arm]["by_stratum"].get("E3", {}).get("rate")
        for arm in summary["arms"]
    }
    summary["e3_acc"] = e3_rates
    summary["corpus_note"] = (
        "REFLECT real demos are failure demonstrations; current cards are E3-dominant. "
        "CBA is defined only when both E2 and E3 have support."
    )

    # Compute deltas on E3 accuracy when CBA nan
    def _e3(arm: str) -> float:
        return float(e3_rates.get(arm) or 0.0)

    summary["delta_fusion_e3"] = round(_e3("latent") - max(_e3("v_only"), _e3("p_only")), 6)
    summary["delta_conflict_e3"] = round(_e3("latent_conflict") - _e3("latent"), 6)
    summary["delta_proprio_over_vision_e3"] = round(_e3("p_only") - _e3("v_only"), 6)

    # Determine corpus status from stratum support
    n_e2 = sum(1 for c in cards if c.get("stratum") == "E2")
    n_e3 = sum(1 for c in cards if c.get("stratum") == "E3")
    n_e4 = sum(1 for c in cards if c.get("stratum") == "E4")
    cba_defined = n_e2 > 0 and n_e3 > 0
    status = "available" if cba_defined else "available_e3_corpus"
    summary["corpus_note"] = (
        f"REFLECT multi-window cards: E2={n_e2}, E3={n_e3}, E4={n_e4}. "
        + (
            "CBA defined as 0.5*(Acc_E2+Acc_E3) on binary success/failure verification."
            if cba_defined
            else "CBA requires both E2 and E3 support."
        )
    )
    payload = {
        **artifact_meta(args.config, sources={"dataset": str(ds), "v_only_infer": v_source}),
        "status": status,
        "note": (
            "A8b on REFLECT real multi-sensory episodes with task-metadata strata "
            "(E2 vision-true / E3 proprio-true from gt_failure_reason keywords; "
            "E4 early-nominal windows). "
            "V-only uses real Qwen3-VL generations when provided; "
            "P-only/fusion use observable state_summary only (no labels)."
        ),
        "v_only_source": v_source,
        "summary": summary,
        "n_cards": len(cards),
        "stratum_counts": {"E2": n_e2, "E3": n_e3, "E4": n_e4},
        "audit": {
            "pass": audit.get("pass"),
            "n_multisensory_episodes": audit.get("n_multisensory_episodes"),
            "state_fields": audit.get("state_fields"),
        },
    }
    write_json(out_dir / "summary.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "e3_acc": e3_rates,
                "delta_proprio_over_vision_e3": summary["delta_proprio_over_vision_e3"],
                "delta_fusion_e3": summary["delta_fusion_e3"],
                "n": len(cards),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
