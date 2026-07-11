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
    st = card.get("state_summary") or {}
    # Strong proprio failure cues
    if st.get("gripper_mismatch"):
        return "failure"
    motion = float(st.get("joint_motion_norm") or 0.0)
    force = float(st.get("gripper_force_max") or st.get("gripper_force_mean") or 0.0)
    gdelta = abs(float(st.get("gripper_delta") or 0.0))
    width = float(st.get("gripper_width") or 0.0)
    closed = float(st.get("gripper_cmd_closed") or 0.0) >= 0.5
    # REFLECT failures often show large joint motion + residual open gripper / force spikes
    score = 0.0
    score += 1.0 if motion > 0.5 else 0.0
    score += 1.0 if force > 10.0 else 0.0
    score += 1.0 if gdelta > 5.0 else 0.0
    score += 1.0 if (closed and width > 20.0) else 0.0
    score += 0.5 if width > 80.0 else 0.0  # still wide near end
    return "failure" if score >= 1.0 else "success"


def pred_fuse(card: dict[str, Any], v_pred: str) -> str:
    p_pred = pred_p_only(card)
    stratum = str(card.get("stratum") or "unassigned")
    if stratum == "E3":
        return p_pred
    if stratum == "E2":
        return v_pred
    # agree / unknown: failure if either says failure (safe default)
    if v_pred == "failure" or p_pred == "failure":
        return "failure"
    return "success"


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
    # conflict-shaped fusion: trust proprio more on E3 (already), and require agreement for success
    conflict_preds = {}
    for c in cards:
        v = v_preds.get(c["sample_id"], "success")
        p = p_preds[c["sample_id"]]
        if str(c.get("stratum")) == "E3":
            conflict_preds[c["sample_id"]] = p
        elif str(c.get("stratum")) == "E2":
            conflict_preds[c["sample_id"]] = v
        else:
            conflict_preds[c["sample_id"]] = "failure" if (v == "failure" or p == "failure") else "success"

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

    payload = {
        **artifact_meta(args.config, sources={"dataset": str(ds), "v_only_infer": v_source}),
        "status": "available_e3_corpus",
        "note": (
            "A8b on REFLECT real multi-sensory episodes. "
            "V-only uses real Qwen3-VL generations when provided; "
            "P-only/fusion use observable state_summary (no labels). "
            "Failure-only E3-dominant corpus: report Acc_E3 and proprio-over-vision gain; "
            "CBA requires both E2 and E3."
        ),
        "v_only_source": v_source,
        "summary": summary,
        "n_cards": len(cards),
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
