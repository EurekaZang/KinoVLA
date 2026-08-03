#!/usr/bin/env python
"""A3.3 — build the B5-conflict-bi (bidirectional conflict) SFT dataset.

B5-conflict (A2) learned only the vision-true direction (T2 matched-O4: proprio says mud,
vision says adhesion => trust vision). The decisive A3 question is whether that is genuine
EVIDENCE WEIGHING or a "conflict => trust camera" vision-dominance shortcut. B5-conflict-bi
trains on BOTH conflict directions so the operation is taught per evidence-direction:

  - T2 vision-true  (from A2): matched-O4 => trust vision => adhesion
  - T3 proprio-true (NEW here): O7 looks_safe => trust proprio => low_friction

The reverse probe (O7_reverse: vision alarms, proprio nominal => continue) and the held-out O7
appearances are NEVER trained -- the A3.2 generalization test of whether the bidirectional
operation transfers to "trust proprio over a vision alarm" zero-shot.

Output: ``outputs/eval/a3/sft_conflict_bi`` = the A2 conflict-heldout records + the A3 T3
train-split O7 looks_safe records (mu in {0.09,0.15}, train appearances) with a trust-proprio CoT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

A2_DIR = "outputs/eval/a2/sft_conflict_heldout"
T3_DIR = "outputs/eval/a3/corpus_t3"
A03_DIR = "outputs/eval/a0/corpus"
# T3 proprio-true training sources: O7 looks_safe at clearly-slippery mu, on TRAIN appearances.
# (test appearances + the O7_reverse probe are held out -- the A3.2 generalization test.)
T3_TRAIN_MU = (0.09, 0.15)
T3_TRAIN_APPEARANCES = {"solid_ground", "concrete_speckle"}

# Scripted, appearance-agnostic trust-PROPRIO CoT (the T3 direction E2/A2 did not teach). Names
# the PHYSICS (slip), never the decal colour, so it cannot install a colour shortcut.
T3_TRUST_PROPRIO_COT = (
    "The surface looks like safe solid ground, but the proprioception shows a clear foot slip; "
    "when the vision looks benign yet the body slips, trust the proprioception over the "
    "appearance: this is low_friction. Slow to a constrained gait (Set_Constraint), do not cruise."
)
# O8 shares the SAME solid_ground appearance as O7 looks_safe, so a model keyed on the appearance
# would shortcut "solid_ground -> low_friction". Including O8 (solid_ground + a WALL signature) with
# the invisible_obstacle label FORCES proprio disambiguation: slip -> low_friction, wall ->
# invisible_obstacle. This is the anti-shortcut the bidirectional set must carry.
O8_TRUST_PROPRIO_COT = (
    "The surface looks like solid ground, but the proprioception shows a hard tracking deficit "
    "with NO foot slip and NO effort spike -- the feet keep grip yet the body makes no forward "
    "progress. Trust the proprioception: this is an unseen rigid barrier (invisible_obstacle). "
    "Do not push through; detour (Replan_Waypoint / Update_Topology)."
)


def _load(d: str) -> tuple[list[dict], dict[str, np.ndarray]]:
    lines = Path(d, "samples.jsonl").read_text().splitlines()
    recs = [json.loads(ln) for ln in lines if ln]
    npz = np.load(Path(d, "frames.npz"))
    return recs, {k: npz[k] for k in npz.files}


def main() -> None:
    ap = argparse.ArgumentParser(description="A3.3 B5-conflict-bi bidirectional conflict dataset")
    ap.add_argument("--a2", default=A2_DIR)
    ap.add_argument("--t3", default=T3_DIR)
    ap.add_argument("--a03", default=A03_DIR)
    ap.add_argument("--no-hindsight", action="store_true",
                    help="robustness trick: drop the hindsight base (keep matched + O7 + O8 only)")
    ap.add_argument("--out", default="outputs/eval/a3/sft_conflict_bi")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) the A2 conflict-heldout records (hindsight + matched-O4/O2 trust-VISION CoT) -- unchanged.
    a2_recs, a2_frames = _load(args.a2)
    if args.no_hindsight:  # ROBUSTNESS TRICK: drop the hindsight base's solid_ground→low_friction
        # prior (95 O7) so O7/O8 proprio discrimination is the ONLY way to succeed on T3 (no
        # majority shortcut). Keeps matched (T2) + O7/O8 (T3) = pure conflict (no hindsight prior).
        keep = [r for r in a2_recs if r["sample_id"].startswith("a0corpus_matched")]
        drop = len(a2_recs) - len(keep)
        a2_recs = keep
        print(f"[bidir] --no-hindsight: dropped {drop} hindsight records, kept {len(keep)} matched")

    # 2) the T3 proprio-true records: O7 looks_safe, train appearances, clearly-slippery mu.
    t3_recs_all, t3_frames = _load(args.t3)
    t3_recs: list[dict] = []
    for rec in t3_recs_all:
        if rec.get("a3_direction") != "looks_safe":
            continue
        if rec.get("a3_mu") not in T3_TRAIN_MU:
            continue
        if rec.get("appearance_id") not in T3_TRAIN_APPEARANCES:
            continue
        rec = json.loads(json.dumps(rec))  # deep copy
        cat = rec["ground_truth"]["category"]            # low_friction
        rec["annotation"] = {  # overwrite with the scripted trust-PROPRIO CoT
            "thought": T3_TRUST_PROPRIO_COT,
            "attribution": cat,
            "attribution_raw": cat,
            "action": {"primitive": "Set_Constraint",
                       "params": {"max_speed": 0.3, "stiffness": 0.5}},
        }
        rec["verdict"] = {"keep": True, "reason": "A3-conflict-bi-train", "detail": "trust-proprio"}
        t3_recs.append(rec)

    # 3) O8 invisible-collider records from the A0.3 corpus (TRAIN appearances of solid_ground).
    #    Same appearance as O7 looks_safe, OPPOSITE proprio (wall, no slip) -> invisible_obstacle.
    #    Forces the model to read proprio (not the solid_ground look) on T3 -- the anti-shortcut.
    a03_recs_all, a03_frames = _load(args.a03)
    o8_recs: list[dict] = []
    for rec in a03_recs_all:
        if rec["snapshot"]["operator_name"] != "O8_invisible_collider":
            continue
        if rec.get("appearance_id") not in T3_TRAIN_APPEARANCES:
            continue
        rec = json.loads(json.dumps(rec))
        rec["annotation"] = {
            "thought": O8_TRUST_PROPRIO_COT,
            "attribution": "invisible_obstacle",
            "attribution_raw": "invisible_obstacle",
            "action": {"primitive": "Update_Topology",
                       "params": {"region_xy": [0.0, 0.0], "radius_m": 0.6, "status": "blocked"}},
        }
        rec["verdict"] = {"keep": True, "reason": "A3-conflict-bi-train",
                          "detail": "O8 trust-proprio"}
        rec["_orig_sid"] = rec["sample_id"]      # remember for frame lookup after replication
        o8_recs.append(rec)

    # 4b) OVERSAMPLE O8 to balance the hindsight base's O7 (95 O7 vs 0 O8 in the base ⇒ the model
    #    defaults to solid_ground→low_friction). Replicate the 24 train-O8 records O8_REPS× so the
    #    wall signature is not swamped. Frames are shared (copied under each replicate sid).
    O8_REPS = 2  # balance O8 vs O7 in the conflict stratum (robust proprio across seeds)
    o8_expanded: list[dict] = []
    for rec in o8_recs:
        orig = rec["_orig_sid"]
        for r in range(O8_REPS):
            rc = json.loads(json.dumps(rec))
            rc["sample_id"] = orig if r == 0 else f"{orig}_r{r}"
            o8_expanded.append(rc)

    # 4) merge (disjoint sample_ids; disjoint frame keys).
    all_recs = list(a2_recs) + t3_recs + o8_expanded
    all_frames: dict[str, np.ndarray] = dict(a2_frames)
    shipped = {r["sample_id"] for r in all_recs}
    for src in (t3_frames, a03_frames):
        for k, v in src.items():
            sid, mod = k.split("__", 1)
            if sid in shipped and mod in ("rgb", "depth", "proprio"):
                all_frames[k] = v
    # copy frames for the O8 replicates (they share the original's modality arrays)
    for rec in o8_expanded:
        orig = rec["_orig_sid"]
        if orig == rec["sample_id"]:
            continue
        for mod in ("rgb", "depth", "proprio"):
            srck = f"{orig}__{mod}"
            if srck in all_frames:
                all_frames[f"{rec['sample_id']}__{mod}"] = all_frames[srck]
    for rec in all_recs:      # drop the bookkeeping key before writing
        rec.pop("_orig_sid", None)
    # keep only frames for shipped records (drop hindsight frames when --no-hindsight)
    shipped_final = {r["sample_id"] for r in all_recs}
    all_frames = {
        k: v for k, v in all_frames.items()
        if k.split("__", 1)[0] in shipped_final
        and k.split("__", 1)[-1] in ("rgb", "depth", "proprio")
    }

    (out / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in all_recs) + "\n")
    np.savez(out / "frames.npz", **all_frames)

    payload = "\n".join(json.dumps(r, sort_keys=True) for r in all_recs)
    card = {
        "name": "a3_conflict_bi",
        "spec": "experiments_design.md §4 A3.3 (bidirectional: trust-vision + trust-proprio)",
        "n_records": len(all_recs),
        "n_a2_conflict_heldout": len(a2_recs),
        "n_t3_trust_proprio": len(t3_recs),
        "n_o8_trust_proprio": len(o8_expanded),
        "n_o8_unique": len(o8_recs),
        "o8_oversample_reps": O8_REPS,
        "anti_shortcut": "O8 (solid_ground + wall -> invisible_obstacle) shares O7's look, "
                          "forcing proprio use (slip -> low_friction vs wall -> invisible)",
        "t3_train_mu": list(T3_TRAIN_MU),
        "t3_train_appearances": sorted(T3_TRAIN_APPEARANCES),
        "held_out": ["O7_reverse (the A3.2 probe)", "O7 test appearances"],
        "frames_keys": len(all_frames),
        "records_sha256": hashlib.sha256(payload.encode()).hexdigest()[:12],
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    print(f"\n[OK] wrote {len(all_recs)} records + {len(all_frames)} frame arrays -> {out}")


if __name__ == "__main__":
    main()
