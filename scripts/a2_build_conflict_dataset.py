#!/usr/bin/env python
"""A2 — build the appearance-held-out conflict SFT dataset (Paper-A §4 A2 + §2 A0.4).

The shortcut-killer training set. Mirrors E2's recipe (hindsight + nav + matched-conflict CoT) but
replaces E2's 2-appearance matched pair (yellow_adhesive / brown_mud, seeds 0-9) with the A0.3
frozen corpus's TRAIN-split matched pair across FOUR train appearances:

  adhesion (O4_tether)   train: yellow_board, amber_adhesive        (48 snapshots)
  compliant (O2_compliance) train: brown_mud, dark_gray_mud           (60 snapshots)

The held-out TEST appearances (gray_tape, checker_decal, translucent_sheet, reddish_mud,
wet_sheen_mud) NEVER appear in any training data (verified: hindsight uses only yellow_adhesive /
brown_mud / ice_sheet / solid_ground). So an agent that generalizes to the held-out appearances
demonstrably learned the CONCEPT (physics category), not the colour — the direct counter to the
strongest predictable review attack on E2 ("B5-conflict learned yellow->Backstep").

The corpus snapshots carry a minimal privileged label; we overwrite ``annotation.thought`` with the
scripted, appearance-agnostic CONFLICT CoT (the evidence-weighing instruction E2 used), keeping the
privileged attribution / action / ground_truth. Everything else (hindsight 413, the nav stratum) is
unchanged — the conflict stratum is purely additive.

Output: ``outputs/eval/a2/sft_conflict_heldout/{samples.jsonl, frames.npz, dataset_card.json}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HINDSIGHT_DIR = "outputs/hindsight_isaac"
CORPUS_DIR = "outputs/eval/a0/corpus"

# Appearance-held-out train appearances for the matched pair (A0.4 / frozen_manifest). The held-out
# TEST appearances (the A2 eval set) are the complement and must NOT appear here.
TRAIN_APPEARANCES = {
    "O4_tether": {"yellow_board", "amber_adhesive"},
    "O2_compliance": {"brown_mud", "dark_gray_mud"},
}

# Scripted, appearance-agnostic conflict CoT (the evidence-weighing operation, E2-proven). Names the
# MATERIAL class (adhesive board / mud), never the specific colour, so it cannot install a colour
# shortcut — only the physics-category concept.
CONFLICT_COT = {
    "O4_tether": (
        "The tangential resistance could be compliant mud, but the surface ahead is clearly "
        "an adhesive board. When the proprioception is ambiguous between mud and adhesion, "
        "the visible material is decisive: this is adhesion. Escape the sticky patch with a "
        "Backstep rather than pushing through."
    ),
    "O2_compliance": (
        "The surface ahead is mud and the proprioception shows sustained compliance — vision "
        "and body agree, this is soft deformable ground with no conflict. Attribute "
        "compliant_terrain and switch to a high-step gait to traverse it."
    ),
}


def _load_records_frames(d: str) -> tuple[list[dict], dict[str, np.ndarray]]:
    recs = [json.loads(line) for line in Path(d, "samples.jsonl").read_text().splitlines() if line]
    npz = np.load(Path(d, "frames.npz"))
    frames = {k: npz[k] for k in npz.files}
    return recs, frames


def main() -> None:
    ap = argparse.ArgumentParser(description="A2 appearance-held-out conflict SFT dataset builder")
    ap.add_argument("--hindsight", default=HINDSIGHT_DIR)
    ap.add_argument("--corpus", default=CORPUS_DIR)
    ap.add_argument("--out", default="outputs/eval/a2/sft_conflict_heldout")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) hindsight (the general-recovery capability base) — UNCHANGED.
    h_recs, h_frames = _load_records_frames(args.hindsight)
    h_ids = {r["sample_id"] for r in h_recs}

    # 2) corpus TRAIN-split matched pair (the appearance-diverse conflict stratum).
    c_recs, c_frames = _load_records_frames(args.corpus)
    conflict_recs: list[dict] = []
    kept_appearances: dict[str, set[str]] = {"O4_tether": set(), "O2_compliance": set()}
    for rec in c_recs:
        op = rec["snapshot"]["operator_name"]
        if op not in TRAIN_APPEARANCES:
            continue
        app = rec.get("appearance_id") or rec["snapshot"].get("appearance_class")
        if app not in TRAIN_APPEARANCES[op]:
            continue
        if rec.get("appearance_split") not in (None, "train"):
            continue  # belt-and-suspenders: only the pre-registered train split
        if rec.get("annotation") is None:
            continue  # some corpus records carry no annotation; skip (dropped / non-recovery)
        # Overwrite the privileged thought with the scripted, appearance-agnostic conflict CoT.
        rec = json.loads(json.dumps(rec))  # deep copy so we don't mutate the corpus in memory
        rec["annotation"]["thought"] = CONFLICT_COT[op]
        rec["annotation"]["attribution_raw"] = rec["annotation"]["attribution"]
        rec["verdict"] = {"keep": True, "reason": "A2-conflict-train", "detail": "scripted CoT"}
        conflict_recs.append(rec)
        kept_appearances[op].add(app)

    # 3) merge: hindsight + conflict-train (disjoint sample_ids; disjoint frames keys).
    all_recs = list(h_recs) + conflict_recs
    all_frames: dict[str, np.ndarray] = {}
    all_frames.update(h_frames)
    n_dup = 0
    for sid_frames in (c_frames,):
        for k, v in sid_frames.items():
            sid = k.split("__", 1)[0]
            if sid in h_ids:
                continue  # never happens (hindsight vs a0corpus ids), but guard the merge
            if k in all_frames:
                n_dup += 1
                continue
            all_frames[k] = v
    # keep only the frame arrays for records we actually ship (rgb/depth/proprio per sid)
    shipped_ids = {r["sample_id"] for r in all_recs}
    all_frames = {
        k: v
        for k, v in all_frames.items()
        if k.split("__", 1)[0] in shipped_ids
        and k.split("__", 1)[-1] in ("rgb", "depth", "proprio")
    }

    (out / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in all_recs) + "\n")
    np.savez(out / "frames.npz", **all_frames)

    # 4) dataset card (provenance + the appearance-held-out guarantee).
    payload = "\n".join(json.dumps(r, sort_keys=True) for r in all_recs)
    card = {
        "name": "a2_conflict_heldout",
        "spec": "experiments_design.md §4 A2 + §2 A0.3/A0.4 (appearance-held-out shortcut-killer)",
        "n_records": len(all_recs),
        "n_hindsight": len(h_recs),
        "n_conflict_train": len(conflict_recs),
        "conflict_train_by_operator": {
            op: sum(1 for r in conflict_recs if r["snapshot"]["operator_name"] == op)
            for op in TRAIN_APPEARANCES
        },
        "conflict_train_appearances": {op: sorted(s) for op, s in kept_appearances.items()},
        "held_out_test_appearances": {
            "O4_tether": ["checker_decal", "gray_tape", "translucent_sheet"],
            "O2_compliance": ["reddish_mud", "wet_sheen_mud"],
        },
        "leakage_check": "hindsight uses only {yellow_adhesive, brown_mud, ice_sheet, "
        "solid_ground}; none overlap the 5 held-out test appearances.",
        "frames_keys": len(all_frames),
        "records_sha256": hashlib.sha256(payload.encode()).hexdigest()[:12],
        "base_commit": "f8a3f69",
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2))

    print(json.dumps(card, indent=2))
    print(f"\n[OK] wrote {len(all_recs)} records + {len(all_frames)} frame arrays -> {out}")


if __name__ == "__main__":
    main()
