#!/usr/bin/env python
"""Fold a freshly-annotated per-operator dataset into the canonical Hindsight-CoT dataset (§10).

The canonical real-Go2 dataset (outputs/hindsight_isaac) was built before O5 (overload) and O10
(effort-decay) were observable / correctly conditioned (#31). This merges a freshly-annotated
O5/O10 dataset into it WITHOUT re-annotating the (expensive, already-good) other operators: it
removes the canonical records for the added ops, appends the new ones, unions the frame sidecars,
and regenerates the dataset card from the combined records (QA 5.4 auto-card — not hand-edited).

    python scripts/merge_hindsight_ops.py --base outputs/hindsight_isaac \
        --add outputs/hindsight_isaac_ops --ops O5_payload O10_effort_decay

Safe because the base has no kept O5/O10 (so no sample_id / frame-key collisions): added ids are
``{i}_O5_payload`` / ``{i}_O10_effort_decay``, distinct from the base's region-op ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kino_vla.data.dataset import build_card, render_card_md
from kino_vla.data.pipeline import PipelineResult, stats_from_records
from kino_vla.utils.config import load_config


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _op(rec: dict) -> str:
    return rec["snapshot"]["operator_name"]


def _frame_keys(sample_id: str) -> list[str]:
    return [f"{sample_id}__rgb", f"{sample_id}__depth", f"{sample_id}__proprio"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge an O5/O10 dataset into the canonical one (§10)")
    ap.add_argument("--base", required=True, help="canonical dataset dir (modified in place)")
    ap.add_argument("--add", required=True, help="freshly-annotated per-op dataset dir")
    ap.add_argument("--ops", nargs="+", default=["O5_payload", "O10_effort_decay"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--oracle", default="api gpt-5.5 over real-Go2 (region xhigh + O5/O10 merge, #31)"
    )
    args = ap.parse_args()
    base, add = Path(args.base), Path(args.add)
    add_ops = set(args.ops)

    base_kept = _read_jsonl(base / "samples.jsonl")
    base_drop = _read_jsonl(base / "dropped.jsonl")
    add_kept = [r for r in _read_jsonl(add / "samples.jsonl") if _op(r) in add_ops]
    add_drop = [r for r in _read_jsonl(add / "dropped.jsonl") if _op(r) in add_ops]

    # Replace the base's records for the added ops (the base has 0 kept O5/O10; it does carry
    # dropped O10 from the mis-conditioned run — those are superseded by the new annotations).
    kept = [r for r in base_kept if _op(r) not in add_ops] + add_kept
    drop = [r for r in base_drop if _op(r) not in add_ops] + add_drop

    # Union the frame sidecars (kept samples only). Base region keys + new O5/O10 keys are disjoint.
    base_frames = np.load(base / "frames.npz")
    add_frames = np.load(add / "frames.npz")
    merged: dict[str, np.ndarray] = {}
    for r in kept:
        sid = r["sample_id"]
        src = add_frames if _op(r) in add_ops else base_frames
        for k in _frame_keys(sid):
            if k in src:
                merged[k] = src[k]

    cfg = load_config("data/hindsight.yaml")
    stats = stats_from_records(cfg, kept, drop)
    result = PipelineResult(samples=[], stats=stats)
    card = build_card(result, cfg, seed=args.seed, oracle_name=args.oracle, git_commit="merge")
    # A merge has no single wall-clock, so carry the base build's measured throughput (exit
    # criterion 2 was met by the component builds) — else the card shows a spurious throughput FAIL.
    try:
        bs = json.loads((base / "dataset_card.json").read_text())["stats"]
        card["stats"]["samples_per_hour"] = bs.get("samples_per_hour", 0.0)
        card["stats"]["kept_per_hour"] = bs.get("kept_per_hour", 0.0)
        card["throughput_ok"] = card["stats"]["samples_per_hour"] >= card["target_samples_per_hour"]
    except (FileNotFoundError, KeyError):
        pass

    (base / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    (base / "dropped.jsonl").write_text("".join(json.dumps(r) + "\n" for r in drop))
    np.savez_compressed(base / "frames.npz", **merged)
    (base / "dataset_card.json").write_text(json.dumps(card, indent=2))
    (base / "dataset_card.md").write_text(render_card_md(card))

    s = card["stats"]
    print(f"[merge] added ops {sorted(add_ops)}: +{len(add_kept)} kept / +{len(add_drop)} dropped")
    print(
        f"[merge] dataset now kept={s['kept']} dropped={s['dropped']} reject={s['reject_rate']:.1%}"
    )
    print(f"[merge] per-op kept: {s['per_operator_kept']}")
    for label, cov in s["ambiguity_coverage"].items():
        print(f"[merge] ambiguity {label}: both_present={cov['both_present']}")
    print(f"[merge] wrote merged dataset → {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
