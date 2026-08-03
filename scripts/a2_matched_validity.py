#!/usr/bin/env python
"""A2 validity gate — is the FROZEN corpus matched pair proprio-indistinguishable? (serves C1/C5).

A2's headline rests on "no proprio-only agent can attribute BOTH sides of the matched pair" — i.e.
its balanced accuracy is capped at 0.5 by construction. That cap is only real if the corpus
matched-O4 vs matched-O2 windows are genuinely indistinguishable in the representation the agents
consume. This re-runs the E1/A1 C2ST machinery on the CORPUS snapshots (not the fresh-app traces)
for both the binding obs48+tau window (60-dim; B1 / the C2ST discriminator) and the VLA proprio
window (25x11; what the Kino-Projector lifts) — a per-corpus re-certification (R7: certification
travels with the config hash).

Writes outputs/eval/a2/matched_validity.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _make(idlist: list[str], npz, key: str, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice each snapshot's window into disjoint T-length sub-windows; lane = snapshot index."""
    xs: list[np.ndarray] = []
    lanes: list[int] = []
    for li, sid in enumerate(idlist):
        w = npz[f"{sid}__{key}"]
        for s in range(0, w.shape[0] - T + 1, T):
            xs.append(w[s : s + T])
            lanes.append(li)
    return np.asarray(xs, dtype=np.float32), np.asarray(lanes)


def main() -> None:
    ap = argparse.ArgumentParser(description="A2 matched-pair proprio-indistinguishability gate")
    ap.add_argument("--corpus", default="outputs/eval/a0/corpus")
    ap.add_argument("--out", default="outputs/eval/a2/matched_validity.json")
    args = ap.parse_args()

    from kino_vla.eval.c2st import c2st

    recs = [
        json.loads(x)
        for x in Path(args.corpus, "samples.jsonl").read_text().splitlines()
        if x
    ]
    npz = np.load(Path(args.corpus, "frames.npz"))

    def ids(op: str) -> list[str]:
        return [
            r["sample_id"]
            for r in recs
            if r["snapshot"]["operator_name"] == op and r.get("ambiguity_pair")
        ]

    o4, o2 = ids("O4_tether"), ids("O2_compliance")
    results = {}
    for key, T in [("binding", 50), ("binding", 100), ("proprio", 25)]:
        xa, la = _make(o4, npz, key, T)
        xb, lb = _make(o2, npz, key, T)
        na = len(set(la.tolist()))
        half = na // 2
        tr_a, tr_b = la < half, lb < half
        r = c2st(
            xa[tr_a], xb[tr_b], xa[~tr_a], xb[~tr_b], la[~tr_a], lb[~tr_b],
            classifier="cnn1d", n_boot=1000, n_perm=500, seed=0, window_len=T, importance=False,
        )
        results[f"{key}_T{T}"] = {
            "representation": key,
            "window_len": T,
            "n_windows": [int(xa.shape[0]), int(xb.shape[0])],
            "auc": round(r.auc, 4),
            "auc_ci": [round(float(x), 4) for x in r.auc_ci],
            "perm_p": round(float(r.perm_p), 4),
            "indistinguishable": bool(r.indistinguishable),
        }
        print(f"{key:8s} T={T:3d}: AUC={r.auc:.3f} CI={tuple(round(x,3) for x in r.auc_ci)} "
              f"perm_p={r.perm_p:.3f} indistinguishable={r.indistinguishable}", flush=True)

    manifest = json.loads(Path(args.corpus, "frozen_manifest.json").read_text())
    out = {
        "corpus": args.corpus,
        "corpus_hash_sha256": manifest.get("corpus_hash_sha256"),
        "commit": manifest.get("commit"),
        "claim": "matched-O4 vs matched-O2 are proprio-indistinguishable on the frozen corpus "
        "(C2ST AUC=0.5) in both the binding obs48+tau and the VLA proprio representation => a "
        "proprio-only attributor is capped at 0.5 balanced accuracy over the matched pair by "
        "construction (C1/C5).",
        "c2st": results,
        "all_indistinguishable": all(v["indistinguishable"] for v in results.values()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nall_indistinguishable={out['all_indistinguishable']}  wrote {args.out}")


if __name__ == "__main__":
    main()
