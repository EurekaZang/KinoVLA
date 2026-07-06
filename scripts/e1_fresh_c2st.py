#!/usr/bin/env python
"""A1 determinism arbiter — C2ST on the FRESH-APP-per-lane collection (no cross-lane residual).

Loads the ``lane_<op>_<seed>.npz`` files written by ``isaac_e1_singlelane.py`` (each collected in
its own Isaac app) and runs the O4↔O2 C2ST + the same-operator validity control on this
confound-free substrate. If the reused-app C2ST scores O4↔O2 ≈ 1.0 while this fresh-app one is at
chance (and the same-operator control is also at chance), the reused-app separation is the #51
operator-order physics-residual confound, not a real proprioceptive difference — and the E1
matched-pair claim holds under determinism.

Run:  python scripts/e1_fresh_c2st.py --dir outputs/eval/a1/traces_fresh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_fresh(d: Path) -> dict:
    """Return {op: [(seed, {obs48,tau12,feat12,manifest,pen,fell}), ...]} sorted by seed."""
    out: dict[str, list] = {}
    for f in sorted(d.glob("lane_*.npz")):
        z = np.load(f, allow_pickle=True)
        op = str(z["op"])
        rec = {k: z[k] for k in ("obs48", "tau12", "feat12", "manifest", "pen")}
        rec["fell"] = bool(z["fell"])
        if "posx" in z:  # position-based windowing for clean/no-manifest lanes (A1.4)
            rec["posx"] = z["posx"]
        out.setdefault(op, []).append((int(z["seed"]), rec))
    for op in out:
        out[op].sort(key=lambda x: x[0])
    return out


def _longest_run(m: np.ndarray) -> tuple[int, int]:
    best = (0, 0)
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def seg(rec: dict, fset: str) -> np.ndarray | None:
    lo, hi = _longest_run(rec["manifest"])
    if hi - lo <= 0:
        return None
    if fset == "obs48":
        return rec["obs48"][lo:hi].astype(np.float64)
    if fset == "obs48_tau":
        return np.concatenate([rec["obs48"][lo:hi], rec["tau12"][lo:hi]], 1).astype(np.float64)
    if fset == "feat12":
        return rec["feat12"][lo:hi].astype(np.float64)
    raise ValueError(fset)


def lanes_of(recs: list, fset: str, seeds: set) -> list:
    out = []
    for s, r in recs:
        if s not in seeds or r["fell"]:
            continue
        g = seg(r, fset)
        if g is not None and g.shape[0] > 0:
            out.append(g)
    return out


def run_c2st(a_tr, b_tr, a_te, b_te, T, clf, seed=0):
    from kino_vla.eval.c2st import c2st, windows_from_lanes
    xa, _ = windows_from_lanes(a_tr, T)
    xb, _ = windows_from_lanes(b_tr, T)
    xta, la = windows_from_lanes(a_te, T)
    xtb, lb = windows_from_lanes(b_te, T)
    if not (len(xa) and len(xb) and len(xta) and len(xtb)):
        return None
    return c2st(xa, xb, xta, xtb, la, lb, classifier=clf, n_boot=500, n_perm=300, seed=seed,
                window_len=T, importance=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="A1 fresh-app determinism-arbiter C2ST")
    ap.add_argument("--dir", default="outputs/eval/a1/traces_fresh")
    ap.add_argument("--test-frac", type=float, default=0.4)
    ap.add_argument("--out", default="outputs/eval/a1/a1_fresh_c2st.json")
    args = ap.parse_args()

    data = load_fresh(REPO / args.dir if not Path(args.dir).is_absolute() else Path(args.dir))
    print(f"[fresh] ops={ {k: len(v) for k, v in data.items()} }", flush=True)
    o4 = data.get("O4", [])
    o2 = data.get("O2", [])
    all_seeds = sorted({s for s, _ in o4} & {s for s, _ in o2})
    n_test = max(1, int(round(args.test_frac * len(all_seeds))))
    test_seeds = set(all_seeds[-n_test:])
    train_seeds = set(all_seeds[:-n_test])
    print(f"[fresh] shared seeds={all_seeds} train={sorted(train_seeds)} test={sorted(test_seeds)}",
          flush=True)

    report: dict = {"dir": args.dir, "train_seeds": sorted(train_seeds),
                    "test_seeds": sorted(test_seeds), "cross": {}, "same_op_control": {}}
    for fset in ("obs48", "obs48_tau", "feat12"):
        for T in (25, 50, 100):
            for clf in ("logreg", "mlp", "cnn1d"):
                r = run_c2st(lanes_of(o4, fset, train_seeds), lanes_of(o2, fset, train_seeds),
                             lanes_of(o4, fset, test_seeds), lanes_of(o2, fset, test_seeds), T, clf)
                if r is None:
                    continue
                report["cross"][f"{fset}|T{T}|{clf}"] = r.to_dict()
                print(f"[fresh] O4vsO2 {fset:>9} T={T:<3} {clf:>6}: AUC={r.auc:.3f} "
                      f"CI=({r.auc_ci[0]:.2f},{r.auc_ci[1]:.2f}) p={r.perm_p:.3f} "
                      f"indist={r.indistinguishable}", flush=True)

    # same-operator control (validity gate): split ONE operator's seeds in half, fresh data →
    # must be ≈0.5, else even the fresh collection has per-lane structure.
    for op, recs in (("O2", o2), ("O4", o4)):
        seeds = sorted(s for s, _ in recs)
        half = len(seeds) // 2
        grp_a, grp_b = set(seeds[0::2]), set(seeds[1::2])  # interleaved halves
        te = set(seeds[-max(1, half // 2):])
        for T in (50, 100):
            r = run_c2st(lanes_of(recs, "obs48_tau", grp_a - te),
                         lanes_of(recs, "obs48_tau", grp_b - te),
                         lanes_of(recs, "obs48_tau", grp_a & te or grp_a),
                         lanes_of(recs, "obs48_tau", grp_b & te or grp_b), T, "cnn1d")
            if r:
                report["same_op_control"][f"{op}|T{T}|cnn1d"] = r.to_dict()
                print(f"[fresh] SAME-OP {op} split obs48_tau T={T} cnn1d: AUC={r.auc:.3f} "
                      f"indist={r.indistinguishable}", flush=True)

    binding = [v for k, v in report["cross"].items()
               if k.startswith("obs48_tau|") and k.endswith("cnn1d")]
    report["summary"] = {
        "binding_cnn1d_aucs": [round(v["auc"], 3) for v in binding],
        "binding_all_indistinguishable": all(v["indistinguishable"] for v in binding) if binding
        else None,
    }
    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[fresh] summary={report['summary']}\n[fresh] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
