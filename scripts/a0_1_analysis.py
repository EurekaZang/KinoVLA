#!/usr/bin/env python
"""A0.1 offline analysis — assemble the determinism certificate (CPU, deterministic, re-runnable).

Consumes the artifacts of scripts/a0_1_determinism.py (deep + naive) and the fresh-app references
(scripts/isaac_e1_singlelane.py) and produces outputs/eval/a0/a0_1_certificate.{json,md}:

  1. cross-ordering identity  — max|Δ obs48⊕τ| for the same (op,seed) run in two orderings, per
     mechanism (deep should be 0; naive should be ≫0).
  2. gold match               — deep(fixed-y) vs the fresh-app reference for each (op,seed): does
     the in-app deep_reset reproduce the byte-identical fresh-process lane (not just self-agree)?
  3. same-op validity gate    — a structured same-operator C2ST (even-seed vs odd-seed O4 / O2): a
     VALID deterministic collection gives AUC≈0.5 (no operator-ORDER residue to exploit; contrast
     A1's reused-app structured same-op AUC 1.0 / 0.893).
"""

from __future__ import annotations

import json

import numpy as np

from kino_vla.eval.c2st import c2st_holdout, windows_from_lanes
from kino_vla.utils.config import REPO_ROOT

A0 = REPO_ROOT / "outputs" / "eval" / "a0"
DET = A0 / "determinism"
FRESH = A0 / "_fresh"
LANES = [("O4", 0), ("O4", 1), ("O4", 2), ("O2", 0), ("O2", 1), ("O2", 2), ("O1", 0)]


def _fresh_lane(op: str, seed: int) -> np.ndarray | None:
    p = FRESH / f"lane_{op}_{seed}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return np.concatenate([d["obs48"], d["tau12"]], axis=1).astype(np.float32)  # (S, 60)


def main() -> int:
    cert: dict = {"lanes": [f"{o}_{s}" for o, s in LANES]}

    # ---- (1) cross-ordering identity per mechanism -----------------------------------------------
    cross = {}
    for mech in ("deep", "naive"):
        sp = DET / f"summary_{mech}.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            cross[mech] = {
                "all_identical": s["all_identical"],
                "max_cross_delta": s["max_cross_delta"],
                "per_lane": {k: v["max_abs_delta"] for k, v in s["cross_ordering"].items()},
            }
    cert["cross_ordering"] = cross

    # ---- (2) gold match: deep(fixed-y) vs fresh-app reference ------------------------------------
    gold = {}
    deep_npz = DET / "lanes_deep.npz"
    if deep_npz.exists():
        dd = np.load(deep_npz)
        for op, s in LANES:
            key = f"{op}_{s}"
            deep_arr = dd.get(f"cross_A_{key}")
            fresh_arr = _fresh_lane(op, s)
            if deep_arr is None or fresh_arr is None:
                gold[key] = {"status": "missing"}
                continue
            n = min(len(deep_arr), len(fresh_arr))
            delta = float(np.abs(deep_arr[:n] - fresh_arr[:n]).max()) if n else float("nan")
            gold[key] = {"max_abs_delta": delta, "identical": bool(delta == 0.0),
                         "len_deep": int(len(deep_arr)), "len_fresh": int(len(fresh_arr))}
    cert["gold_match_deep_vs_fresh"] = gold
    cert["gold_match_all_identical"] = bool(
        gold and all(v.get("identical") for v in gold.values())
    )

    # ---- (3) same-op validity gate: even-seed vs odd-seed C2ST (deep) ----------------------------
    same_gate = {}
    if deep_npz.exists():
        dd = np.load(deep_npz)
        for op in ("O4", "O2"):
            lanes = []
            for i in range(10):
                k = f"sameop_{op}_{i}"
                if k in dd:
                    lanes.append((i, dd[k].astype(np.float64)))
            if len(lanes) < 4:
                same_gate[op] = {"status": "insufficient_lanes", "n": len(lanes)}
                continue
            even = [a for i, a in lanes if i % 2 == 0]
            odd = [a for i, a in lanes if i % 2 == 1]
            res = {}
            for t in (25, 50):
                xa, la = windows_from_lanes(even, t)
                xb, lb = windows_from_lanes(odd, t)
                r = c2st_holdout(xa, xb, la, lb, classifier="cnn1d", window_len=t,
                                 feature_set="obs48_tau", importance=False, test_frac=0.4,
                                 n_boot=500, n_perm=300, seed=0)
                res[f"T{t}"] = {"auc": round(r.auc, 3), "ci": [round(r.auc_ci[0], 3),
                               round(r.auc_ci[1], 3)], "perm_p": round(r.perm_p, 3),
                               "indistinguishable": r.indistinguishable}
            same_gate[op] = res
    cert["same_op_validity_gate"] = same_gate

    # ---- verdict --------------------------------------------------------------------------------
    deep_ok = cross.get("deep", {}).get("all_identical", False)
    gold_ok = cert["gold_match_all_identical"]
    gate_ok = all(
        v.get(f"T{t}", {}).get("indistinguishable", False)
        for v in same_gate.values() if isinstance(v, dict) and "status" not in v
        for t in (25, 50)
    ) if same_gate else False
    cert["verdict"] = {
        "deep_cross_ordering_byte_identical": deep_ok,
        "deep_reproduces_fresh_gold": gold_ok,
        "same_op_gate_indistinguishable": gate_ok,
        "A0_1_PASS": bool(deep_ok and gold_ok),
    }

    A0.mkdir(parents=True, exist_ok=True)
    (A0 / "a0_1_certificate.json").write_text(json.dumps(cert, indent=2))

    # markdown summary
    lines = ["# A0.1 determinism certificate\n"]
    lines.append("## Cross-ordering identity (same (op,seed), 2 orderings, 1 reused app)\n")
    lines.append("| mechanism | all identical | max cross Δ |")
    lines.append("|---|---|---|")
    for m, c in cross.items():
        lines.append(f"| {m} | {c['all_identical']} | {c['max_cross_delta']:.6g} |")
    lines.append("\n## Gold match: deep(fixed-y) vs fresh-app reference\n")
    lines.append("| lane | max|Δ| | identical |")
    lines.append("|---|---|---|")
    for k, v in gold.items():
        if "max_abs_delta" in v:
            lines.append(f"| {k} | {v['max_abs_delta']:.6g} | {v['identical']} |")
    lines.append("\n## Same-op validity gate (even vs odd seed C2ST; want AUC≈0.5)\n")
    lines.append("| op | T | AUC | CI | indistinguishable |")
    lines.append("|---|---|---|---|---|")
    for op, r in same_gate.items():
        if "status" in r:
            lines.append(f"| {op} | — | {r['status']} | | |")
            continue
        for t in ("T25", "T50"):
            e = r[t]
            lines.append(f"| {op} | {t} | {e['auc']} | {e['ci']} | {e['indistinguishable']} |")
    lines.append(f"\n## Verdict\n\n```\n{json.dumps(cert['verdict'], indent=2)}\n```\n")
    (A0 / "a0_1_certificate.md").write_text("\n".join(lines))
    print(json.dumps(cert["verdict"], indent=2))
    print(f"gold match: {gold}")
    print(f"same-op gate: {same_gate}")
    print(f"wrote {A0}/a0_1_certificate.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
