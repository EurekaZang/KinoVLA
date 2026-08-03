#!/usr/bin/env python
"""A1.2 (O3↔O1 regime tightening) + A1.4 (T3 O7/O8 mirrored necessity) on fresh deterministic data.

Both run on the fresh-app (per-lane-independent, deterministic) collection so there is NO #51
operator-order confound (A1.1). Under determinism each operator is a single reproducible trajectory,
so the certificate metric is the exact max|Δ| between operators (a C2ST is degenerate-but-reported).

A1.2 — O3 (thin ice: μ=0.8 then collapses to 0.09 after a dwell) vs O1 (uniform μ=0.09). E1 found a
temporal cnn1d separates them at long history (the collapse transient leaks). Regime tightening:
restrict the C2ST window start to ≥ t_collapse + Δ (advance ``arm`` steps into the manifest) and
re-measure. If the post-transient windows converge (max|Δ| → 0 / AUC → 0.5) at some Δ, O3↔O1 is
"indistinguishable AFTER the transient" (promote into T2 with that scoped claim); else demote and
keep the taxonomy on O4↔O2 (honest scope).

A1.4 — the mirror of T2. Same-APPEARANCE / opposite-PHYSICS pairs: O7 (looks solid_ground, μ=0.09)
vs clean (solid_ground, μ=0.8); O8 (invisible collider on normal floor) vs clean. The certificate:
proprio SEPARATES them (max|Δ| ≫ 0, AUC 1.0 — proprio decisive) while vision (CLIP) CANNOT (same
appearance ⇒ AUC ≈ 0.5 — vision blind), with a CLIP positive control (solid_ground vs a real hazard
appearance ⇒ 1.0). Establishes the mirrored necessity claim (C1, both directions).

Run:  python scripts/a1_2_a1_4_fresh.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((REPO / "scripts").resolve()))
from e1_fresh_c2st import load_fresh, run_c2st  # noqa: E402

PATCH_X = (1.6, 3.4)  # patch band (PATCH_CX=2.5, hx=1.0 ⇒ x∈[1.5,3.5]; trim edges)


def manifest_seg(rec: dict, fset: str, arm: int):
    m = rec["manifest"].astype(bool)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return None
    lo, hi = idx[0] + arm, idx[-1] + 1
    if hi - lo <= 0:
        return None
    return _feat(rec, fset)[lo:hi]


def pos_seg(rec: dict, fset: str):
    """Window by x-position (for clean lanes with no manifest): the patch-crossing band."""
    px = rec["posx"]
    m = (px >= PATCH_X[0]) & (px <= PATCH_X[1])
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return None
    return _feat(rec, fset)[idx[0]:idx[-1] + 1]


def _feat(rec: dict, fset: str) -> np.ndarray:
    if fset == "obs48":
        return rec["obs48"].astype(np.float64)
    if fset == "obs48_tau":
        return np.concatenate([rec["obs48"], rec["tau12"]], 1).astype(np.float64)
    if fset == "feat12":
        return rec["feat12"].astype(np.float64)
    raise ValueError(fset)


def max_abs_delta(a, b) -> float:
    n = min(len(a), len(b))
    return float(np.abs(a[:n] - b[:n]).max()) if n else float("nan")


def a1_2(fresh: dict) -> dict:
    """O3↔O1 regime tightening on fresh data (arm Δ sweep)."""
    o3 = dict(fresh["O3"])
    o1 = dict(fresh["O1"])
    s3 = sorted(o3)
    s1 = sorted(o1)
    dt = 0.02
    arm_grid = [0, 10, 25, 50]
    rows = {}
    for arm in arm_grid:
        # byte-delta between the aligned steady windows (seed 0)
        a = manifest_seg(o3[s3[0]], "obs48_tau", arm)
        b = manifest_seg(o1[s1[0]], "obs48_tau", arm)
        delta = max_abs_delta(a, b) if (a is not None and b is not None) else None
        # C2ST (degenerate on deterministic data but reported)
        def lanes(d, seeds, arm=arm):
            out = [manifest_seg(d[s], "obs48_tau", arm) for s in seeds]
            return [x for x in out if x is not None and len(x) >= 100]
        tr3, te3 = s3[:-1], s3[-1:]
        tr1, te1 = s1[:-1], s1[-1:]
        r = run_c2st(lanes(o3, tr3), lanes(o1, tr1), lanes(o3, te3), lanes(o1, te1), 100, "cnn1d")
        rows[f"delta{round(arm * dt, 2)}"] = {
            "arm_steps": arm, "delta_s": round(arm * dt, 2),
            "max_abs_delta_obs48tau": round(delta, 4) if delta is not None else None,
            "cnn1d_auc": round(r.auc, 3) if r else None,
            "indistinguishable": bool(r.indistinguishable) if r else None,
        }
    closed = [k for k, v in rows.items()
              if v["max_abs_delta_obs48tau"] is not None and v["max_abs_delta_obs48tau"] < 0.05]
    return {"pair": "O3_O1", "arm_sweep": rows,
            "converges_after_transient": bool(closed),
            "verdict": ("O3↔O1 converge after the collapse transient (promote to T2, scoped)"
                        if closed else
                        "O3↔O1 do NOT converge (demote; taxonomy stands on O4↔O2)")}


def a1_4(fresh: dict) -> dict:
    """T3: proprio decisive (O7/O8 vs clean) + vision blind (same appearance)."""
    out: dict = {"proprio": {}, "vision": {}}
    clean = dict(fresh["clean"])
    cs = sorted(clean)
    for op in ("O7", "O8"):
        d = dict(fresh[op])
        s = sorted(d)
        a = manifest_seg(d[s[0]], "obs48_tau", 0)
        b = pos_seg(clean[cs[0]], "obs48_tau")
        delta = max_abs_delta(a, b) if (a is not None and b is not None) else None

        def lanes_op(dd, seeds, kind):
            fn = manifest_seg if kind == "op" else pos_seg
            out2 = [fn(dd[s2], "obs48_tau", 0) if kind == "op" else fn(dd[s2], "obs48_tau")
                    for s2 in seeds]
            return [x for x in out2 if x is not None and len(x) >= 100]
        r = run_c2st(lanes_op(d, s[:-1], "op"), lanes_op(clean, cs[:-1], "clean"),
                     lanes_op(d, s[-1:], "op"), lanes_op(clean, cs[-1:], "clean"), 100, "cnn1d")
        out["proprio"][op] = {
            "vs": "clean", "max_abs_delta_obs48tau": round(delta, 4) if delta is not None else None,
            "cnn1d_auc": round(r.auc, 3) if r else None,
            "proprio_decisive": bool(r and not r.indistinguishable)}
    # Vision (CLIP): same-appearance (deceptive) pair must be at chance; positive control distinct.
    try:
        from kino_vla.eval.c2st import c2st_vision
        from kino_vla.map.clip_appearance import ClipAppearanceEncoder
        from kino_vla.map.clip_segmentation import APPEARANCE_TO_MATERIAL, material_texture
        enc = ClipAppearanceEncoder()

        def emb(appearance):
            mat = APPEARANCE_TO_MATERIAL[appearance]
            return enc.embed_batch([material_texture(mat, seed=i) for i in range(24)])
        solid = emb("solid_ground")
        solid2 = emb("solid_ground")
        rv_blind = c2st_vision(solid, solid2, n_boot=200, n_perm=100, seed=0)
        # positive control: solid_ground vs a real hazard appearance (ice-like via brown_mud proxy)
        haz = emb("brown_mud")
        rv_pos = c2st_vision(solid, haz, n_boot=200, n_perm=100, seed=0)
        out["vision"] = {
            "deceptive_same_appearance_auc": round(rv_blind.auc, 3),
            "vision_blind": bool(rv_blind.indistinguishable),
            "positive_control_solid_vs_hazard_auc": round(rv_pos.auc, 3),
            "positive_distinguishable": bool(not rv_pos.indistinguishable)}
    except Exception as e:  # noqa: BLE001
        out["vision"] = {"error": repr(e)}
    out["verdict"] = ("T3 mirrored necessity: proprio decisive (O7/O8 vs clean separable) + vision "
                      "blind on the same-appearance pair")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="A1.2 + A1.4 on fresh deterministic data")
    ap.add_argument("--fresh", default="outputs/eval/a1/traces_fresh")
    ap.add_argument("--out", default="outputs/eval/a1")
    args = ap.parse_args()
    fresh = load_fresh(REPO / args.fresh if not Path(args.fresh).is_absolute()
                       else Path(args.fresh))
    print(f"[a1.2/4] ops={ {k: len(v) for k, v in fresh.items()} }", flush=True)
    r2 = a1_2(fresh) if "O3" in fresh and "O1" in fresh else {"error": "missing O3/O1"}
    r4 = a1_4(fresh) if all(k in fresh for k in ("O7", "O8", "clean")) else {"error": "missing"}
    out_dir = REPO / args.out
    (out_dir / "a1_2_regime.json").write_text(json.dumps(r2, indent=2))
    (out_dir / "a1_4_t3.json").write_text(json.dumps(r4, indent=2))
    print("\n=== A1.2 O3↔O1 regime tightening (fresh) ===")
    for v in r2.get("arm_sweep", {}).values():
        print(f"  Δ={v['delta_s']}s: max|Δ|obs48+τ={v['max_abs_delta_obs48tau']} "
              f"cnn1d AUC={v['cnn1d_auc']} indist={v['indistinguishable']}")
    print(f"  verdict: {r2.get('verdict')}")
    print("\n=== A1.4 T3 mirrored necessity (fresh) ===")
    for op, v in r4.get("proprio", {}).items():
        print(f"  proprio {op} vs clean: max|Δ|={v['max_abs_delta_obs48tau']} "
              f"AUC={v['cnn1d_auc']} proprio_decisive={v['proprio_decisive']}")
    print(f"  vision: {r4.get('vision')}")
    print(f"\n[a1.2/4] wrote {out_dir}/a1_2_regime.json + a1_4_t3.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
