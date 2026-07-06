#!/usr/bin/env python
"""A1.1 + A1.2 — offline C2ST re-analysis on frozen real-Go2 traces (design-doc R1).

Consumes the higher-seed traces saved by ``scripts/isaac_e1_collect.py`` (with ``--o4-shaping``
for the #49 matched O4) and re-runs the E1 classifier-two-sample-test battery entirely offline —
no Isaac — so the statistics are deterministic and re-runnable from disk (R1: frozen snapshots are
the primary substrate).

  A1.1  Tighten O4↔O2 statistics. More seeds ⇒ a tighter lane-grouped bootstrap CI on the binding
        obs48+torque representation at all T × classifier. Target: decisive (mlp/cnn1d) CI
        half-width ≤ 0.10 with the point estimate ≈ 0.5 and the controls at 1.0. Writes
        ``a1_1_c2st.json`` in the E1 result-card schema (so it drops into the A1.5 triangulation
        and the E1 markdown card).

  A1.2  Close the O3↔O1 temporal leak (regime tightening). The deployed 25-step / calibrated run
        showed the temporal cnn1d separates O3↔O1 at long history (0.64→0.72→0.85) — a history
        encoder exploits the thin-ice *collapse transient*. Restrict the C2ST windows to
        post-collapse-transient (window start ≥ t_collapse + Δ) by advancing ``arm_steps`` over
        Δ ∈ {0, 0.2, 0.5, 1.0}s and re-run. Gate: cnn1d AUC CI covers 0.5 at ALL T ⇒ O3↔O1 is
        "indistinguishable *after* the transient" (promote into T2 with that scoped claim); else
        report the residual leak and keep the taxonomy on O4↔O2 alone (honest scope).

Run:  python scripts/e1_a1_analysis.py \
        --train outputs/eval/a1/traces/e1_trace_train.npz \
        --test  outputs/eval/a1/traces/e1_trace_test.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((REPO / "scripts").resolve()))
from isaac_e1_collect import feature_names, git_commit, lane_trace  # noqa: E402


# --------------------------------------------------------------------------------------
# Load frozen traces → {unit_key: [trace dict, ...]} (reconstruct per-lane from the flat npz)
# --------------------------------------------------------------------------------------
def load_traces(npz_path: Path) -> tuple[dict, dict]:
    d = np.load(npz_path, allow_pickle=True)
    obs48, feat12, tau12 = d["obs48"], d["feat12"], d["tau12"]
    key, lane, manifest = d["key"], d["lane"], d["manifest"]
    t, pen = d["t"], d["pen"]
    fell = d["fell"] if "fell" in d else np.zeros(len(lane), np.int8)
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    out: dict[str, list] = {}
    for ln in np.unique(lane):
        m = lane == ln
        order = np.argsort(t[m], kind="stable")
        k = str(key[m][0])
        tr = {
            "obs48": obs48[m][order], "feat12": feat12[m][order], "tau12": tau12[m][order],
            "manifest": manifest[m][order], "t": t[m][order], "pen": pen[m][order],
            "fell": bool(fell[m][0]), "key": k, "lane": int(ln),
        }
        out.setdefault(k, []).append(tr)
    return out, meta


def _find_key(traces: dict, op_id: str, *, shaped: bool | None = None) -> str:
    """Match a trace key like ``O4[c=6,f_break=1e+09,k=14](shape)`` by op id + shape tag."""
    cands = [k for k in traces if k.split("[")[0] == op_id]
    if shaped is not None:
        cands = [k for k in cands if (("(shape)" in k) == shaped)]
    if not cands:
        raise KeyError(f"no unit for {op_id} (shaped={shaped}) in {list(traces)}")
    return cands[0]


def lane_list(traces: list, feature_set: str, arm: int) -> tuple[list, list]:
    """Per-lane (S,F) segments + [dropped, kept]; drop fallen / empty-segment lanes (R8)."""
    out, dropped = [], 0
    for tr in traces:
        if tr["fell"]:
            dropped += 1
            continue
        seg = lane_trace(tr, feature_set, arm)
        if seg is None or seg.shape[0] == 0:
            dropped += 1
            continue
        out.append(seg)
    return out, [dropped, len(out)]


# --------------------------------------------------------------------------------------
# C2ST VALIDITY GATE — the same-operator negative control. A valid C2ST must score ≈0.5 on
# two disjoint RANDOM halves of the SAME operator; if it separates them, the collection has a
# per-lane / collection-order confound (the reused-app physics drift, #51) and any A/B AUC is
# untrustworthy. This gate is what the interleaved collection is designed to pass.
# --------------------------------------------------------------------------------------
def validity_control(train: dict, test: dict, keys: list, *, binding="obs48_tau", T=100,
                     n_trials=6) -> dict:
    from kino_vla.eval.c2st import c2st, windows_from_lanes

    out: dict = {}
    for key in keys:
        s_tr, _ = lane_list(train[key], binding, 0)
        s_te, _ = lane_list(test[key], binding, 0)
        if len(s_tr) < 4 or len(s_te) < 4:
            out[key.split("[")[0]] = {"note": "too few lanes for a same-op control"}
            continue
        aucs = []
        for trial in range(n_trials):
            rng = np.random.default_rng(trial)
            pi, pj = rng.permutation(len(s_tr)), rng.permutation(len(s_te))
            h, hj = len(s_tr) // 2, len(s_te) // 2
            xa, _ = windows_from_lanes([s_tr[i] for i in pi[:h]], T)
            xb, _ = windows_from_lanes([s_tr[i] for i in pi[h:2 * h]], T)
            xta, la = windows_from_lanes([s_te[i] for i in pj[:hj]], T)
            xtb, lb = windows_from_lanes([s_te[i] for i in pj[hj:2 * hj]], T)
            if not (len(xa) and len(xb) and len(xta) and len(xtb)):
                continue
            r = c2st(xa, xb, xta, xtb, la, lb, classifier="cnn1d", n_boot=100, n_perm=50,
                     seed=trial, window_len=T, importance=False)
            aucs.append(r.auc)
        mean = float(np.mean(aucs)) if aucs else None
        out[key.split("[")[0]] = {"same_op_auc_trials": [round(a, 3) for a in aucs],
                                  "mean_auc": round(mean, 3) if mean is not None else None,
                                  "clean": bool(mean is not None and abs(mean - 0.5) <= 0.1)}
    out["ALL_CLEAN"] = all(v.get("clean") for v in out.values() if "clean" in v)
    return out


# --------------------------------------------------------------------------------------
# A1.1 — tighten O4↔O2 (+ O5↔O10 T4 cert + O2↔O1 power control), E1-card schema
# --------------------------------------------------------------------------------------
def run_a1_1(train: dict, test: dict, meta: dict, *, out_dir: Path, vision: bool) -> dict:
    from kino_vla.eval.c2st import c2st_sweep

    window_lens = [25, 50, 100]
    feature_sets = ["feat12", "obs48", "obs48_tau"]
    classifiers = ["logreg", "mlp", "cnn1d"]
    binding = "obs48_tau"
    arm = 0
    common = dict(n_boot=1000, n_perm=500, alpha=0.05, stride=1)

    o4 = _find_key(train, "O4", shaped=True)
    o2 = _find_key(train, "O2")
    o1 = _find_key(train, "O1")
    o5 = _find_key(train, "O5")
    o10 = _find_key(train, "O10")
    comparisons = [
        {"name": "O4_O2", "key_a": o4, "key_b": o2,
         "disambiguator": {"kind": "appearance", "a": "yellow_adhesive", "b": "brown_mud"}},
        {"name": "O5_O10", "key_a": o5, "key_b": o10, "disambiguator": {"kind": "none"}},
        {"name": "O2_O1", "key_a": o2, "key_b": o1, "disambiguator": {"kind": "none"},
         "is_control": True},
    ]

    report: dict = {"meta": {"commit": git_commit(), "config": "eval/e1_c2st.yaml (A1.1)",
                             "binding": binding, "seeds_train": meta.get("seeds"),
                             "seeds_test": None, "arm_steps": arm,
                             "interleave": meta.get("interleave"),
                             "o4_shaping": meta.get("o4_shaping")},
                    "comparisons": {}}
    # Validity gate FIRST — a same-operator control must be ≈0.5 or the collection is confounded.
    vc = validity_control(train, test, [o4, o2, o1, o5], binding=binding)
    report["validity_control"] = vc
    print(f"[a1.1] VALIDITY (same-op control) ALL_CLEAN={vc['ALL_CLEAN']}: "
          f"{ {k: v.get('mean_auc') for k, v in vc.items() if isinstance(v, dict)} }", flush=True)
    for comp in comparisons:
        name, ka, kb = comp["name"], comp["key_a"], comp["key_b"]
        cinfo: dict = {"key_a": ka, "key_b": kb, "is_control": comp.get("is_control", False),
                       "grid": {}, "counts": {}}
        for fset in feature_sets:
            tr_a, ca = lane_list(train[ka], fset, arm)
            tr_b, cb = lane_list(train[kb], fset, arm)
            te_a, da = lane_list(test[ka], fset, arm)
            te_b, db = lane_list(test[kb], fset, arm)
            cinfo["counts"][fset] = {"train_a": ca, "train_b": cb, "test_a": da, "test_b": db}
            if not (tr_a and tr_b and te_a and te_b):
                print(f"[a1.1] {name}/{fset}: insufficient lanes, skip", flush=True)
                continue
            res = c2st_sweep(tr_a, tr_b, te_a, te_b, window_lens=window_lens,
                             classifiers=classifiers, feature_names=feature_names(fset),
                             feature_set=fset, importance=(fset == binding), seed=0, **common)
            for (t, clf), r in res.items():
                cinfo["grid"][f"{fset}|T{t}|{clf}"] = r.to_dict()
                tag = "CTRL" if comp.get("is_control") else "pair"
                print(f"[a1.1] {tag} {name:>6} {fset:>9} T={t:<3} {clf:>6}: "
                      f"AUC={r.auc:.3f} CI=({r.auc_ci[0]:.2f},{r.auc_ci[1]:.2f}) "
                      f"p={r.perm_p:.3f} indist={r.indistinguishable}", flush=True)
        report["comparisons"][name] = cinfo

    # Vision (real CLIP) — carry E1's certified control, or recompute if --vision.
    report["vision"] = _vision_block(recompute=vision, n_boot=common["n_boot"],
                                     n_perm=common["n_perm"])

    # verdict: decisive-classifier binding cells indistinguishable + CI-halfwidth summary
    dec = ("mlp", "cnn1d")
    o4g = report["comparisons"]["O4_O2"]["grid"]
    dec_cells = [r for k, r in o4g.items()
                 if k.startswith(f"{binding}|") and k.split("|")[-1] in dec]
    hw = [(r["auc_ci"][1] - r["auc_ci"][0]) / 2.0 for r in dec_cells]
    report["verdict"] = {
        "validity_control_clean": report["validity_control"]["ALL_CLEAN"],
        "all_decisive_indistinguishable": all(r["indistinguishable"] for r in dec_cells),
        "max_decisive_ci_halfwidth": round(max(hw), 3) if hw else None,
        "mean_decisive_ci_halfwidth": round(float(np.mean(hw)), 3) if hw else None,
        "target_halfwidth": 0.10,
        "meets_target": bool(hw and max(hw) <= 0.10),
        "certified": bool(report["validity_control"]["ALL_CLEAN"]
                          and all(r["indistinguishable"] for r in dec_cells)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a1_1_c2st.json").write_text(json.dumps(report, indent=2))
    print(f"[a1.1] wrote {out_dir}/a1_1_c2st.json  verdict={report['verdict']}", flush=True)
    return report


def _vision_block(*, recompute: bool, n_boot: int, n_perm: int) -> dict:
    """The O4↔O2 CLIP disambiguator control (yellow_adhesive vs brown_mud)."""
    if not recompute:
        cal = json.loads((REPO / "outputs/eval/e1/calibrated/e1_results.json").read_text())
        v = cal.get("vision", {}).get("O4_O2")
        if v:
            return {"O4_O2": {**v, "carried_from": "outputs/eval/e1/calibrated"}}
    try:
        from kino_vla.eval.c2st import c2st_vision
        from kino_vla.map.clip_appearance import ClipAppearanceEncoder
        from kino_vla.map.clip_segmentation import APPEARANCE_TO_MATERIAL, material_texture

        enc = ClipAppearanceEncoder()
        ma, mb = APPEARANCE_TO_MATERIAL["yellow_adhesive"], APPEARANCE_TO_MATERIAL["brown_mud"]
        ea = enc.embed_batch([material_texture(ma, seed=s) for s in range(24)])
        eb = enc.embed_batch([material_texture(mb, seed=s) for s in range(24)])
        rv = c2st_vision(ea, eb, n_boot=n_boot, n_perm=n_perm, alpha=0.05, seed=0)
        return {"O4_O2": {"kind": "appearance", **rv.to_dict()}}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


# --------------------------------------------------------------------------------------
# A1.2 — regime tightening on O3↔O1 (sweep Δ = arm_steps · dt)
# --------------------------------------------------------------------------------------
def run_a1_2(train: dict, test: dict, meta: dict, *, out_dir: Path, dt: float = 0.02) -> dict:
    from kino_vla.eval.c2st import c2st_sweep

    o3 = _find_key(train, "O3")
    o1 = _find_key(train, "O1")
    binding = "obs48_tau"
    window_lens = [25, 50, 100]
    classifiers = ["logreg", "mlp", "cnn1d"]
    common = dict(n_boot=1000, n_perm=500, alpha=0.05, stride=1)
    arm_grid = [0, 10, 25, 50]  # Δ = 0 / 0.2 / 0.5 / 1.0 s at 50 Hz

    report: dict = {"meta": {"commit": git_commit(), "pair": "O3_O1", "binding": binding,
                             "dt": dt, "arm_grid_steps": arm_grid,
                             "delta_s": [round(a * dt, 3) for a in arm_grid]},
                    "sweep": {}}
    for arm in arm_grid:
        delta = round(arm * dt, 3)
        tr_a, ca = lane_list(train[o3], binding, arm)
        tr_b, cb = lane_list(train[o1], binding, arm)
        te_a, da = lane_list(test[o3], binding, arm)
        te_b, db = lane_list(test[o1], binding, arm)
        row: dict = {"delta_s": delta, "counts": {"train_a": ca, "train_b": cb,
                                                   "test_a": da, "test_b": db}, "grid": {}}
        if not (tr_a and tr_b and te_a and te_b):
            row["grid"] = {"note": "insufficient lanes at this Δ (manifest too short for T)"}
            report["sweep"][f"delta{delta}"] = row
            print(f"[a1.2] Δ={delta}s: insufficient lanes {ca}/{cb}/{da}/{db}", flush=True)
            continue
        res = c2st_sweep(tr_a, tr_b, te_a, te_b, window_lens=window_lens, classifiers=classifiers,
                         feature_names=feature_names(binding), feature_set=binding,
                         importance=False, seed=0, **common)
        for (t, clf), r in res.items():
            row["grid"][f"T{t}|{clf}"] = r.to_dict()
            print(f"[a1.2] Δ={delta}s T={t:<3} {clf:>6}: AUC={r.auc:.3f} "
                  f"CI=({r.auc_ci[0]:.2f},{r.auc_ci[1]:.2f}) p={r.perm_p:.3f} "
                  f"indist={r.indistinguishable}", flush=True)
        # per-Δ verdict on the decisive temporal encoder (cnn1d): CI covers 0.5 at ALL T?
        cnn = [row["grid"].get(f"T{t}|cnn1d") for t in window_lens]
        cnn = [c for c in cnn if c]
        row["cnn1d_all_T_indistinguishable"] = bool(cnn) and all(
            c["indistinguishable"] for c in cnn)
        row["cnn1d_max_auc"] = max((c["auc"] for c in cnn), default=None)
        report["sweep"][f"delta{delta}"] = row

    closed = [k for k, v in report["sweep"].items()
              if v.get("cnn1d_all_T_indistinguishable")]
    report["verdict"] = {
        "closes_at_delta": closed[0] if closed else None,
        "promoted_to_T2": bool(closed),
        "claim": ("O3↔O1 indistinguishable after the collapse transient (window start ≥ "
                  f"t_collapse + {report['sweep'][closed[0]]['delta_s']}s)" if closed
                  else "O3↔O1 temporal leak did NOT close — demote; taxonomy stands on O4↔O2"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a1_2_regime_sweep.json").write_text(json.dumps(report, indent=2))
    print(f"[a1.2] wrote {out_dir}/a1_2_regime_sweep.json  verdict={report['verdict']}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="A1.1/A1.2 offline C2ST re-analysis")
    ap.add_argument("--train", default="outputs/eval/a1/traces/e1_trace_train.npz")
    ap.add_argument("--test", default="outputs/eval/a1/traces/e1_trace_test.npz")
    ap.add_argument("--out", default="outputs/eval/a1")
    ap.add_argument("--vision", action="store_true", help="recompute the CLIP control (else carry)")
    ap.add_argument("--only", choices=["a1_1", "a1_2", "both"], default="both")
    args = ap.parse_args()

    train, meta_tr = load_traces(REPO / args.train if not Path(args.train).is_absolute()
                                 else Path(args.train))
    test, _meta_te = load_traces(REPO / args.test if not Path(args.test).is_absolute()
                                 else Path(args.test))
    out_dir = REPO / args.out
    print(f"[a1] train units={list(train)}\n[a1] test units={list(test)}", flush=True)
    if args.only in ("a1_1", "both"):
        run_a1_1(train, test, meta_tr, out_dir=out_dir, vision=args.vision)
    if args.only in ("a1_2", "both"):
        run_a1_2(train, test, meta_tr, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
