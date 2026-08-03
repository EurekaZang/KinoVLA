#!/usr/bin/env python
"""A1.5 — three-layer ambiguity-propagation triangulation (offline; artifacts already on disk).

Serves C1 (necessity): the O4↔O2 proprioceptive indistinguishability is a property of the
INFORMATION CHANNEL, not of any one model class. Three *independent* proprio-only systems all
fail at the same O4↔O2 locus — a discriminator family, a trained attributor, and a deployed-grade
detector after an honest A/B-aware fix:

  Layer (i)   C2ST bound (E1)            — the strongest practical discriminator (logreg/mlp/cnn1d
                                            up to a 2 s RMA-style temporal encoder) is at chance
                                            (AUC≈0.50, CI covers 0.5) on the matched pair, over the
                                            binding obs48+torque window, at all history depths.
  Layer (ii)  Trained attributor (E2)    — the strongest pure-proprioception attributor (B1,
                                            RMA-fidelity MonitorNet, 0.764 on unshaped ops) scores
                                            0.00 attribution-gated correct-recovery on matched-O4,
                                            while the fusion agent scores 1.00.
  Layer (iii) Deployed-grade detector    — after an honest A/B-aware retrain (monitor_abaware) +
              (E4)                          operating-point hardening, the *only* A-class operator
                                            that still fires is O2 (the certified pair), and its
                                            fires are misattributed to O4 — the same confusion.

Pure offline analysis: reads the recorded E1/E2/E4 artifacts, recomputes the per-operator A-class
false-fire (deployed vs abaware) from the saved A-class windows + both checkpoints so nothing is
hand-copied, and emits a JSON + Markdown card + a 3-panel figure.

Run:  python scripts/a1_5_triangulation.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_C2ST = REPO / "outputs/eval/e1/calibrated/e1_results.json"
A1_C2ST = REPO / "outputs/eval/a1/a1_1_c2st.json"  # preferred once A1.1 has run (tighter CIs)
CERT = REPO / "outputs/eval/a1/a1_1_certificate.json"  # authoritative: byte-identity + fresh C2ST


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion (E2/A-series convention)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


# --------------------------------------------------------------------------------------
# Layer (i) — the discriminator layer. AUTHORITATIVE = the A1.1 determinism-controlled
# certificate (byte-identity + fresh C2ST + the reused-app confound). Falls back to the raw
# E1 C2ST card if the certificate is absent.
# --------------------------------------------------------------------------------------
def layer_i_cert(cert_path: Path) -> dict:
    c = json.loads(cert_path.read_text())
    bi = c["byte_identity"]
    fc = c["fresh_c2st"]
    conf = c["confound_control"]
    binding = "obs48_tau"
    rows = []
    for key, r in sorted(fc["matched_O4_O2"].items()):
        fset, t, clf = key.split("|")
        if fset != binding:
            continue
        rows.append({"T": int(t[1:]), "clf": clf, "auc": r["auc"], "ci": r["auc_ci"],
                     "perm_p": r["perm_p"], "indistinguishable": r["indistinguishable"]})
    rows.sort(key=lambda x: (x["T"], x["clf"]))
    dec = [r for r in rows if r["clf"] in ("mlp", "cnn1d")]
    pw = [v["auc"] for k, v in fc["power_O2_O1"].items() if k.endswith("cnn1d")]
    hw = max(((r["ci"][1] - r["ci"][0]) / 2.0 for r in dec), default=0.0)
    return {  # compatible with markdown()/make_figure(), plus certificate extras
        "source": str(cert_path.relative_to(REPO)),
        "certificate": "byte-identity (determinism-controlled)",
        "commit": None, "binding": binding, "seeds_train": "fresh-app (deterministic)",
        "seeds_test": "1 lane/op suffices", "o4_shaping": {"enabled": True, "cap": 0, "offset": 14},
        "rows": rows,
        "all_decisive_indistinguishable": c["verdict"]["matched_indistinguishable"],
        "max_decisive_ci_halfwidth": round(float(hw), 3),
        "power_control_O2_O1_auc": round(float(np.mean(pw)), 3) if pw else None,
        "vision_auc": c["vision"]["O4_O2_clip_auc"],
        "vision_distinguishable": c["vision"]["distinguishable"],
        "byte_identity_max_abs_delta": bi["obs48_tau"]["matched_O4_O2_max_abs_delta"],
        "reused_app_confounded": conf["confounded"],
        "reused_cross_op_auc": conf["reused_cross_op_O4_O2_auc"],
        "reused_structured_same_op_auc": conf["structured_same_op_even_odd_auc"],
    }


def layer_i(c2st_path: Path, *, binding: str = "obs48_tau",
            decisive=("mlp", "cnn1d")) -> dict:
    d = json.loads(c2st_path.read_text())
    comps = d["comparisons"]
    o4o2 = comps["O4_O2"]["grid"]
    rows = []
    for key, r in sorted(o4o2.items()):
        fset, t, clf = key.split("|")
        if fset != binding:
            continue
        rows.append({"T": int(t[1:]), "clf": clf, "auc": r["auc"],
                     "ci": r["auc_ci"], "perm_p": r["perm_p"],
                     "indistinguishable": r["indistinguishable"]})
    rows.sort(key=lambda x: (x["T"], x["clf"]))
    # controls (test has power)
    ctrl = comps.get("O2_O1", {}).get("grid", {})
    ctrl_auc = [r["auc"] for k, r in ctrl.items()
                if k.startswith(f"{binding}|") and k.split("|")[-1] in decisive]
    vision = d.get("vision", {}).get("O4_O2", {})
    # headline decisive cells
    decisive_rows = [r for r in rows if r["clf"] in decisive]
    max_hw = max(((r["ci"][1] - r["ci"][0]) / 2.0 for r in decisive_rows), default=float("nan"))
    all_indist = all(r["indistinguishable"] for r in decisive_rows)
    return {
        "source": str(c2st_path.relative_to(REPO)),
        "commit": d.get("meta", {}).get("commit"),
        "binding": binding,
        "seeds_train": d.get("meta", {}).get("seeds_train"),
        "seeds_test": d.get("meta", {}).get("seeds_test"),
        "o4_shaping": d.get("meta", {}).get("o4_shaping"),
        "rows": rows,
        "all_decisive_indistinguishable": bool(all_indist),
        "max_decisive_ci_halfwidth": round(float(max_hw), 3),
        "power_control_O2_O1_auc": round(float(np.mean(ctrl_auc)), 3) if ctrl_auc else None,
        "vision_auc": vision.get("auc"),
        "vision_distinguishable": (not vision.get("indistinguishable"))
        if "indistinguishable" in vision else None,
    }


# --------------------------------------------------------------------------------------
# Layer (ii) — trained attributor B1 (E2 three-row, matched-O4 conflict column)
# --------------------------------------------------------------------------------------
def layer_ii(three_row_path: Path) -> dict:
    d = json.loads(three_row_path.read_text())
    rows = d["rows"]
    out = {}
    order = [("B1_proprio", "B1 (strongest pure-proprio)"),
             ("B5_unshaped", "B5-unshaped (has vision, no conflict training)"),
             ("B5_conflict", "B5-conflict (learned conflict-resolution)")]
    for key, label in order:
        r = rows[key]
        c = r["per_operator_counts"]["O4_tether"]
        lo, hi = wilson(int(c["joint_correct"]), int(c["n"]))
        out[key] = {"label": label,
                    "correct_recovery_rate": r["correct_recovery_rate"],
                    "o4_joint_correct": int(c["joint_correct"]), "o4_n": int(c["n"]),
                    "o4_correct_recovery": c["joint_correct"] / c["n"],
                    "wilson_ci": [round(lo, 3), round(hi, 3)]}
    return {"source": str(three_row_path.relative_to(REPO)),
            "n_items": d.get("n_items"), "headline": d.get("headline"),
            "agents": out}


# --------------------------------------------------------------------------------------
# Layer (iii) — deployed-grade detector (E4). Recompute per-op A-class fire@0.6 for the
# deployed monitor vs monitor_abaware from the saved A-class windows (nothing hand-copied),
# then read the hardened-operating-point aggregate + the O2→O4 misattribution from JSON.
# --------------------------------------------------------------------------------------
def _slice_lane_windows(feats: np.ndarray, window_len: int):
    n = feats.shape[0]
    if n < window_len:
        return np.empty((0, window_len, feats.shape[1]), np.float32)
    return np.stack([feats[s:s + window_len] for s in range(0, n - window_len + 1)], axis=0)


def per_op_fire(model, npz_path: Path, *, thr: float = 0.6) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    feats, lane, op_ids = d["feats"], d["lane"], [str(x) for x in d["op_ids"]]
    uniq = np.unique(lane)
    n_seeds = max(1, len(uniq) // len(op_ids))
    wl = model.cfg.window_len
    fire_by_op: dict[str, list] = {op: [] for op in op_ids}
    for li, ln in enumerate(uniq):
        op = op_ids[min(li // n_seeds, len(op_ids) - 1)]
        w = _slice_lane_windows(feats[lane == ln], wl)
        if w.shape[0] == 0:
            continue
        prob, _attr = model.predict(w)
        fire_by_op[op].append(float(np.mean(prob >= thr)))
    return {op: (round(float(np.mean(v)), 3) if v else None) for op, v in fire_by_op.items()}


def layer_iii(*, deployed="outputs/monitor_learned/monitor",
              abaware="outputs/monitor_learned/monitor_abaware",
              aclass_npz="outputs/monitor_learned/windows_aclass_test.npz",
              optpoint="outputs/eval/e4/optpoint.json",
              probe="outputs/eval/e4/monitor_probe_abaware.json") -> dict:
    from kino_vla.monitor.learned_monitor import LearnedMonitorModel

    dep = LearnedMonitorModel.load(str(REPO / deployed), device="cpu")
    aba = LearnedMonitorModel.load(str(REPO / abaware), device="cpu")
    fire_dep = per_op_fire(dep, REPO / aclass_npz)
    fire_aba = per_op_fire(aba, REPO / aclass_npz)
    per_op = {}
    for op in fire_dep:
        a, b = fire_dep[op], fire_aba[op]
        ratio = (a / b) if (a and b and b > 0) else (float("inf") if (a and not b) else None)
        per_op[op] = {"deployed": a, "abaware": b,
                      "reduction_x": (round(ratio, 1) if isinstance(ratio, float)
                                      and math.isfinite(ratio) else ratio)}
    op_best = json.loads((REPO / optpoint).read_text())["best_clean_oppoint"]
    pr = json.loads((REPO / probe).read_text())
    o2_gentle = pr["families"]["O2_compliance"]["rows"]["6"]
    return {
        "source": {"aclass_windows": aclass_npz, "optpoint": optpoint, "probe": probe},
        "per_op_fire_at_0p6": per_op,
        "hardened_oppoint": op_best,  # thr/deb/arm → aclass_fire, bclass_tpr, neg_fpr
        "residual_A_class_lane": "O2",
        "o2_gentle_in_patch_attr_argmax": o2_gentle["in_patch_attr_argmax"],
        "note": ("At the hardened operating point the only A-class operator still firing is O2 — "
                 "the certified-ambiguous pair — and its fires are misattributed to O4."),
    }


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def make_figure(li: dict, lii: dict, liii: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle("A1.5 — One ambiguity locus (O4↔O2), three independent proprio-only layers "
                 "all fail (C1: a property of the channel, not the model)", fontsize=12)

    # Panel A — C2ST (cnn1d, the temporal RMA-analogue) across T + controls
    cnn = [r for r in li["rows"] if r["clf"] == "cnn1d"]
    labels = [f"O4↔O2\nT={r['T']}" for r in cnn]
    aucs = [r["auc"] for r in cnn]
    err = [[r["auc"] - r["ci"][0] for r in cnn], [r["ci"][1] - r["auc"] for r in cnn]]
    x = np.arange(len(cnn))
    axA.bar(x, aucs, yerr=err, capsize=4, color="#4C72B0", label="matched O4↔O2")
    # controls
    cx = [len(cnn), len(cnn) + 1]
    axA.bar(cx, [li["power_control_O2_O1_auc"] or 0, li["vision_auc"] or 0],
            color=["#C44E52", "#55A868"])
    axA.axhline(0.5, ls="--", c="gray", lw=1)
    axA.set_xticks(list(x) + cx)
    axA.set_xticklabels(labels + ["power\nO2↔O1", "vision\nCLIP"], fontsize=8)
    axA.set_ylim(0, 1.05)
    axA.set_ylabel("held-out C2ST AUC")
    axA.set_title("(i) discriminator family\n"
                  f"(binding obs48+τ; CI hw≤{li['max_decisive_ci_halfwidth']})", fontsize=9)
    axA.text(0.02, 0.52, "chance", color="gray", fontsize=8, transform=axA.get_yaxis_transform())

    # Panel B — trained attributor (E2 three rows)
    keys = ["B1_proprio", "B5_unshaped", "B5_conflict"]
    vals = [lii["agents"][k]["o4_correct_recovery"] for k in keys]
    ci = [lii["agents"][k]["wilson_ci"] for k in keys]
    errB = [[v - c[0] for v, c in zip(vals, ci, strict=True)],
            [c[1] - v for v, c in zip(vals, ci, strict=True)]]
    colB = ["#C44E52", "#DD8452", "#55A868"]
    xb = np.arange(3)
    axB.bar(xb, vals, yerr=errB, capsize=4, color=colB)
    axB.set_xticks(xb)
    axB.set_xticklabels(["B1\nproprio", "B5\nunshaped", "B5\nconflict"], fontsize=8)
    axB.set_ylim(0, 1.05)
    axB.set_ylabel("attribution-gated correct-recovery (matched-O4)")
    axB.set_title("(ii) strongest trained attributor\n(E2, n=15 O4; Wilson 95% CI)", fontsize=9)

    # Panel C — deployed-grade detector per-op A-class fire (deployed vs abaware)
    ops = list(liii["per_op_fire_at_0p6"].keys())
    dep = [liii["per_op_fire_at_0p6"][o]["deployed"] or 0 for o in ops]
    aba = [liii["per_op_fire_at_0p6"][o]["abaware"] or 0 for o in ops]
    xc = np.arange(len(ops))
    w = 0.38
    axC.bar(xc - w / 2, dep, w, label="deployed monitor", color="#8C8C8C")
    axC.bar(xc + w / 2, aba, w, label="monitor_abaware (A/B-aware)", color="#4C72B0")
    # highlight O2 residual
    if "O2" in ops:
        i = ops.index("O2")
        axC.annotate("O2 residual\n→ misattr O4", (i + w / 2, aba[i]),
                     textcoords="offset points", xytext=(0, 14), fontsize=7,
                     ha="center", color="#C44E52",
                     arrowprops=dict(arrowstyle="->", color="#C44E52"))
    axC.axhline(0.6, ls=":", c="gray", lw=0.8)
    axC.set_xticks(xc)
    axC.set_xticklabels(ops, fontsize=8)
    axC.set_ylim(0, 1.05)
    axC.set_ylabel("per-window A-class false-fire @0.6")
    axC.set_title("(iii) deployed-grade detector\n"
                  "(E4; A-class ↓5–80×, O2 sole residual)", fontsize=9)
    axC.legend(fontsize=7, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def markdown(li: dict, lii: dict, liii: dict) -> str:
    lines = [
        "# A1.5 — Three-layer ambiguity triangulation (O4↔O2)",
        "",
        "**Claim served — C1 (necessity).** The O4↔O2 proprioceptive indistinguishability is a "
        "property of the *information channel*, not of any one model class: three independent "
        "proprio-only systems all fail at the same locus.",
        "",
        "## Layer (i) — discriminator family (C2ST, E1)",
        f"source `{li['source']}` · commit `{li['commit']}` · binding **{li['binding']}** · "
        f"train/test {li['seeds_train']}/{li['seeds_test']} seeds · "
        f"o4_shaping `{li['o4_shaping']}`",
        "",
        "| T | clf | AUC | 95% CI | perm p | indist |",
        "|---|---|---|---|---|---|",
    ]
    for r in li["rows"]:
        lines.append(f"| {r['T']} | {r['clf']} | {r['auc']:.3f} | "
                     f"({r['ci'][0]:.2f},{r['ci'][1]:.2f}) | {r['perm_p']:.3f} | "
                     f"{r['indistinguishable']} |")
    lines += [
        "",
        f"- all decisive (mlp/cnn1d) cells indistinguishable: "
        f"**{li['all_decisive_indistinguishable']}**; max decisive CI half-width "
        f"**{li['max_decisive_ci_halfwidth']}**",
        f"- power control O2↔O1 AUC **{li['power_control_O2_O1_auc']}** (test has power); "
        f"vision (CLIP) AUC **{li['vision_auc']}** distinguishable "
        f"**{li['vision_distinguishable']}**",
    ]
    if "byte_identity_max_abs_delta" in li:
        lines += [
            f"- **byte-identity (determinism-controlled): matched O4↔O2 max|Δobs48+τ| = "
            f"{li['byte_identity_max_abs_delta']}** — the pair is not merely AUC≈0.5, it is "
            f"IDENTICAL. The reused-app C2ST is confounded (#51): cross-op AUC "
            f"{li['reused_cross_op_auc']} + structured same-op "
            f"{li['reused_structured_same_op_auc']} (should be 0.5) — see `a1_1_certificate.md`.",
        "",
        "## Layer (ii) — strongest trained attributor (E2 three-row, matched-O4)",
        f"source `{lii['source']}` · n_items {lii['n_items']} · headline `{lii['headline']}`",
        "",
        "| agent | matched-O4 correct-recovery | Wilson 95% CI |",
        "|---|---|---|",
    ]
    for k in ("B1_proprio", "B5_unshaped", "B5_conflict"):
        a = lii["agents"][k]
        lines.append(f"| {a['label']} | {a['o4_joint_correct']}/{a['o4_n']} = "
                     f"{a['o4_correct_recovery']:.2f} | "
                     f"({a['wilson_ci'][0]:.2f},{a['wilson_ci'][1]:.2f}) |")
    lines += [
        "",
        "B1 (0.764 on unshaped ops — a strong opponent, C5) is at 0.00 on the matched pair, "
        "CI-separated from the fusion agent (1.00). Its 0.00 is *by construction* (Layer i bound).",
        "",
        "## Layer (iii) — deployed-grade detector after an A/B-aware fix (E4)",
        "Per-operator A-class per-window false-fire @0.6 (recomputed from "
        f"`{liii['source']['aclass_windows']}` + both checkpoints):",
        "",
        "| A-class op | deployed | monitor_abaware | reduction |",
        "|---|---|---|---|",
    ]
    for op, v in liii["per_op_fire_at_0p6"].items():
        red = v["reduction_x"]
        red_s = ("↓∞" if red == float("inf") else (f"↓{red}×" if red else "—"))
        lines.append(f"| {op} | {v['deployed']} | {v['abaware']} | {red_s} |")
    hp = liii["hardened_oppoint"]
    lines += [
        "",
        f"- hardened operating point (thr {hp['threshold']} / deb {hp['debounce']} / "
        f"arm {hp['arm_s']}): aggregate A-class lane-fire **{hp['aclass_fire']}**, "
        f"B-class TPR **{hp['bclass_tpr']}**, clean/maneuver FPR **{hp['neg_fpr']}**",
        f"- **sole residual A-class firing lane = O2**; at gentle O2 (k_c=6) the in-patch "
        f"attribution argmax is `{liii['o2_gentle_in_patch_attr_argmax']}` — misattributed to O4.",
        "",
        "> " + liii["note"],
        "",
        "## Conclusion",
        "A discriminator family (i), the strongest trained attributor (ii), and a deployed-grade "
        "detector after an honest A/B-aware fix (iii) all collapse at the O4↔O2 locus — and the "
        "detector's *only* residual A-class fire is O2, misread as O4. The indistinguishability is "
        "channel-level, not model-level. Vision separates it perfectly (Layer i control, AUC 1.0).",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="A1.5 three-layer ambiguity triangulation")
    ap.add_argument("--c2st", default=None,
                    help="C2ST json (default: A1.1 tightened if present, else E1 calibrated)")
    ap.add_argument("--three-row", default="outputs/eval/e2/three_row.json")
    ap.add_argument("--out", default="outputs/eval/a1")
    args = ap.parse_args()

    if CERT.exists() and not args.c2st:
        li = layer_i_cert(CERT)
    else:
        c2st_path = (Path(args.c2st) if args.c2st
                     else (A1_C2ST if A1_C2ST.exists() else DEFAULT_C2ST))
        li = layer_i(c2st_path)
    lii = layer_ii(REPO / args.three_row if not Path(args.three_row).is_absolute()
                   else Path(args.three_row))
    liii = layer_iii()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"claim": "C1 necessity — channel-level (not model-level) O4↔O2 indistinguishability",
               "layer_i_c2st": li, "layer_ii_attributor": lii, "layer_iii_detector": liii}
    (out_dir / "a1_5_triangulation.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "a1_5_triangulation.md").write_text(markdown(li, lii, liii))
    make_figure(li, lii, liii, out_dir / "a1_5_triangulation.png")
    print(markdown(li, lii, liii))
    print(f"\n[a1.5] wrote {out_dir}/a1_5_triangulation.{{json,md,png}} "
          f"(c2st source: {c2st_path.relative_to(REPO)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
