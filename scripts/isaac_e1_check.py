#!/usr/bin/env python
"""E1 gate — proprioceptive-indistinguishability classifier-two-sample test on the REAL Go2.

Collects obs+history traces (scripts/isaac_e1_collect.collect_units) for the three matched
ambiguity pairs + the power control, then runs the C2ST battery (kino_vla.eval.c2st) over the
deployed-12 / raw-48 / raw-48+torque representations at history depths T∈{25,50,100} with both a
strong flat learner (mlp) and a temporal RMA-analogue (cnn1d). The BINDING claim is the raw
obs48+torque representation at all T (what an RMA baseline consumes).

Writes a result card (outputs/eval/e1/e1_results.{json,md}) and a PASS/FAIL verdict.

PASS (default = the infrastructure + controls gate, what a regression gate must protect):
  * the POWER control (O2 vs O1) is DISTINGUISHABLE over the binding representation (AUC ≥ auc_high)
    — the C2ST has power, so a ≈0.5 on a real pair is genuine indistinguishability, not a dead test;
  * for each appearance pair (O4↔O2), the VISION disambiguator IS distinguishable (AUC ≥ auc_high)
    if CLIP is available — the signal proprioception lacks is present in vision.
The per-pair proprioceptive indistinguishability is REPORTED (per T × classifier × representation,
with the leaking-channel readout) but NOT gated by default: the central finding is that O4↔O2 on the
unshaped operators is EXPECTED to be distinguishable (O2 const drag vs O4 penetration ramp), and
reaching indistinguishability is the calibration loop's job (regime tightening / the #49 O4 knob).

--require-indistinguishable escalates: additionally require every pair indistinguishable over
the binding representation at ALL T for ALL decisive classifiers — the calibrated-claim certificate.

Run:  python scripts/isaac_e1_check.py --headless
      python scripts/isaac_e1_check.py --headless --require-indistinguishable   # certify the claim
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _lane_list(traces: list, feature_set: str, arm_steps: int, lane_fn) -> tuple[list, list]:
    """Per-lane (S,F) arrays + counts [dropped, kept], dropping fallen/empty-segment lanes (R8).
    (c2st_sweep re-derives lane ids by list index, so we return the segments, not lane ids.)"""
    out, dropped = [], 0
    for tr in traces:
        if tr["fell"]:
            dropped += 1
            continue
        seg = lane_fn(tr, feature_set, arm_steps)
        if seg is None or seg.shape[0] == 0:
            dropped += 1
            continue
        out.append(seg)
    return out, [dropped, len(out)]


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 proprioceptive-indistinguishability C2ST gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--config", default="eval/e1_c2st.yaml")
    ap.add_argument("--out", default="outputs/eval/e1")
    ap.add_argument("--seeds-train", type=int, default=0, help="0 = config seeds.train")
    ap.add_argument("--seeds-test", type=int, default=0, help="0 = config seeds.test")
    ap.add_argument("--require-indistinguishable", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smaller battery (CI/smoke)")
    ap.add_argument("--arm-steps", type=int, default=-1, help="override regime.arm_steps (-1=cfg)")
    ap.add_argument("--o4-shaping", action="store_true",
                    help="enable the #49 O4 peel-plateau knob (calibration STEP 2)")
    ap.add_argument("--o4-cap", type=float, default=0.0, help="force_cap_n for --o4-shaping")
    ap.add_argument("--o4-offset", type=float, default=14.0, help="force_offset_n for --o4-shaping")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841  (keeps the sim app alive for the process)

    from kino_vla.eval.c2st import c2st_sweep, c2st_vision
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import REPO_ROOT, load_config

    # collection utilities (real-stack) live in the collect script
    sys.path.insert(0, str((REPO_ROOT / "scripts").resolve()))
    from isaac_e1_collect import collect_units, e1_plan, feature_names, git_commit, lane_trace

    overrides = {}
    if args.o4_shaping:  # STEP 2: make O4 forward grip a constant drag = O2's (matched preset)
        overrides = {"o4_shaping.enabled": True, "o4_shaping.force_cap_n": float(args.o4_cap),
                     "o4_shaping.force_offset_n": float(args.o4_offset)}
    cfg = load_config(args.config, overrides=overrides)
    d = cfg.to_dict()
    seeds_tr = args.seeds_train or int(d["seeds"]["train"])
    seeds_te = args.seeds_test or int(d["seeds"]["test"])
    seed_base_te = int(d["seeds"]["seed_base_test"])
    steps = int(d["regime"]["steps"])
    cruise = float(d["regime"]["cruise_mps"])
    arm = int(d["regime"]["arm_steps_after_entry"]) if args.arm_steps < 0 else int(args.arm_steps)
    window_lens = list(d["window"]["window_lens"])
    feature_sets = list(d["c2st"]["feature_sets"])
    classifiers = list(d["c2st"]["classifiers"])
    decisive = list(d["c2st"]["decisive_classifiers"])
    binding = str(d["c2st"]["binding_feature_set"])
    auc_high = float(d["c2st"]["auc_high"])
    alpha = float(d["c2st"]["alpha"])
    n_boot, n_perm = int(d["c2st"]["n_boot"]), int(d["c2st"]["n_perm"])
    deployed_T = int(window_lens[0])
    if args.quick:
        window_lens = [deployed_T, window_lens[-1]]
        feature_sets = sorted({"feat12", binding})
        classifiers = sorted({"logreg", *decisive})
        n_boot, n_perm = 200, 100

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    units, pairs, control = e1_plan(cfg)
    print(f"[e1] units={list(units)} pairs={[p['name'] for p in pairs]} "
          f"train={seeds_tr} test={seeds_te} binding={binding} steps={steps}", flush=True)

    train = collect_units(backend, units, seeds=seeds_tr, seed_base=0, steps=steps,
                          cruise=cruise, dt=dt)
    test = collect_units(backend, units, seeds=seeds_te, seed_base=seed_base_te, steps=steps,
                         cruise=cruise, dt=dt, lane_offset=10_000)

    comparisons = [*pairs, {"name": control["name"], "key_a": control["key_a"],
                            "key_b": control["key_b"], "disambiguator": {"kind": "none"},
                            "is_control": True}]
    common = dict(n_boot=n_boot, n_perm=n_perm, alpha=alpha, stride=int(d["c2st"]["stride"]))
    report: dict = {"meta": {"commit": git_commit(), "config": args.config, "binding": binding,
                             "seeds_train": seeds_tr, "seeds_test": seeds_te, "steps": steps,
                             "cruise": cruise, "arm_steps": arm, "auc_high": auc_high,
                             "o4_shaping": d["o4_shaping"]},
                    "comparisons": {}}

    for comp in comparisons:
        name, ka, kb = comp["name"], comp["key_a"], comp["key_b"]
        is_control = comp.get("is_control", False)
        cinfo: dict = {"key_a": ka, "key_b": kb, "is_control": is_control, "grid": {}, "counts": {}}
        for fset in feature_sets:
            tr_a, ca = _lane_list(train[ka], fset, arm, lane_trace)
            tr_b, cb = _lane_list(train[kb], fset, arm, lane_trace)
            te_a, da = _lane_list(test[ka], fset, arm, lane_trace)
            te_b, db = _lane_list(test[kb], fset, arm, lane_trace)
            cinfo["counts"][fset] = {"train_a": ca, "train_b": cb, "test_a": da, "test_b": db}
            if not (tr_a and tr_b and te_a and te_b):
                print(f"[e1] {name}/{fset}: insufficient lanes, skipping", flush=True)
                continue
            res = c2st_sweep(
                tr_a, tr_b, te_a, te_b,
                window_lens=window_lens, classifiers=classifiers,
                feature_names=feature_names(fset), feature_set=fset,
                importance=(fset == binding), seed=0, **common,
            )
            for (t, clf), r in res.items():
                cinfo["grid"][f"{fset}|T{t}|{clf}"] = r.to_dict()
                tag = "CTRL" if is_control else "pair"
                print(f"[e1] {tag} {name:>6} {fset:>9} T={t:<3} {clf:>6}: "
                      f"AUC={r.auc:.3f} CI=({r.auc_ci[0]:.2f},{r.auc_ci[1]:.2f}) "
                      f"p={r.perm_p:.3f} indist={r.indistinguishable}", flush=True)
                if fset == binding and clf == "cnn1d" and r.feature_importance:
                    top = sorted(r.feature_importance.items(), key=lambda kv: -kv[1])[:3]
                    print(f"        leak(top3 AUC-drop): {[(k, round(v, 3)) for k, v in top]}",
                          flush=True)
        report["comparisons"][name] = cinfo

    # --- Vision / disambiguator control (real CLIP) -------------------------------------
    vision_report: dict = {}
    if bool(d["vision"]["enabled"]):
        try:
            from kino_vla.map.clip_appearance import ClipAppearanceEncoder
            from kino_vla.map.clip_segmentation import APPEARANCE_TO_MATERIAL, material_texture

            enc = ClipAppearanceEncoder()
            n_s = int(d["vision"]["n_samples"])
            for comp in pairs:
                dis = comp["disambiguator"]
                if dis.get("kind") != "appearance":
                    vision_report[comp["name"]] = {"kind": dis.get("kind"), "note": "characterize"}
                    continue
                mat_a = APPEARANCE_TO_MATERIAL[dis["a"]]
                mat_b = APPEARANCE_TO_MATERIAL[dis["b"]]
                emb_a = enc.embed_batch([material_texture(mat_a, seed=s) for s in range(n_s)])
                emb_b = enc.embed_batch([material_texture(mat_b, seed=s) for s in range(n_s)])
                rv = c2st_vision(emb_a, emb_b, n_boot=n_boot, n_perm=n_perm, alpha=alpha, seed=0)
                vision_report[comp["name"]] = {"kind": "appearance", **rv.to_dict()}
                print(f"[e1] VISION {comp['name']}: {mat_a} vs {mat_b} AUC={rv.auc:.3f} "
                      f"distinguishable={not rv.indistinguishable}", flush=True)
        except Exception as e:  # noqa: BLE001
            vision_report["error"] = repr(e)
            print(f"[e1] vision control unavailable: {e!r}", flush=True)
    report["vision"] = vision_report

    # --- Verdict ------------------------------------------------------------------------
    def binding_results(name: str) -> list:
        g = report["comparisons"][name]["grid"]
        return [g[k] for k in g if k.startswith(f"{binding}|") and k.split("|")[-1] in decisive]

    ctrl = report["comparisons"][control["name"]]["grid"]
    ctrl_binding = [ctrl[k] for k in ctrl
                    if k.startswith(f"{binding}|T{deployed_T}|") and k.split("|")[-1] in decisive]
    power_ok = bool(ctrl_binding) and any((not r["indistinguishable"]) and r["auc"] >= auc_high
                                          for r in ctrl_binding)
    # An appearance pair declared in config MUST produce a distinguishable vision control when
    # vision is enabled — a missing/errored/weak control fails the gate (don't silently pass).
    vision_enabled = bool(d["vision"]["enabled"])
    vision_ok = True
    for comp in pairs:
        if comp["disambiguator"].get("kind") != "appearance":
            continue  # 'none' pairs: vision not claimed to disambiguate (finding #37, not a gate)
        v = vision_report.get(comp["name"], {})
        ok = ("auc" in v) and (v["auc"] >= auc_high)
        if vision_enabled and not ok:
            vision_ok = False
    claim_ok = all(
        bool(binding_results(p["name"]))
        and all(r["indistinguishable"] for r in binding_results(p["name"]))
        for p in pairs
    )
    passed = power_ok and vision_ok and (claim_ok if args.require_indistinguishable else True)
    report["verdict"] = {
        "power_control_distinguishable": power_ok,
        "vision_distinguishable": vision_ok,
        "all_pairs_indistinguishable_binding": claim_ok,
        "require_indistinguishable": bool(args.require_indistinguishable),
        "PASS": bool(passed),
    }

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e1_results.json").write_text(json.dumps(report, indent=2))
    (out_dir / "e1_results.md").write_text(_markdown_card(report, binding, deployed_T, decisive))
    print("\n" + _markdown_card(report, binding, deployed_T, decisive), flush=True)
    print(("PASS" if passed else "FAIL") + ": E1 proprioceptive indistinguishability "
          f"(power={power_ok} vision={vision_ok} claim={claim_ok})", flush=True)
    sys.stdout.flush()
    os._exit(0 if passed else 1)


def _markdown_card(report: dict, binding: str, deployed_T: int, decisive: list) -> str:
    m = report["meta"]
    lines = [
        "# E1 — Proprioceptive Indistinguishability (C2ST)",
        "",
        f"commit `{m['commit']}` · config `{m['config']}` · binding **{binding}** · "
        f"train/test {m['seeds_train']}/{m['seeds_test']} seeds · cruise {m['cruise']} m/s · "
        f"arm {m['arm_steps']} · auc_high {m['auc_high']} · "
        f"O4-shaping {m.get('o4_shaping', {})}",
        "",
        "Per comparison, the BINDING representation (raw obs+torque) — `indist=True` means the "
        "discriminator CANNOT separate the pair (proprioception insufficient):",
        "",
        "| comparison | repr | T | clf | AUC | 95% CI | perm p | indist |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in report["comparisons"].items():
        for k, r in c["grid"].items():
            fset, t, clf = k.split("|")
            if fset != binding:
                continue
            tag = name + ("*" if c.get("is_control") else "")
            ci = f"({r['auc_ci'][0]:.2f},{r['auc_ci'][1]:.2f})"
            lines.append(f"| {tag} | {fset} | {t[1:]} | {clf} | {r['auc']:.3f} | {ci} | "
                         f"{r['perm_p']:.3f} | {r['indistinguishable']} |")
    lines += ["", "(* = power control; it MUST be distinguishable.)", ""]
    # secondary representations summary (deployed-12 etc.) at the deployed T
    lines += ["Other representations @T=" + str(deployed_T) + " (decisive classifiers):", "",
              "| comparison | repr | clf | AUC | indist |", "|---|---|---|---|---|"]
    for name, c in report["comparisons"].items():
        for k, r in c["grid"].items():
            fset, t, clf = k.split("|")
            if fset == binding or t != f"T{deployed_T}" or clf not in decisive:
                continue
            lines.append(f"| {name} | {fset} | {clf} | {r['auc']:.3f} | {r['indistinguishable']} |")
    if report.get("vision"):
        lines += ["", "## Disambiguator control (real CLIP)", "",
                  "| pair | kind | AUC | distinguishable |", "|---|---|---|---|"]
        for name, v in report["vision"].items():
            if name == "error":
                lines.append(f"| _ | error | {v} | - |")
            elif v.get("kind") == "appearance" and "auc" in v:
                sep = not v["indistinguishable"]
                lines.append(f"| {name} | appearance | {v['auc']:.3f} | {sep} |")
            else:
                lines.append(f"| {name} | {v.get('kind')} | - | characterize (finding #37) |")
    vd = report["verdict"]
    lines += [
        "", "## Verdict",
        f"- power control distinguishable: **{vd['power_control_distinguishable']}**",
        f"- vision distinguishable (appearance pairs): **{vd['vision_distinguishable']}**",
        f"- all pairs indistinguishable over binding (claim): "
        f"**{vd['all_pairs_indistinguishable_binding']}**",
        f"- **{'PASS' if vd['PASS'] else 'FAIL'}** "
        f"(require_indistinguishable={vd['require_indistinguishable']})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
