#!/usr/bin/env python
"""E2 gate — the proprioceptive-ceiling A/B dichotomy on the REAL Go2.

Compares B1 (strongest pure-proprio: the RMA-fidelity attributor → canonical recovery), B2 (the
cause-blind rule-FSM), and B5 (the vision+proprio agent) across:

  * Suite-Sem (matched construction): on the #49-matched O4/O2, B1 is chance attribution + recovery
    FAILURE (wrong opposite recovery), while B5 attributes (vision) and succeeds.
  * Suite-Cal (A-class): on the six A-class operators where low-level adaptation suffices, B1 ≈ B5
    (parity ⇒ B1 is a FAIR opponent, not a strawman).

Both Suite-Sem metrics come from the SAME closed-loop rollouts (RolloutResult.first_attribution +
.success), so the head-to-head is on identical scenes (same seeds across policies).

The closed-loop recovery-failure half needs an O4 matched at the ~0.5 s detection window AND that
traps push-through. The G1/G2 sweep finds the trapping-matched preset (G1: detection C2ST≈0.5 vs
matched O2; G2: forced-Backstep reaches but forced-push-through does not). If none passes, fall back
to the attribution-consequence (B1's chance attribution → wrong non-escaping primitive), reported
honestly. Enabling a trap preset is flagged as deviation #50.

Run:  python scripts/isaac_e2_check.py --headless
      python scripts/isaac_e2_check.py --headless --quick   # reduced battery (sim gate)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _drive_detection(backend, scenario, *, cruise, n_steps, seeds, monitor_features, rect):
    """Straight-cruise the operator; return per-seed in-region 12-dim feature traces (G1 C2ST)."""
    from kino_vla.sim.operators import OperatorStack

    traces = []
    for s in seeds:
        backend._start_pos = np.array([0.0, 0.0])
        backend._start_heading = 0.0
        obs = backend.reset(s)
        OperatorStack([scenario.operator]).on_reset(backend)
        rows = []
        for _ in range(n_steps):
            if rect.contains(obs.pos):
                rows.append(monitor_features(obs))
            obs = backend.step(np.array([cruise, 0.0, 0.0]))
            if obs.fallen:
                break
        if len(rows) >= 1:
            traces.append(np.asarray(rows, dtype=np.float64))
    return traces


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 proprioceptive-ceiling A/B-dichotomy gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--config", default="eval/e2.yaml")
    ap.add_argument("--out", default="outputs/eval/e2")
    ap.add_argument("--repeats", type=int, default=0, help="0 = config run.repeats")
    ap.add_argument("--quick", action="store_true", help="reduced battery (sim gate)")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841 (keeps the sim app alive)

    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.c2st import c2st_holdout, windows_from_lanes
    from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy
    from kino_vla.eval.suite_sem import MajorityBaselinePolicy
    from kino_vla.monitor.hazard_lab import monitor_features
    from kino_vla.monitor.learned_monitor import load_deployed_monitor
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla import scenarios as S
    from kino_vla.vla.rollout import run_closed_loop

    sys.path.insert(0, str((REPO_ROOT / "scripts").resolve()))
    from isaac_e1_collect import git_commit

    cfg = load_config(args.config)
    d = cfg.to_dict()
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    repeats = args.repeats or int(d["run"]["repeats"])
    seed_base = int(d["run"]["seed_base"])
    cruise = 0.6
    if args.quick:
        repeats = 2

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    monitor = load_deployed_monitor(backend.dt)  # shared triggering monitor (run_episode resets it)
    # B5 reads the recorder's PROCEDURAL appearance render — B5's M6/M7 training+eval distribution
    # (the 0.974 attribution number); real-RTX + real-CLIP vision separation is established by E1
    # (AUC=1.0). Feeding real RTX pixels would be OOD for B5's procedural-trained attribution head.

    # --- B1: the trained RMA-fidelity proprio attributor → canonical recovery ---------------
    b1 = ProprioBaselinePolicy.from_deployed(
        pcfg, tax, model_path=str(d["b1"]["model_path"]), device="cpu"
    )
    # --- B5: the vision+proprio agent ---------------------------------------------------------
    vcfg = load_config(str(d["b5"]["config"]), {"route": str(d["b5"]["route"])})
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy

    print("[e2] loading B5 agent (Qwen3-VL-4B + LoRA + Kino-Projector) …", flush=True)
    model = KinoVLA.from_pretrained(vcfg, device="cuda", adapter_dir=str(d["b5"]["adapter_dir"]))
    model.eval()
    torch.set_grad_enabled(False)
    b5 = ModelVlaPolicy(
        model, pcfg, tax, route=str(vcfg.get("route", "latent")),
        n_images=int(vcfg.data.get("n_images", 1)), temperature=float(d["b5"]["temperature"]),
        proprio_detail=str(vcfg.data.get("proprio_detail", "binned")),
    )

    def run_one(scenario, *, policy=None, fsm=False, seed=0, live_camera=None):
        backend._start_pos = np.asarray(scenario.start_xy, dtype=np.float64)
        backend._start_heading = float(scenario.start_heading)
        return run_closed_loop(
            backend, scenario, policy, monitor_cfg="monitor/rule_v0_isaac.yaml",
            fsm_cfg="recovery/fsm_isaac.yaml", seed=seed, fsm_baseline=fsm, monitor=monitor,
            live_camera=live_camera,
        )

    report = {"meta": {"commit": git_commit(), "config": args.config, "repeats": repeats,
                       "b1_model": d["b1"]["model_path"], "b5_adapter": d["b5"]["adapter_dir"]}}

    # ============================ G1/G2 calibration of the matched-O4 trap preset ============
    cal = d["suite_sem"]["calibration"]
    o2m = S.o2_compliance_matched(0.0)
    rect = o2m.scene_region.rect
    g1_seeds = [seed_base + i for i in range(2 if args.quick else 3)]
    o2_tr = _drive_detection(backend, o2m, cruise=cruise, n_steps=160, seeds=g1_seeds,
                             monitor_features=monitor_features, rect=rect)
    o2_win, o2_lane = windows_from_lanes(o2_tr, cal["g1_detection_steps"])
    preset = {"force_cap_n": float(d["suite_sem"]["attribution_preset"]["force_cap_n"]),
              "force_offset_n": float(d["suite_sem"]["attribution_preset"]["force_offset_n"]),
              "peel_factor": float(d["suite_sem"]["attribution_preset"]["peel_factor"])}
    cal_log = {"candidates": [], "adopted": None, "fallback": False}
    if bool(cal.get("enabled", True)) and o2_win.shape[0] > 0:
        cands = [(float(k), float(c)) for k in cal["k_grid"] for c in cal["force_cap_grid"]]
        for k, capn in cands:
            o4c = S.o4_tether_matched(0.0, k=k, force_cap_n=capn,
                                      force_offset_n=float(cal["force_offset_n"]),
                                      peel_factor=float(cal["peel_factor"]))
            o4_tr = _drive_detection(backend, o4c, cruise=cruise, n_steps=160, seeds=g1_seeds,
                                     monitor_features=monitor_features, rect=rect)
            o4_win, o4_lane = windows_from_lanes(o4_tr, cal["g1_detection_steps"])
            if o4_win.shape[0] == 0:
                continue
            res = c2st_holdout(o4_win, o2_win, o4_lane, o2_lane, test_frac=0.5,
                               classifier="logreg", n_boot=200, n_perm=120, importance=False)
            g1_pass = res.auc <= float(cal["g1_auc_high"])
            entry = {"k": k, "force_cap_n": capn, "detection_auc": round(res.auc, 3),
                     "g1_pass": bool(g1_pass), "g2_pass": None}
            if g1_pass:  # G2: forced-Backstep reaches, forced-push-through does not
                bs = run_one(o4c, policy=MajorityBaselinePolicy(pcfg, "adhesion", "Backstep"),
                             seed=seed_base)
                pt = run_one(o4c, policy=MajorityBaselinePolicy(pcfg, "compliant_terrain",
                                                                "Switch_Gait"), seed=seed_base)
                g2_pass = bs.success and not pt.success
                entry["g2_pass"] = bool(g2_pass)
                entry["g2_backstep_success"] = bool(bs.success)
                entry["g2_pushthrough_success"] = bool(pt.success)
                cal_log["candidates"].append(entry)
                print(f"[e2.cal] k={k} cap={capn} detAUC={res.auc:.3f} G1={g1_pass} "
                      f"G2={g2_pass} (Backstep={bs.success}, push={pt.success})", flush=True)
                if g2_pass:
                    preset = {"k": k, "force_cap_n": capn,
                              "force_offset_n": float(cal["force_offset_n"]),
                              "peel_factor": float(cal["peel_factor"])}
                    cal_log["adopted"] = entry
                    break
            else:
                cal_log["candidates"].append(entry)
                print(f"[e2.cal] k={k} cap={capn} detAUC={res.auc:.3f} G1=False", flush=True)
    if cal_log["adopted"] is None:
        cal_log["fallback"] = True
        print("[e2.cal] no trapping-matched preset passed G1+G2 → FALLBACK to the attribution "
              "preset; recovery failure shown via the attribution→wrong-primitive consequence.",
              flush=True)
    report["calibration"] = {**cal_log, "preset": preset}

    # ============================ Suite-Sem (matched) closed-loop ============================
    sem_scns = [
        ("adhesion", S.o4_tether_matched(0.0, **{k: v for k, v in preset.items()})),
        ("compliant_terrain", S.o2_compliance_matched(0.0)),
    ]
    policies = [("B1", dict(policy=b1)), ("B2_fsm", dict(fsm=True)), ("B5", dict(policy=b5))]
    sem = {name: {"attr_correct": 0, "n": 0, "success": 0, "rows": []} for name, _ in policies}
    for truth, scn in sem_scns:
        for name, kw in policies:
            for rep in range(repeats):
                r = run_one(scn, seed=seed_base + rep, **kw)
                a_ok = r.first_attribution == truth
                sem[name]["attr_correct"] += int(a_ok)
                sem[name]["n"] += 1
                sem[name]["success"] += int(r.success)
                sem[name]["rows"].append({"scn": scn.name, "truth": truth, "rep": rep,
                                          "attr": r.first_attribution, "prim": r.first_primitive,
                                          "success": r.success})
                print(f"[e2.sem] {name:6} {scn.name:22} rep{rep}: attr={r.first_attribution} "
                      f"(truth {truth}) ok={a_ok} success={r.success}", flush=True)
    for name in sem:
        n = max(1, sem[name]["n"])
        sem[name]["attribution_acc"] = sem[name]["attr_correct"] / n
        sem[name]["recovery_success"] = sem[name]["success"] / n
    report["suite_sem"] = sem

    # ============================ Suite-Cal (A-class) closed-loop ============================
    cal_scns = S.suite_cal_scenarios(0.0)
    suite_cal = {name: {"success": 0, "n": 0, "rows": []} for name, _ in policies}
    for scn in cal_scns:
        for name, kw in policies:
            for rep in range(repeats):
                r = run_one(scn, seed=seed_base + rep, **kw)
                suite_cal[name]["success"] += int(r.success)
                suite_cal[name]["n"] += 1
                suite_cal[name]["rows"].append({"scn": scn.name, "rep": rep, "success": r.success})
                print(f"[e2.cal2] {name:6} {scn.name:20} rep{rep}: success={r.success}", flush=True)
    for name in suite_cal:
        n = max(1, suite_cal[name]["n"])
        suite_cal[name]["success_rate"] = suite_cal[name]["success"] / n
    report["suite_cal"] = suite_cal

    # ============================ Verdict + card ============================
    acc = d["accept"]
    sem_gap = sem["B5"]["attribution_acc"] - sem["B1"]["attribution_acc"]
    cal_parity = abs(suite_cal["B1"]["success_rate"] - suite_cal["B5"]["success_rate"])
    passed = (sem_gap >= float(acc["sem_attribution_gap_min"])
              and cal_parity <= float(acc["cal_parity_tol"]))
    report["verdict"] = {
        "sem_attribution_gap": round(sem_gap, 3),
        "sem_attribution_gap_min": float(acc["sem_attribution_gap_min"]),
        "cal_parity_abs_diff": round(cal_parity, 3),
        "cal_parity_tol": float(acc["cal_parity_tol"]),
        "trap_preset_found": cal_log["adopted"] is not None,
        "PASS": bool(passed),
    }

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2_results.json").write_text(json.dumps(report, indent=2))
    (out_dir / "e2_results.md").write_text(_card(report))
    print("\n" + _card(report), flush=True)
    print(("PASS" if passed else "FAIL") + ": E2 proprioceptive-ceiling dichotomy "
          f"(Sem gap {sem_gap:.3f}, Cal parity Δ {cal_parity:.3f})", flush=True)
    sys.stdout.flush()
    os._exit(0 if passed else 1)


def _card(r: dict) -> str:
    m = r["meta"]
    sem, cal = r["suite_sem"], r["suite_cal"]
    lines = [
        "# E2 — Proprioceptive-Ceiling A/B Dichotomy",
        "",
        f"commit `{m['commit']}` · config `{m['config']}` · repeats {m['repeats']} · "
        f"B1 `{m['b1_model']}` · B5 `{m['b5_adapter']}`",
        "",
        f"Trapping-matched O4 preset: **{'found' if r['calibration']['adopted'] else 'FALLBACK'}** "
        f"({r['calibration']['preset']})",
        "",
        "| Policy | Suite-Sem attribution acc | Suite-Sem recovery success | Suite-Cal success |",
        "|---|---|---|---|",
    ]
    for name in ("B1", "B2_fsm", "B5"):
        lines.append(
            f"| {name} | {sem[name]['attribution_acc']:.3f} | {sem[name]['recovery_success']:.3f} "
            f"| {cal[name]['success_rate']:.3f} |"
        )
    v = r["verdict"]
    lines += [
        "",
        f"- Suite-Sem attribution gap (B5−B1): **{v['sem_attribution_gap']}** "
        f"(min {v['sem_attribution_gap_min']})",
        f"- Suite-Cal parity |B1−B5|: **{v['cal_parity_abs_diff']}** (tol {v['cal_parity_tol']})",
        f"- trapping-matched preset found: **{v['trap_preset_found']}**",
        f"- **{'PASS' if v['PASS'] else 'FAIL'}**",
        "",
        "B1 chance attribution on the matched pair is E1-backed (the C2ST optimal-discriminator"
        " ceiling at T=25/50/100). Suite-Cal parity proves B1 is a fair opponent; the Suite-Sem gap"
        " is therefore the real cost of proprioceptive blindness on the semantic class.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
