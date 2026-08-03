#!/usr/bin/env python
"""A4.4 + A4.5 — closed-loop agent validation of the ERS composition (Paper-A §4 A4.4/A4.5, C3).

A4.3 composes each agent's OPEN-LOOP label distribution with M to predict expected cost (ERS) +
regret. A4.4 validates that composition: run B5-conflict-bi TRULY closed-loop (the agent decides
the recovery from its snapshot, oracle trigger A0.2-primary, deterministic backend A0.1) on 3
scenarios × N seeds, and check the REALIZED success ≈ the composed ERS within CI. If they agree,
the open-loop⇒consequence bridge is sound (attribution error really does translate to physical
cost the matrix predicts). A4.5 re-runs the subset with the hardened realistic trigger
(``monitor_abaware`` @ thr 0.6 / debounce 25 / arm 2.5) replacing the oracle: expected identical on
the O8/T3 cell (monitor silent or a single clean fire), the O2 residual-ambiguity fires reported
as-is (the A1.5 signature, not noise).

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES KINOVLA_MODEL_ID=… \\
        ~/miniconda3/envs/kinovla/bin/python scripts/a4_4_closedloop.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from kino_vla.eval.a4_scenarios import build_rollout_scenario

SCENARIOS = ("matched_O4_twophase", "matched_O2", "O8_invisible")  # T2 adhesion / T1 mud / T3 wall
SEED_BASE = 800


def main() -> int:
    ap = argparse.ArgumentParser(description="A4.4/A4.5 closed-loop agent ERS validation")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument(
        "--adapter",
        default="outputs/eval/a3/b5_conflict_bi/adapter_best",
        help="B5-conflict-bi LoRA adapter (the bidirectional agent)",
    )
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--lane-y", type=float, default=4.0)
    ap.add_argument("--arm", type=float, default=0.2)
    ap.add_argument(
        "--trigger",
        default="oracle",
        choices=["oracle", "realistic"],
        help="A4.4=oracle (primary); A4.5=realistic (monitor_abaware @ deb25/arm2.5)",
    )
    ap.add_argument("--out", default="outputs/eval/a4")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.oracle_trigger import OracleTrigger
    from kino_vla.eval.registry import load_registry
    from kino_vla.monitor.learned_monitor import load_deployed_monitor
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy
    from kino_vla.vla.rollout import run_closed_loop

    reg = load_registry()
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    backend = IsaacPolicyBackend(
        load_config("sim/go2_skeleton.yaml"), np.array([0.0, args.lane_y]), 0.0
    )
    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    print(f"[a4.4] trigger={args.trigger} scenarios={names} seeds={args.seeds}", flush=True)

    # the realistic trigger (A4.5): the hardened monitor_abaware operating point
    realistic_mon = None
    if args.trigger == "realistic":
        realistic_mon = load_deployed_monitor(
            backend.dt,
            model_path="outputs/monitor_learned/monitor_abaware",
            debounce_steps=25,
            arm_s=2.5,
        )

    print(f"[a4.4] loading B5-conflict-bi ({args.adapter}) …", flush=True)
    vcfg = load_config("vla/sft.yaml")
    model = KinoVLA.from_pretrained(vcfg, device="cuda", adapter_dir=args.adapter)
    model.eval()
    torch.set_grad_enabled(False)
    route = str(vcfg.get("route", "latent"))
    n_images = int(vcfg.data.get("n_images", 1))
    policy = ModelVlaPolicy(
        model,
        pcfg,
        tax,
        route=route,
        n_images=n_images,
        temperature=0.0,
        proprio_detail=str(vcfg.data.get("proprio_detail", "binned")),
    )

    rows: list[dict] = []
    for sname in names:
        spec = reg[sname]
        scn = build_rollout_scenario(spec, args.lane_y)
        n_succ = 0
        for si in range(args.seeds):
            seed = SEED_BASE + si
            backend._start_pos = np.asarray(scn.start_xy, dtype=np.float64)
            backend._start_heading = 0.0
            if args.trigger == "oracle":
                mon = OracleTrigger.for_scenario(scn, dt=backend.dt, arm_delay_s=args.arm)
            else:
                mon = realistic_mon
                if hasattr(mon, "reset"):
                    mon.reset()
            r = run_closed_loop(
                backend,
                scn,
                policy,
                monitor_cfg="monitor/rule_v0_isaac.yaml",
                fsm_cfg="recovery/fsm_isaac.yaml",
                seed=seed,
                monitor=mon,
                deep_reset=True,
            )
            n_succ += int(r.success)
            rows.append(
                {
                    "scenario": sname,
                    "seed": seed,
                    "trigger": args.trigger,
                    "success": bool(r.success),
                    "reached": bool(r.reached),
                    "fell": bool(r.fell),
                    "first_attribution": r.first_attribution,
                    "first_primitive": r.first_primitive,
                    "final_dist_m": round(r.final_dist_m, 2),
                }
            )
            print(
                f"[a4.4] {sname:20} s{seed} succ={int(r.success)} prim={r.first_primitive} "
                f"attr={r.first_attribution} dist={r.final_dist_m:.1f}",
                flush=True,
            )
        rate = n_succ / max(1, args.seeds)
        print(
            f"[a4.4] >>> {sname}: realized success {rate:.2f} ({n_succ}/{args.seeds})", flush=True
        )

    out = REPO_ROOT / args.out
    tag = "a4_4_closedloop" if args.trigger == "oracle" else "a4_5_realistic"
    (out / f"{tag}.json").write_text(json.dumps(rows, indent=2))
    # summarize realized vs composed ERS (if a4_results.json exists)
    res_path = out / "a4_results.json"
    print(f"\n=== A4.4/{args.trigger} realized success vs composed ERS ===")
    if res_path.exists():
        res = json.loads(res_path.read_text())
        comp = res.get("composition", {}).get("B5_conflict_bi", {})
        for sname in names:
            real = sum(1 for r in rows if r["scenario"] == sname and r["success"]) / max(
                1, args.seeds
            )
            argmin = comp.get("argmin_label", {}).get(sname, "?")
            regret = comp.get("regret_per_scenario", {}).get(sname, "?")
            print(
                f"  {sname:20} realized={real:.2f}  agent-label={argmin}  composed-regret={regret}"
            )
    else:
        print("  (a4_results.json not found — run scripts/a4_analyze.py first)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
