#!/usr/bin/env python
"""VLA Recovery Planner CLOSED LOOP on the real Go2 (spec §1/§11 Stage 2) — the M7 exit-3 gate.

The spec runs the recovery loop on Isaac Lab (§1 layer 03-05, §11 "SFT 模型置于 Isaac Lab 闭环").
This drives the physically-simulated Go2 toward a goal; when the Kino-Monitor fires, the trained
KinoVLA attributes the cause from the REAL RGB + proprioception, picks one §5 primitive, the CBF
shield + Primitive Compiler execute it, and we measure the physical outcome (reached vs fell /
dead-loop). Each Suite-Sem scenario runs in its own lateral lane (Isaac is one-app/process, #21a),
reusing a single Isaac app.

Modes:
  --mode eval   : one rollout per scenario at temperature 0 → the success rate (compare SFT vs DPO).
  --mode pairs  : sample the configured temperatures per scenario → save Embodied-DPO preference
                  pairs (physical-outcome Chosen/Rejected, spec §11) + the SFT success rate.

Run:  python scripts/isaac_vla_rollout.py --headless --adapter outputs/vla/sft_latent/adapter_best \
          --mode pairs --out outputs/vla/isaac_rollout
"""

from __future__ import annotations

import argparse
import json
import os
import threading


def main() -> int:
    parser = argparse.ArgumentParser(description="VLA closed loop on the real Go2 (M7 exit-3)")
    parser.add_argument("--adapter", default=None, help="VLA adapter dir (SFT or DPO); None=stub")
    parser.add_argument("--policy", choices=["model", "stub", "fsm"], default="model")
    parser.add_argument("--config", default="vla/dpo.yaml")
    parser.add_argument("--route", default=None, choices=["latent", "text"], help="route arm")
    parser.add_argument(
        "--proprio-detail",
        default="binned",
        choices=["binned", "scalar", "none"],
        help="text route: 'none' = B3 vision-only",
    )
    parser.add_argument("--mode", choices=["eval", "pairs"], default="pairs")
    parser.add_argument("--temperatures", default="0.0,0.7,1.0")
    parser.add_argument("--seeds", type=int, default=1, help="eval: rollouts/op (own lane)")
    parser.add_argument(
        "--ops",
        default="o1,o2,o3,o4,o5",
        help="comma-separated operators to run (default = the probe-dataset-covered B-class set)",
    )
    parser.add_argument(
        "--fsm-cfg",
        default="recovery/fsm_isaac.yaml",
        help="recovery/nav config (use the no-probe variant for the Gap-3 probe ablation)",
    )
    parser.add_argument("--lane-spacing", type=float, default=6.0)
    parser.add_argument("--out", default="outputs/vla/isaac_rollout")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import numpy as np

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config
    from kino_vla.vla import scenarios as S
    from kino_vla.vla.dpo import build_preference_pairs, pair_stats, save_pairs
    from kino_vla.vla.planner import StubVlaPolicy
    from kino_vla.vla.rollout import run_closed_loop

    cfg = load_config(args.config, {"route": args.route} if args.route else None)
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    temps = [float(t) for t in args.temperatures.split(",")]
    os.makedirs(args.out, exist_ok=True)

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)

    fsm_baseline = args.policy == "fsm"
    model = None
    if args.policy == "model":
        import torch

        from kino_vla.vla.model import KinoVLA

        model = KinoVLA.from_pretrained(cfg, device="cuda", adapter_dir=args.adapter)
        model.eval()
        torch.set_grad_enabled(False)

    def make_policy(temp: float):
        if args.policy == "fsm":
            return None  # the cause-blind FSM baseline (run_closed_loop builds it via fsm_baseline)
        if args.policy == "stub":
            return StubVlaPolicy(pcfg, tax, error_mode="sibling" if temp > 0 else None)
        from kino_vla.vla.planner import ModelVlaPolicy

        return ModelVlaPolicy(
            model,
            pcfg,
            tax,
            route=str(cfg.get("route", "latent")),
            n_images=int(cfg.data.get("n_images", 1)),
            temperature=temp,
            proprio_detail=args.proprio_detail,
        )

    # Lateral lanes: each (scenario, seed/temp) at a distinct y so the persistent Isaac prims don't
    # overlap (Isaac is one-app/process, #21a). The full B-class set drives the §2.5 dichotomy.
    op_builders = {
        "o1": S.o1_ice,
        "o2": S.o2_compliance,
        "o3": S.o3_collapse,
        "o4": S.o4_tether,
        "o5": S.o5_payload,
        "o8": S.o8_invisible,
        "o10": S.o10_effort_decay,
    }
    builders = [op_builders[k.strip()] for k in args.ops.split(",") if k.strip() in op_builders]
    rows: list[dict] = []
    all_pairs = []
    lane = 0
    # eval: success rate over `--seeds` rollouts/op @temp0 (the dichotomy table). pairs: the
    # configured temperatures @seed0 (the §11 on-policy DPO harvest).
    runs = [(s, 0.0) for s in range(args.seeds)] if args.mode == "eval" else list(enumerate(temps))
    for builder in builders:
        node_results = []
        for seed, temp in runs:
            y = lane * float(args.lane_spacing)
            lane += 1
            scenario = builder(y=y)
            backend._start_pos = np.array([0.0, y])
            backend._start_heading = 0.0
            backend.clear_payload()
            backend.set_effort_scale(1.0)
            r = run_closed_loop(
                backend,
                scenario,
                make_policy(temp),
                monitor_cfg="monitor/rule_v0_isaac.yaml",
                fsm_cfg=args.fsm_cfg,
                seed=seed,
                use_compiler=True,
                fsm_baseline=fsm_baseline,
            )
            node_results.append(r)
            rows.append({"temp": temp, "seed": seed, "lane_y": y, **r.to_dict()})
            print(
                f"[isaac] {scenario.name} T={temp} seed={seed} -> success={r.success} "
                f"reached={r.reached} fell={r.fell} stuck={r.stuck} "
                f"attr={r.first_attribution} prim={r.first_primitive}",
                flush=True,
            )
        if args.mode == "pairs":
            all_pairs.extend(
                build_preference_pairs(node_results, max_pairs=int(cfg.rollout.max_pairs_per_node))
            )

    succ = sum(1 for r in rows if r["success"] and r["temp"] == 0.0)
    n0 = sum(1 for r in rows if r["temp"] == 0.0)
    summary = {
        "adapter": args.adapter,
        "policy": args.policy,
        "mode": args.mode,
        "success_rate_temp0": succ / max(1, n0),
        "n_temp0": n0,
        "rows": rows,
    }
    if args.mode == "pairs":
        save_pairs(all_pairs, args.out)
        summary["pair_stats"] = pair_stats(all_pairs)
    with open(f"{args.out}/isaac_rollout_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\nISAAC closed loop: temp-0 success {succ}/{n0} = {summary['success_rate_temp0']:.3f}; "
        f"{'pairs=' + str(len(all_pairs)) if args.mode == 'pairs' else 'eval'}"
    )

    threading.Timer(5.0, lambda: os._exit(0)).start()  # Isaac close() can hang (#6)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
