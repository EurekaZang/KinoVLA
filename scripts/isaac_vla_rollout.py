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
    parser.add_argument("--policy", choices=["model", "stub"], default="model")
    parser.add_argument("--config", default="vla/dpo.yaml")
    parser.add_argument("--mode", choices=["eval", "pairs"], default="pairs")
    parser.add_argument("--temperatures", default="0.0,0.7,1.0")
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

    cfg = load_config(args.config)
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    temps = [float(t) for t in args.temperatures.split(",")]
    os.makedirs(args.out, exist_ok=True)

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)

    model = None
    if args.policy == "model":
        import torch

        from kino_vla.vla.model import KinoVLA

        model = KinoVLA.from_pretrained(cfg, device="cuda", adapter_dir=args.adapter)
        model.eval()
        torch.set_grad_enabled(False)

    def make_policy(temp: float):
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
        )

    # Lateral lanes: each scenario at a distinct y so the persistent Isaac prims don't overlap.
    builders = [S.o8_invisible, S.o3_collapse, S.o1_ice]
    rows: list[dict] = []
    all_pairs = []
    lane = 0
    eval_temps = [0.0] if args.mode == "eval" else temps
    for builder in builders:
        node_results = []
        for ti, temp in enumerate(eval_temps):
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
                fsm_cfg="recovery/fsm_isaac.yaml",
                seed=ti,
                use_compiler=True,
            )
            node_results.append(r)
            rows.append({"temp": temp, "lane_y": y, **r.to_dict()})
            print(
                f"[isaac] {scenario.name} T={temp} -> success={r.success} reached={r.reached} "
                f"fell={r.fell} stuck={r.stuck} attr={r.first_attribution} "
                f"prim={r.first_primitive}",
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
