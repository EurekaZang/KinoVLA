#!/usr/bin/env python
"""A6.1 — θ* ground truth on the O10 effort-decay axis (Paper-A §4 A6, serves C4).

Privileged GROUND-TRUTH intervention boundary: at what effort-decay floor does the DR base policy
(low-level adaptation, NO semantic recovery) stop crossing? Run the BASE condition (monitor
SUPPRESSED ⇒ NeverFireMonitor ⇒ the planner stays NOMINAL, pure cruise) on O10 at floors
{0.4, 0.3, 0.25, 0.2, 0.15} × N seeds, deep_reset (A0.1 determinism), measure reach-in-40s. θ* = the
floor where base reach crosses 0.5 (logistic fit). The grid extends DOWNWARD from E4's base=1.0 at
floor 0.4 toward the startup-paralysis point 0.15 (E4 §3.3 ⇒ θ*∈(0.15, 0.4)). A0.1 gates this: the
early E4 base zeros at 0.4-0.7 were residue-contaminated; deep_reset removes them.

This is the agent-independent θ* (R4): a privileged controller defines it, never the compared agent.

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES \\
        ~/miniconda3/envs/kinovla/bin/python scripts/a6_1_theta_sweep.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


class NeverFireMonitor:
    """A Monitor that never fires — suppresses semantic recovery (pure low-level cruise, base)."""

    def __init__(self) -> None:
        self.events: list = []

    @property
    def anomaly_score(self) -> float:
        return 0.0

    def reset(self) -> None:
        self.events = []

    def step(self, obs) -> None:  # noqa: ANN001, ARG002
        return None


def _logistic(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (x - x0)))


def main() -> int:
    ap = argparse.ArgumentParser(description="A6.1 θ* O10 base-condition sweep (real Go2, A0.1)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--floors", default="0.4,0.3,0.25,0.2,0.15")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--lane-y", type=float, default=4.0)
    ap.add_argument("--out", default="outputs/eval/a6/a6_1_theta_sweep.json")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy
    from kino_vla.map.types import SemanticRegion
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import EffortDecay
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.geometry import Rect
    from kino_vla.vla.rollout import Scenario, run_closed_loop

    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    cfg = load_config("eval/e2.yaml").to_dict()
    backend = IsaacPolicyBackend(
        load_config("sim/go2_skeleton.yaml"), np.array([0.0, args.lane_y]), 0.0
    )
    never = NeverFireMonitor()
    b1 = ProprioBaselinePolicy.from_deployed(
        pcfg, tax, model_path=str(cfg["b1"]["model_path"]), device="cpu"
    )
    floors = [float(f) for f in args.floors.split(",")]
    seed_base = 600  # disjoint

    rows: list[dict] = []
    print(f"[a6.1] BASE condition (monitor suppressed, deep_reset) O10 floors {floors}", flush=True)
    for floor in floors:
        succ = 0
        last = {}
        for si in range(args.seeds):
            seed = seed_base + si
            rect = Rect(cx=3.0, cy=args.lane_y, hx=1.0, hy=1.0)
            scn = Scenario(
                name=f"O10_floor{floor:g}",
                operator=EffortDecay(decay_rate_per_s=0.6, floor=floor, t_start_s=2.0),
                scene_region=SemanticRegion(rect=rect, appearance_class="solid_ground"),
                operator_name="O10_effort_decay",
                appearance_class="solid_ground",
                goal_xy=(6.0, args.lane_y),
                start_xy=(0.0, args.lane_y),
                max_time_s=40.0,
            )
            backend._start_pos = np.asarray(scn.start_xy, dtype=np.float64)
            backend._start_heading = 0.0
            never.reset()
            r = run_closed_loop(
                backend,
                scn,
                b1,
                monitor_cfg="monitor/rule_v0_isaac.yaml",
                fsm_cfg="recovery/fsm_isaac.yaml",
                seed=seed,
                monitor=never,
                deep_reset=True,
            )
            succ += int(r.success)
            last = {
                "reached": bool(r.reached),
                "fell": bool(r.fell),
                "final_dist_m": round(r.final_dist_m, 2),
            }
        rate = succ / max(1, args.seeds)
        rows.append({"floor": floor, "base_reach": round(rate, 3), "n": args.seeds, "last": last})
        print(
            f"[a6.1] floor={floor:<5g} base_reach={rate:.2f} ({succ}/{args.seeds}) last={last}",
            flush=True,
        )

    # logistic fit θ* (floor where base_reach crosses 0.5)
    xs = np.array([r["floor"] for r in rows])
    ys = np.array([r["base_reach"] for r in rows])
    theta_star = float("nan")
    fit = None
    try:
        from scipy.optimize import curve_fit

        p0 = [1.0, 10.0, float(np.mean(xs))]
        popt, _ = curve_fit(_logistic, xs, ys, p0=p0, maxfev=10000)
        L, k, x0 = popt
        theta_star = float(x0)  # logistic midpoint (reach=0.5)
        fit = {
            "L": round(float(L), 3),
            "k": round(float(k), 3),
            "x0": round(float(x0), 4),
            "curve": {
                str(float(f)): round(float(_logistic(np.array([f]), *popt)[0]), 3)
                for f in np.linspace(xs.min(), xs.max(), 20)
            },
        }
        print(f"[a6.1] θ* (logistic midpoint, base_reach=0.5) = {theta_star:.4f}", flush=True)
    except Exception as e:  # noqa: BLE001
        # fallback: linear-interp the 0.5 crossing
        order = np.argsort(-xs)  # floor high→low (reach high→low)
        xs2, ys2 = xs[order], ys[order]
        for i in range(len(xs2) - 1):
            if (ys2[i] - 0.5) * (ys2[i + 1] - 0.5) <= 0 and ys2[i] != ys2[i + 1]:
                theta_star = float(
                    xs2[i] + (0.5 - ys2[i]) * (xs2[i + 1] - xs2[i]) / (ys2[i + 1] - ys2[i])
                )
                break
        print(
            f"[a6.1] θ* (linear-interp 0.5 crossing) = {theta_star:.4f} (scipy unavailable: {e})",
            flush=True,
        )

    result = {
        "floors": floors,
        "seeds": args.seeds,
        "seed_base": seed_base,
        "determinism": "deep_reset(A0.1) + fixed lane_y",
        "rows": rows,
        "theta_star": round(theta_star, 4) if theta_star == theta_star else None,
        "logistic_fit": fit,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[OK] wrote {args.out} | θ*={result['theta_star']}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
