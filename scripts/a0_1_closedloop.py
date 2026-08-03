#!/usr/bin/env python
"""A0.1 closed-loop confirmation — does deep_reset remove the E4 order-sensitivity? (real Go2).

The snapshot certificate (scripts/a0_1_analysis.py) proved deep_reset + fixed geometry ⇒ byte-
identical, order-independent lanes. This confirms the same on the CLOSED loop — the E4 §3.2
reconcile signature: the SAME (scenario, seed) closed-loop reach outcome must NOT depend on the
predecessor episode. E4 found O2_A ∈ {0.00 (after O1_A), 0.67 (isolated)} on the reused app; with
deep_reset the two histories must AGREE (and match a fresh run). Also re-runs the O10 base grid to
check the §3.7 floor=0.5 non-monotonic dip / base 0.00↔1.00 flip are gone.

Compares mechanism ∈ {naive, deep}: for each, run scenario X(seed) FIRST, then after a different
predecessor, and diff the RolloutResult. deep should give identical outcomes; naive should not.

Run:  KINOVLA_MODEL_ID=... python scripts/a0_1_closedloop.py --headless --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


class NeverFire:
    """Monitor that never fires (base condition: pure low-level cruise)."""

    events: list = []

    @property
    def anomaly_score(self) -> float:
        return 0.0

    def reset(self) -> None:
        self.events = []

    def step(self, obs):  # noqa: ANN001, ANN204, ARG002
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.1 closed-loop order-independence confirmation")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--config", default="eval/e2.yaml")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="outputs/eval/a0/a0_1_closedloop.json")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy
    from kino_vla.monitor.learned_monitor import load_deployed_monitor
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla import scenarios as S
    from kino_vla.vla.rollout import run_closed_loop

    cfg = load_config(args.config).to_dict()
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    deployed = load_deployed_monitor(backend.dt)
    b1 = ProprioBaselinePolicy.from_deployed(
        pcfg, tax, model_path=str(cfg["b1"]["model_path"]), device="cpu"
    )

    def run(scn, monitor, seed, deep):  # noqa: ANN001
        backend._start_pos = np.asarray(scn.start_xy, dtype=np.float64)
        backend._start_heading = float(scn.start_heading)
        if hasattr(monitor, "reset"):
            monitor.reset()
        r = run_closed_loop(backend, scn, b1, monitor_cfg="monitor/rule_v0_isaac.yaml",
                            fsm_cfg="recovery/fsm_isaac.yaml", seed=seed, monitor=monitor,
                            deep_reset=deep)
        return {"success": bool(r.success), "reached": bool(r.reached), "fell": bool(r.fell),
                "final_dist_m": round(r.final_dist_m, 3), "attr": r.first_attribution}

    out: dict = {"seeds": args.seeds, "mechanisms": {}}
    for mech, deep in (("naive", False), ("deep", True)):
        o2A, o1A = S.o2_compliance_A, S.o1_ice_A
        # (1) O2_A isolated-first vs after-O1_A predecessors, per seed → order-sensitivity check.
        iso, after, agree = [], [], []
        for s in range(args.seeds):
            r_iso = run(o2A(0.0), deployed, s, deep)          # O2_A as the first episode
            run(o1A(0.0), deployed, (s + 1) % 7, deep)         # a DIFFERENT predecessor episode
            run(o1A(0.0), deployed, (s + 2) % 7, deep)
            r_after = run(o2A(0.0), deployed, s, deep)         # same O2_A(seed) after O1_A history
            iso.append(r_iso)
            after.append(r_after)
            same = (r_iso["success"] == r_after["success"] and
                    abs(r_iso["final_dist_m"] - r_after["final_dist_m"]) < 1e-3)
            agree.append(same)
            print(f"[a0.1cl/{mech}] O2_A seed{s}: iso={r_iso['success']}"
                  f"(d{r_iso['final_dist_m']}) after={r_after['success']}"
                  f"(d{r_after['final_dist_m']}) agree={same}", flush=True)
        # (2) O10 base grid (monitor suppressed) with this mechanism — floor=0.5 dip / flip check.
        never = NeverFire()
        o10_grid = {}
        for floor in (0.9, 0.7, 0.5, 0.4):
            from kino_vla.map.types import SemanticRegion
            from kino_vla.sim.operators import EffortDecay
            from kino_vla.utils.geometry import Rect
            from kino_vla.vla.rollout import Scenario
            rect = Rect(cx=S._HAZARD_CX, cy=0.0, hx=S._HX, hy=S._HY)
            scn = Scenario(name=f"O10_f{floor}",
                           operator=EffortDecay(decay_rate_per_s=0.6, floor=floor, t_start_s=2.0),
                           scene_region=SemanticRegion(rect=rect, appearance_class="solid_ground"),
                           operator_name="O10_effort_decay", appearance_class="solid_ground",
                           goal_xy=(S._GOAL_X, 0.0), start_xy=(0.0, 0.0), max_time_s=S._MAX_T)
            succ = sum(run(scn, never, s, deep)["success"] for s in range(args.seeds))
            rate = round(succ / max(1, args.seeds), 3)
            o10_grid[f"{floor:g}"] = rate
            print(f"[a0.1cl/{mech}] O10 base floor={floor}: reach={rate}", flush=True)
        out["mechanisms"][mech] = {
            "o2A_isolated": iso, "o2A_after_o1": after,
            "o2A_order_independent": bool(all(agree)),
            "o10_base_grid": o10_grid,
        }

    out["verdict"] = {
        "deep_order_independent": out["mechanisms"]["deep"]["o2A_order_independent"],
        "naive_order_independent": out["mechanisms"]["naive"]["o2A_order_independent"],
    }
    p = REPO_ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("\n=== A0.1 closed-loop confirmation ===")
    print(json.dumps(out["verdict"], indent=2))
    print(f"deep O10 base grid : {out['mechanisms']['deep']['o10_base_grid']}")
    print(f"naive O10 base grid: {out['mechanisms']['naive']['o10_base_grid']}")
    print(f"wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
