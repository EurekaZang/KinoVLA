#!/usr/bin/env python
"""E4 PREREQUISITE — left-endpoint existence check (real Go2), BEFORE any θ-sweep.

E4's curve is a SUCCESS→FAILURE flip: at low θ the low-level adaptation succeeds (the A-class
region), past θ* it fails and only semantic recovery reaches the goal. That flip is only meaningful
if a LEFT ENDPOINT exists — a gentle θ_low where low-level closed-loop reach succeeds cleanly
(≥0.8). E2's Suite-Cal already warns this may be missing on exactly E4's two axes (O2 k_c, O10
effort decay): O2_A had B1=1.00 (endpoint likely present) but O10_A was 0.00 for ALL agents. If even
the gentlest θ fails, the curve is a flat failure→failure line and θ* does not exist — the problem
would be the closed-loop reach CALIPER (monitor false-fire / recovery disrupting reach within the
time budget), not the failure, and E4 must fix the caliper first (the user's directive).

This check separates the two causes by running each θ under TWO conditions on the SAME backend:

  - ``base``  — monitor SUPPRESSED (NeverFireMonitor) ⇒ pure low-level cruise to goal. Answers:
                "is this θ physically crossable by the DR-trained base policy at all?" (the true
                left endpoint of E4's low-level-success curve).
  - ``b1``    — the deployed LearnedMonitor + B1 (ProprioBaselinePolicy, proprio-only) ⇒ the actual
                closed-loop. Answers: "does the monitor+recovery harness PRESERVE that success?"

Diagnosis per operator:
  base≥0.8 & b1≥0.8 at some θ_low → left endpoint EXISTS, caliper OK → E4 green.
  base≥0.8 but b1<0.8            → caliper BROKEN (false-fire / recovery disrupts reach) → fix it.
  base<0.8 even at gentlest θ    → task/geometry too hard for this operator → fix θ/caliper.

NOTE (recorded, not resolved here): the eventual E4 truth boundary θ* must be defined by an IDEAL
privileged low-level controller's success/failure flip — NEVER by B1 or any compared agent (else
θ* and the agent decision point both depend on the same model ⇒ circular). The ``base`` condition
here (DR base policy, no attribution) is the right primitive for that, but θ* calibration is E4
proper. This script only answers the cheaper, prerequisite question: does the left endpoint exist.

Run:  KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct HF_HUB_OFFLINE=1 \
        python scripts/isaac_e4_leftpoint.py --headless --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


class NeverFireMonitor:
    """A Monitor (event.py Protocol) that never fires — suppresses all semantic recovery so the
    planner stays NOMINAL and the rollout measures PURE low-level cruise-to-goal."""

    def __init__(self) -> None:
        self.events: list = []

    @property
    def anomaly_score(self) -> float:
        return 0.0

    def reset(self) -> None:
        self.events = []

    def step(self, obs) -> None:  # noqa: ANN001, ARG002 - constant no-fire by design
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 left-endpoint existence check (O2 k_c, O10 floor)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--config", default="eval/e2.yaml")
    ap.add_argument("--seeds", type=int, default=3, help="rollouts per (operator, θ, condition)")
    ap.add_argument("--o2-kc", default="6,10,14,18", help="O2 compliance k_c grid (gentle→matched)")
    ap.add_argument("--o10-floor", default="0.9,0.7,0.5,0.4", help="O10 effort-decay floor grid")
    ap.add_argument("--success-thr", type=float, default=0.8, help="clean-success bar for ≥")
    ap.add_argument("--monitor-path", default=None,
                    help="deployed-monitor checkpoint for the b1 condition (default: deployed); "
                         "pass monitor_learned/monitor_abaware for the caliper-fix re-check")
    ap.add_argument("--debounce", type=int, default=None,
                    help="online op-point hardening: debounce steps (default deployed 5; the E4 "
                         "caliper fix uses ~25 = require the hazard to PERSIST, reject spikes)")
    ap.add_argument("--arm", type=float, default=None, help="arm warmup seconds (default 2.5)")
    ap.add_argument("--out", default="outputs/eval/e4/leftpoint.json")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841 (keeps the sim app alive)

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy
    from kino_vla.map.types import SemanticRegion
    from kino_vla.monitor.learned_monitor import load_deployed_monitor
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import ComplianceField, EffortDecay
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.geometry import Rect
    from kino_vla.vla import scenarios as S
    from kino_vla.vla.rollout import Scenario, run_closed_loop

    cfg = load_config(args.config).to_dict()
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    op_overrides: dict = {}
    if args.debounce is not None:
        op_overrides["debounce_steps"] = int(args.debounce)
    if args.arm is not None:
        op_overrides["arm_s"] = float(args.arm)
    mp = {} if args.monitor_path is None else {"model_path": args.monitor_path}
    deployed = load_deployed_monitor(backend.dt, **mp, **op_overrides)
    never = NeverFireMonitor()
    b1 = ProprioBaselinePolicy.from_deployed(
        pcfg, tax, model_path=str(cfg["b1"]["model_path"]), device="cpu"
    )

    # Use the canonical Suite-Cal geometry (start 0 → hazard x∈[2,4] → goal x=6, tol 0.6, 40 s).
    def _o2(k_c: float) -> Scenario:
        rect = Rect(cx=S._HAZARD_CX, cy=0.0, hx=S._HX, hy=S._HY)
        op = ComplianceField(rect, k_c=float(k_c), c_c=float(k_c) * 0.5, d_sink=0.03)
        return Scenario(
            name=f"O2_kc{k_c:g}", operator=op, scene_region=op.scene_region(),
            operator_name="O2_compliance", appearance_class=op.scene_region().appearance_class,
            goal_xy=(S._GOAL_X, 0.0), start_xy=(0.0, 0.0), max_time_s=S._MAX_T,
        )

    def _o10(floor: float) -> Scenario:
        rect = Rect(cx=S._HAZARD_CX, cy=0.0, hx=S._HX, hy=S._HY)
        return Scenario(
            name=f"O10_floor{floor:g}",
            operator=EffortDecay(decay_rate_per_s=0.6, floor=float(floor), t_start_s=2.0),
            scene_region=SemanticRegion(rect=rect, appearance_class="solid_ground"),
            operator_name="O10_effort_decay", appearance_class="solid_ground",
            goal_xy=(S._GOAL_X, 0.0), start_xy=(0.0, 0.0), max_time_s=S._MAX_T,
        )

    families = [
        ("O2_compliance", "k_c", [float(x) for x in args.o2_kc.split(",")], _o2),
        ("O10_effort_decay", "floor", [float(x) for x in args.o10_floor.split(",")], _o10),
    ]
    conditions = [("base", never, "monitor suppressed → pure low-level"),
                  ("b1", deployed, "deployed monitor + B1 → closed loop")]

    results: dict = {"geometry": {"start_x": 0.0, "hazard_x": [2.0, 4.0], "goal_x": S._GOAL_X,
                                  "goal_tol_m": 0.6, "max_time_s": S._MAX_T},
                     "seeds": args.seeds, "success_thr": args.success_thr, "families": {}}
    for op_name, pname, grid, build in families:
        fam: dict = {"param": pname, "grid": grid, "rows": {}}
        for theta in grid:
            fam["rows"][f"{theta:g}"] = {}
            for cname, mon, _desc in conditions:
                succ = 0
                last = {}
                for s in range(args.seeds):
                    # reset the deployed monitor's running state across seeds (never is stateless)
                    if hasattr(mon, "reset"):
                        mon.reset()
                    scn = build(theta)
                    backend._start_pos = np.asarray(scn.start_xy, dtype=np.float64)
                    backend._start_heading = 0.0
                    r = run_closed_loop(
                        backend, scn, b1, monitor_cfg="monitor/rule_v0_isaac.yaml",
                        fsm_cfg="recovery/fsm_isaac.yaml", seed=s, monitor=mon)
                    succ += int(r.success)
                    last = {"reached": bool(r.reached), "fell": bool(r.fell),
                            "stuck": bool(r.stuck), "final_dist_m": round(r.final_dist_m, 2),
                            "n_rounds": r.n_rounds, "first_attr": r.first_attribution,
                            "first_prim": r.first_primitive}
                rate = succ / max(1, args.seeds)
                fam["rows"][f"{theta:g}"][cname] = {"success_rate": rate, "last": last}
                print(f"[e4.left] {op_name:18} {pname}={theta:<5g} {cname:5} "
                      f"success={rate:.2f}  last={last}", flush=True)
        # verdict per family: gentlest θ where base≥thr; does b1 preserve it?
        thr = args.success_thr
        base_ok = [t for t in grid if fam["rows"][f"{t:g}"]["base"]["success_rate"] >= thr]
        b1_ok = [t for t in grid if fam["rows"][f"{t:g}"]["b1"]["success_rate"] >= thr]
        if base_ok and b1_ok:
            verdict = "LEFT_ENDPOINT_EXISTS"
        elif base_ok and not b1_ok:
            verdict = "CALIPER_BROKEN (base crosses but closed-loop recovery disrupts reach)"
        else:
            verdict = "NO_LEFT_ENDPOINT (base policy fails even at gentlest θ → task/θ too hard)"
        fam["base_ok_theta"] = base_ok
        fam["b1_ok_theta"] = b1_ok
        fam["verdict"] = verdict
        results["families"][op_name] = fam
        print(f"[e4.left] >>> {op_name}: {verdict}  (base_ok θ={base_ok}, b1_ok θ={b1_ok})",
              flush=True)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\n=== E4 left-endpoint existence check ===")
    for op_name, fam in results["families"].items():
        print(f"{op_name}: {fam['verdict']}")
    print(f"wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
