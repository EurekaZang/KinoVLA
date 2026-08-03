#!/usr/bin/env python
"""E4 reconcile — is the closed-loop reach outcome on O2_A order/context-sensitive? (real Go2)

E2's recorded Suite-Cal has B1 O2_compliance_A = 1.00 (3/3 reach), but the E4 left-point sweep got
B1 O2 (k_c=6, identical θ/seeds/monitor) = 0.00 — differing only in run ORDER on the reused Isaac
app (E2 ran O2_A after O1_A episodes; E4 ran O2 first, after base-suppressed runs). This isolates
the variable: run B1 closed-loop on the SAME o2_compliance_A under two histories, plus the O10_A
base-vs-closed contrast, on ONE backend. If the two O2_A histories disagree, the closed-loop reach
metric is fragile (state carryover) — itself a reason the E4 curve must not be built on it un-fixed.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="E4 reconcile O2_A order-sensitivity")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--config", default="eval/e2.yaml")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="outputs/eval/e4/reconcile.json")
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

    class Never:
        events: list = []
        @property
        def anomaly_score(self) -> float:
            return 0.0
        def reset(self) -> None:
            self.events = []
        def step(self, obs):  # noqa: ANN001, ANN204, ARG002
            return None

    never = Never()

    def run(scn, monitor, seed):  # noqa: ANN001
        backend._start_pos = np.asarray(scn.start_xy, dtype=np.float64)
        backend._start_heading = float(scn.start_heading)
        if hasattr(monitor, "reset"):
            monitor.reset()
        r = run_closed_loop(backend, scn, b1, monitor_cfg="monitor/rule_v0_isaac.yaml",
                            fsm_cfg="recovery/fsm_isaac.yaml", seed=seed, monitor=monitor)
        return r

    def block(label, scn, monitor):  # noqa: ANN001
        succ, rows = 0, []
        for s in range(args.seeds):
            r = run(scn, monitor, s)
            succ += int(r.success)
            rows.append({"seed": s, "success": bool(r.success), "reached": bool(r.reached),
                         "fell": bool(r.fell), "final_dist_m": round(r.final_dist_m, 2),
                         "n_rounds": r.n_rounds, "attr": r.first_attribution,
                         "prim": r.first_primitive})
            print(f"[reconcile] {label:34} seed{s}: success={r.success} "
                  f"dist={r.final_dist_m:.2f} attr={r.first_attribution} prim={r.first_primitive}",
                  flush=True)
        rate = succ / max(1, args.seeds)
        print(f"[reconcile] {label:34} RATE={rate:.2f}", flush=True)
        return {"rate": rate, "rows": rows}

    out: dict = {}
    o2A, o1A, o10A = S.o2_compliance_A(0.0), S.o1_ice_A(0.0), S.o10_effort_decay_A(0.0)
    # (1) O2_A B1 ISOLATED — first thing on a fresh backend (no preceding episodes).
    out["o2A_b1_isolated_first"] = block("O2_A b1 (isolated, first)", o2A, deployed)
    # (2) Mimic the E2 order: O1_A B1 ×3 THEN O2_A B1 ×3 (history = ice episodes).
    out["o1A_b1_warmup"] = block("O1_A b1 (warmup)", o1A, deployed)
    out["o2A_b1_after_o1"] = block("O2_A b1 (after O1_A, E2 order)", o2A, deployed)
    # (3) O10_A base-vs-closed (confirm the caliper diagnosis reproduces).
    out["o10A_base"] = block("O10_A base (monitor suppressed)", o10A, never)
    out["o10A_b1"] = block("O10_A b1 (closed loop)", o10A, deployed)

    out["o2A_order_sensitive"] = (
        out["o2A_b1_isolated_first"]["rate"] != out["o2A_b1_after_o1"]["rate"]
    )
    p = REPO_ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("\n=== E4 reconcile ===")
    print(f"O2_A b1 isolated-first : {out['o2A_b1_isolated_first']['rate']:.2f}")
    print(f"O2_A b1 after-O1 (E2)  : {out['o2A_b1_after_o1']['rate']:.2f}")
    print(f"O2_A order-sensitive?  : {out['o2A_order_sensitive']}")
    print(f"O10_A base / b1        : {out['o10A_base']['rate']:.2f} / {out['o10A_b1']['rate']:.2f}")
    print(f"wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
