"""M3 shield report: adversarial-command safety + the §6.9 latency budget table.

Two M3 exit-criterion artifacts, both auto-generated from logged rollouts (QA 5.4):

1. Adversarial-command safety (spec §6.6): stream hostile velocity commands at the
   surrogate with the CBF-QP shield active vs. bypassed and report falls. Gate:
   zero falls shielded on dry ground, non-zero bypassed.
2. End-to-end latency budget (spec §6.9): measure the fast-loop stages we have
   (Kino-Monitor, Reflex, CBF-QP solve, CBF adjudicate+compile) over a representative
   command stream and render the budget table; M4/M7 stages stay pending.

Usage:
    python scripts/shield_adversarial.py [--seeds 0 1 2 3] [--steps 600]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from kino_vla.monitor.learned_monitor import load_deployed_monitor
from kino_vla.monitor.reflex import Reflex
from kino_vla.shield.adversarial import HOSTILE_PROFILES, run_adversarial_suite
from kino_vla.shield.cbf_shield import CbfShield
from kino_vla.shield.latency import LatencyBudget
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.utils.config import REPO_ROOT, load_config
from kino_vla.utils.seeding import seed_everything


def adversarial_report(seeds: list[int], steps: int) -> bool:
    summary = run_adversarial_suite(seeds=seeds, include_ice=True, n_steps=steps)
    dry = run_adversarial_suite(seeds=seeds, include_ice=False, n_steps=steps)
    print("=== Adversarial-command safety (spec §6.6) ===")
    print(f"  profiles: {', '.join(HOSTILE_PROFILES)}")
    print(f"  seeds: {seeds}   steps/episode: {steps}")
    print(
        f"  DRY GROUND  shielded falls: {dry.falls(True)}/{dry.count(True)}"
        f"   bypassed falls: {dry.falls(False)}/{dry.count(False)}"
    )
    ice_sh = summary.falls(True) - dry.falls(True)
    ice_by = summary.falls(False) - dry.falls(False)
    n_ice = summary.count(True) - dry.count(True)
    print(
        f"  + ICE (low-μ, outside the capture-point guarantee)  shielded falls: {ice_sh}/{n_ice}"
        f"   bypassed falls: {ice_by}/{n_ice}"
    )
    print(
        "    note: sustained slip is NOT a capture-point topple, so the §6.6 CBF does"
        " not prevent it (and can be anti-protective). Ice safety = M4 μ̂ + M7 planner"
        " (Set_Constraint low-speed / Hold_and_Request), NOT this shield."
    )
    ok = dry.falls(True) == 0 and dry.falls(False) > 0
    print(f"  GATE (dry, zero shielded / nonzero bypassed): {'PASS' if ok else 'FAIL'}")
    return ok


def latency_report(steps: int) -> LatencyBudget:
    lb = LatencyBudget()
    backend = SurrogateBackend(load_config("sim/surrogate.yaml"), np.zeros(2), 0.0)
    obs = backend.reset(0)
    monitor = load_deployed_monitor(backend.dt)
    reflex = Reflex(load_config("recovery/reflex_v0.yaml"), backend)
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    rng = np.random.default_rng(0)
    for k in range(steps):
        # Representative stream: cruise interleaved with hostile bursts.
        cmd = np.array([0.8, 0.0, 0.2]) if k % 3 else rng.uniform(-3, 3, 3)
        t0 = time.perf_counter()
        monitor.step(obs)
        lb.record("kino_monitor", time.perf_counter() - t0)
        t0 = time.perf_counter()
        reflex.step(obs)
        lb.record("reflex", time.perf_counter() - t0)
        obs = backend.step(shield.filter(cmd, obs).cmd)
    lb.extend("cbf_qp", shield.stats.qp_times_s)
    lb.extend("cbf_compile", shield.stats.shield_times_s)
    print("\n=== End-to-end latency budget (spec §6.9) ===")
    print(lb.table())
    return lb


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 shield adversarial + latency report")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--out", default="outputs/shield/latency_budget.md")
    args = parser.parse_args()
    seed_everything(0)

    ok = adversarial_report(args.seeds, args.steps)
    lb = latency_report(args.steps)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# KiNO M3 — CBF-QP latency budget (spec §6.9)\n\n" + lb.table() + "\n")
    print(f"\nlatency table written to {out}")

    qp_ok = lb.within_budget("cbf_qp")
    print(
        f"\nQP p99 budget (<1 ms): {'PASS' if qp_ok else 'FAIL'} "
        f"(p99={lb.stats('cbf_qp')['p99']:.3f} ms)"
    )
    print("PASS: shield_adversarial" if (ok and qp_ok) else "FAIL: shield_adversarial")
    return 0 if (ok and qp_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
