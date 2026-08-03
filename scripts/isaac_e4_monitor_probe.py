#!/usr/bin/env python
"""E4 caliper diagnosis — the deployed monitor's hazard-probability response vs θ (real Go2).

Confirms the false-fire diagnosis QUANTITATIVELY and answers the red-line question: is there a FIXED
observable-signal threshold that naturally separates A-class (low-level suffices, should stay quiet)
from B-class (needs semantics, should fire), or is the detector saturated-high across all θ (⇒ it
must be retrained with A-class perturbations as negatives)?

Method (nothing modified, no retrain): drive the base policy NOMINALLY across the patch (a
``LoggingMonitor`` that returns None every step ⇒ no recovery, pure low-level cruise), while a
PASSIVE copy of the deployed LearnedMonitor — loaded at ``threshold=1.1`` so it never fires and
never enters cooldown — observes the SAME trajectory and records, per step, its raw + EMA hazard
probability and attribution argmax. Sweeps O2 ``k_c`` and O10 ``floor`` from gentle (A) to severe
(B). Per θ we report the peak EMA hazard while IN-PATCH (x∈[2,4]) and whether it crosses the
deployed 0.6 operating point — i.e. would the deployed monitor have (falsely, on gentle θ) fired.

The monitor consumes ONLY ``monitor_features(obs)`` (slip / tracking / tilt / effort / support —
observable signals) and a fixed threshold; θ is NEVER an input. So the hazard-vs-θ curve this
produces is exactly the observable-signal response the E4 caliper fix must calibrate against — a
legitimate boundary, not a θ-peek.

Run:  KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct HF_HUB_OFFLINE=1 \
        python scripts/isaac_e4_monitor_probe.py --headless --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 monitor hazard-vs-θ probe (O2 k_c, O10 floor)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--config", default="eval/e2.yaml")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--o2-kc", default="6,10,14,18")
    ap.add_argument("--o10-floor", default="0.9,0.7,0.5,0.4,0.15")
    ap.add_argument("--fire-thr", type=float, default=0.6, help="deployed op-point fire threshold")
    ap.add_argument("--monitor-path", default=None,
                    help="monitor checkpoint to probe (default: the deployed monitor); "
                         "pass monitor_learned/monitor_abaware for the E4 caliper-fix check")
    ap.add_argument("--out", default="outputs/eval/e4/monitor_probe.json")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy
    from kino_vla.map.types import SemanticRegion
    from kino_vla.monitor.learned_monitor import CLASS_NAMES, load_deployed_monitor
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
    # A recovery policy is REQUIRED by VlaPlanner but is consulted ONLY on a monitor event; the
    # LoggingMonitor never fires, so the planner drives pure nominal cruise (matches leftpoint).
    b1 = ProprioBaselinePolicy.from_deployed(
        pcfg, tax, model_path=str(cfg["b1"]["model_path"]), device="cpu"
    )

    class LoggingMonitor:
        """Passive observer: runs the deployed detector but NEVER fires (threshold 1.1) so the
        robot cruises nominally; records the hazard EMA / raw / attr argmax each step."""

        def __init__(self) -> None:
            # threshold 1.1 ⇒ EMA (∈[0,1]) can never cross it ⇒ never fires, never cools down ⇒
            # continuous logging over the whole crossing.
            mp = args.monitor_path
            self._inner = (
                load_deployed_monitor(backend.dt, threshold=1.1) if mp is None
                else load_deployed_monitor(backend.dt, model_path=mp, threshold=1.1)
            )
            self.events: list = []
            self.trace: list[dict] = []

        @property
        def anomaly_score(self) -> float:
            return self._inner.anomaly_score

        def reset(self) -> None:
            self._inner.reset()
            self.events = []
            self.trace = []

        def step(self, obs):  # noqa: ANN001, ANN204
            self._inner.step(obs)  # advances EMA/attr; returns None (thr 1.1) — discarded
            self.trace.append({
                "t": float(obs.t), "x": float(obs.pos[0]),
                "raw": self._inner.raw_prob, "ema": self._inner.anomaly_score,
                "attr": CLASS_NAMES[int(np.argmax(self._inner.attr_probs))],
            })
            return None  # never fire ⇒ base policy cruises, no recovery contaminates the trace

    log_mon = LoggingMonitor()

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
    px0, px1 = 2.0, 4.0  # in-patch x-band

    results: dict = {"fire_thr": args.fire_thr, "seeds": args.seeds,
                     "patch_x": [px0, px1], "families": {}}
    for op_name, pname, grid, build in families:
        fam: dict = {"param": pname, "grid": grid, "rows": {}}
        for theta in grid:
            peaks, in_peaks, reached_ct, attrs = [], [], 0, []
            for s in range(args.seeds):
                log_mon.reset()
                scn = build(theta)
                backend._start_pos = np.asarray(scn.start_xy, dtype=np.float64)
                backend._start_heading = 0.0
                r = run_closed_loop(backend, scn, b1, monitor_cfg="monitor/rule_v0_isaac.yaml",
                                    fsm_cfg="recovery/fsm_isaac.yaml", seed=s, monitor=log_mon)
                tr = log_mon.trace
                ema = [d["ema"] for d in tr] or [0.0]
                inpatch = [d for d in tr if px0 <= d["x"] <= px1]
                in_ema = [d["ema"] for d in inpatch] or [0.0]
                peaks.append(max(ema))
                in_peaks.append(max(in_ema))
                reached_ct += int(r.reached)
                # attribution argmax at the in-patch peak-EMA step
                if inpatch:
                    top = max(inpatch, key=lambda d: d["ema"])
                    attrs.append(top["attr"])
            peak_ema = float(np.mean(peaks))
            in_peak_ema = float(np.mean(in_peaks))
            would_fire = in_peak_ema >= args.fire_thr
            fam["rows"][f"{theta:g}"] = {
                "peak_ema": round(peak_ema, 3),
                "in_patch_peak_ema": round(in_peak_ema, 3),
                "would_fire_at_thr": bool(would_fire),
                "base_reached_rate": reached_ct / max(1, args.seeds),
                "in_patch_attr_argmax": attrs,
            }
            print(f"[e4.probe] {op_name:18} {pname}={theta:<5g} in-patch peak EMA="
                  f"{in_peak_ema:.3f} would_fire(≥{args.fire_thr})={would_fire} "
                  f"base_reached={reached_ct}/{args.seeds} attr={attrs}", flush=True)
        results["families"][op_name] = fam

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\n=== E4 monitor hazard-vs-θ probe (deployed op-point thr=0.6) ===")
    for op_name, fam in results["families"].items():
        print(f"-- {op_name} ({fam['param']}) --")
        for th, row in fam["rows"].items():
            flag = "FIRE" if row["would_fire_at_thr"] else "quiet"
            print(f"   {fam['param']}={th:<5}: in-patch peak EMA={row['in_patch_peak_ema']:.3f} "
                  f"[{flag}]  base_reached={row['base_reached_rate']:.2f}")
    print(f"wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
