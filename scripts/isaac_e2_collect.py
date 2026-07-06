#!/usr/bin/env python
"""E2 — collect matched-construction CoT samples (real Go2) for the B5-conflict SFT + the eval set.

The #49-matched O4 (proprio-matched to mud) + the matched O2, captured as failure-node Snapshots
(yellow/brown procedural appearance + the matched proprio window), each labelled with the
PRIVILEGED-TRUE category + canonical primitive and a scripted CONFLICT chain-of-thought:

    matched O4 → "proprio resembles compliant mud, but the surface is a yellow adhesive board —
                  trust the vision: adhesion → back off (Backstep), do NOT push through."
    matched O2 → "brown mud, bounded drag — push through with a high-step gait (Switch_Gait)."

The matched-O4 sample is the conflict signal that teaches B5-conflict to weight VISION over the
mud-like proprio (what B5-unshaped, trained only on the ramp-tether O4, never learned). Labels are
privileged-known, so no Oracle is needed. Snapshots are captured by reusing run_closed_loop's
recorder (StubVlaPolicy just triggers the interception); output is in load_suite_sem / SFT format.

Run:  python scripts/isaac_e2_collect.py --headless --split train --seeds 10 \
          --out outputs/eval/e2/matched_train
      python scripts/isaac_e2_collect.py --headless --split eval --seeds 5 --seed-base 500 \
          --out outputs/eval/e2/matched_eval
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Nominal privileged-θ target (TARGET_SCHEMA = mu, payload_kg, effort_scale, support_ratio): the
# matched O2/O4 are tangential-resistance operators, so friction/payload/effort/support are nominal.
_NOMINAL_THETA = [0.8, 0.0, 1.0, 1.0]
_PAIR = "O4_tether|O2_compliance"


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 matched-construction CoT collection")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--config", default="eval/e2.yaml")
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--seeds", type=int, default=10, help="samples per operator (O4 and O2)")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out", default="outputs/eval/e2/matched_train")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841 (keeps the sim app alive)

    from kino_vla.data.dataset import _write_frames, _write_jsonl
    from kino_vla.data.schema import CoTAnnotation, DataSample, RecoveryPrimitive, Verdict
    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.monitor.learned_monitor import load_deployed_monitor
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla import scenarios as S
    from kino_vla.vla.planner import StubVlaPolicy
    from kino_vla.vla.rollout import run_closed_loop

    cfg = load_config(args.config)
    d = cfg.to_dict()
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    monitor = load_deployed_monitor(backend.dt)
    stub = StubVlaPolicy(pcfg, tax)
    preset = d["suite_sem"]["attribution_preset"]

    def _o4(y: float):  # noqa: ANN202
        return S.o4_tether_matched(
            y, force_cap_n=float(preset["force_cap_n"]),
            force_offset_n=float(preset["force_offset_n"]),
            peel_factor=float(preset["peel_factor"]),
        )

    o4_thought = (
        "The tangential resistance resembles compliant mud, but the surface is a yellow adhesive "
        "board — trust the vision over the proprioception: this is adhesion. Back off the sticky "
        "patch rather than pushing through."
    )
    o2_thought = (
        "Brown mud with bounded drag and the same resistance signature — this is compliant "
        "terrain, crossable: push through with a high-step gait."
    )
    # (operator builder, op_name, θ for ground_truth, conflict thought, canonical primitive)
    units = [
        (_o4, "O4_tether", {"k": 14.0, "c": 6.0, "f_break": 1.0e9}, o4_thought,
         RecoveryPrimitive("Backstep", {"distance_m": 0.5})),
        (S.o2_compliance_matched, "O2_compliance", {"k_c": 14.0, "c_c": 6.0, "d_sink": 0.08},
         o2_thought, RecoveryPrimitive("Switch_Gait", {"mode": "high_step"})),
    ]

    samples: list[DataSample] = []
    idx = 0
    for build, op_name, theta, thought, primitive in units:
        gt = tax.ground_truth(op_name, theta)
        category = tax.category_of(op_name)
        for s in range(args.seeds):
            seed = args.seed_base + s
            scn = build(0.0)
            backend._start_pos = np.asarray(scn.start_xy, dtype=np.float64)
            backend._start_heading = float(scn.start_heading)
            res = run_closed_loop(
                backend, scn, stub, monitor_cfg="monitor/rule_v0_isaac.yaml",
                fsm_cfg="recovery/fsm_isaac.yaml", seed=seed, monitor=monitor,
            )
            snap = res.first_snapshot
            if snap is None:
                print(f"[e2.collect] {op_name} seed{seed}: no fire; skip", flush=True)
                continue
            ann = CoTAnnotation(
                thought=thought, attribution=category, primitive=primitive,
                attribution_raw=category, raw_text="",
            )
            sid = f"e2matched_{args.split}_{idx:04d}_{op_name.split('_')[0]}"
            idx += 1
            samples.append(DataSample(
                sample_id=sid, snapshot=snap, ground_truth=gt, annotation=ann,
                verdict=Verdict(keep=True, reason="KEEP", detail="E2 matched-construction label"),
                target_theta=list(_NOMINAL_THETA), ambiguity_pair=_PAIR,
            ))
            print(f"[e2.collect] {op_name} seed{seed} -> {sid} ({category}/{primitive.name})",
                  flush=True)

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "samples.jsonl", samples)
    _write_jsonl(out / "dropped.jsonl", [])
    _write_frames(out / "frames.npz", samples)
    print(f"[e2.collect] wrote {len(samples)} matched samples -> {out}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
