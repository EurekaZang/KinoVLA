#!/usr/bin/env python
"""A4.2 — the interventional label-swap matrix ``M(s, ℓ)`` on the real Go2 (Paper-A §4 A4.2, C3).

Forces each scripted canonical recovery (LABEL) onto each physical scene (SCENARIO) and measures
the outcome, holding everything fixed except the label: oracle trigger (A0.2-primary) marks the
decision moment identically for every label, the deterministic backend (A0.1 deep_reset) makes each
episode reproducible, the base policy is frozen, and the PassThroughShield + the backend's own
velocity/yaw caps are the fixed safety envelope (§0.3 cuts the CBF chapter). The forced label — not
any agent — decides the action, so M(s, ℓ) is a pure causal object.

Scenarios + labels + success criteria come from the pre-registered registry (configs/eval/
a0_registry.yaml, A0.5); the operators are built from the registry θ (the registry is the single
source of truth, no builder/θ drift). Outcomes append incrementally to ``matrix.jsonl`` (resumable;
``--resume`` skips cells already on disk) and a manifest stamps config-hash + commit + the A0.1
determinism note. ``scripts/a4_analyze.py`` reduces the jsonl into the M matrix + cost asymmetry.

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES \\
        ~/miniconda3/envs/kinovla/bin/python scripts/a4_matrix.py --headless [--quick] [--resume]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

from kino_vla.eval.a4_scenarios import build_operator  # shared registry→operator builder

# The forced labels mirror configs/eval/a0_registry.yaml forced_labels.
ALL_LABELS = (
    "continue",
    "backstep_detour",
    "high_step",
    "slow_low",
    "crawl",
    "hold_request",
    "detour_replan",
)
GLOBAL_OPS = {"O5_payload", "O10_effort_decay", "O6_push"}  # temporal, no spatial rect
LANE_Y = 4.0  # A0.1 fixed single-lane geometry (determinism needs absolute coords)
SEED_BASE = 700  # disjoint from training 0–9 and the A0.3 corpus (500)


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"

def main() -> int:
    ap = argparse.ArgumentParser(description="A4 interventional label-swap matrix (real Go2)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--scenarios", default="", help="comma-list (default: all A4 scenarios)")
    ap.add_argument("--labels", default=",".join(ALL_LABELS), help="comma-list of forced labels")
    ap.add_argument("--seeds", type=int, default=10, help="seeds per (scenario, label) cell")
    ap.add_argument("--quick", action="store_true", help="1 seed, first 3 scenarios (smoke)")
    ap.add_argument("--lane-y", type=float, default=LANE_Y, help="fixed lane y (A0.1 determinism)")
    ap.add_argument("--arm", type=float, default=0.2, help="oracle arm delay [s]")
    ap.add_argument("--cruise", type=float, default=0.6)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--out", default="outputs/eval/a4")
    ap.add_argument("--resume", action="store_true", help="skip cells already in matrix.jsonl")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from kino_vla.eval.a4_forced_label import ForcedLabelPolicy, OutcomeRecorder
    from kino_vla.eval.oracle_trigger import OracleTrigger
    from kino_vla.eval.registry import load_registry
    from kino_vla.loop import run_episode
    from kino_vla.map.types import SemanticRegion
    from kino_vla.shield.passthrough import PassThroughShield
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import OperatorStack
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.geometry import Rect
    from kino_vla.vla.rollout import Scenario

    reg = load_registry()
    a4_names = [s for s in reg.scenarios if "A4" in reg[s].used_by]
    names = args.scenarios.split(",") if args.scenarios else a4_names
    names = [n.strip() for n in names if n.strip()]
    if args.quick:
        names = names[:3]
        args.seeds = 1
    labels = [lab.strip() for lab in args.labels.split(",") if lab.strip() in ALL_LABELS]
    print(
        f"[a4] scenarios={names} labels={labels} seeds={args.seeds} lane_y={args.lane_y}",
        flush=True,
    )

    backend = IsaacPolicyBackend(
        load_config("sim/go2_skeleton.yaml"), np.array([0.0, args.lane_y]), 0.0
    )
    dt = backend.dt
    shield = PassThroughShield()
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "matrix.jsonl"
    done: set[str] = set()
    if args.resume and jpath.exists():
        for ln in jpath.read_text().splitlines():
            if ln:
                d = json.loads(ln)
                done.add(f"{d['scenario']}|{d['label']}|{d['seed']}")
        print(f"[a4] --resume: {len(done)} cells already on disk", flush=True)
    if not args.resume:
        jpath.write_text("")  # fresh

    n_done = n_skip = 0
    t0 = time.time()
    for sname in names:
        spec = reg[sname]
        rect = Rect(cx=3.0, cy=args.lane_y, hx=1.0, hy=1.0)
        op = build_operator(spec, rect)
        region = SemanticRegion(rect=rect, appearance_class=spec.appearance_class)
        scn = Scenario(
            name=spec.name,
            operator=op,
            scene_region=region,
            operator_name=spec.operator,
            appearance_class=spec.appearance_class,
            goal_xy=(6.0, args.lane_y),
            start_xy=(0.0, args.lane_y),
            max_time_s=args.max_time,
            success_mode="reach",
        )
        spatial = spec.operator not in GLOBAL_OPS
        for label in labels:
            for si in range(args.seeds):
                seed = SEED_BASE + si
                key = f"{sname}|{label}|{seed}"
                if key in done:
                    n_skip += 1
                    continue
                # fresh deterministic state per cell (A0.1) + a clean oracle/policy/recorder
                backend._start_pos = np.asarray(scn.start_xy, dtype=np.float64)
                backend._start_heading = 0.0
                oracle = OracleTrigger.for_scenario(scn, dt=dt, arm_delay_s=args.arm)
                policy = ForcedLabelPolicy(
                    label, rect, np.asarray(scn.goal_xy), cruise=args.cruise, dt=dt
                )
                rec = OutcomeRecorder(
                    rect=rect,
                    goal_xy=np.asarray(scn.goal_xy),
                    success_criterion=spec.success_criterion,
                    backend=backend,
                    label=label,
                    spatial=spatial,
                    cruise=args.cruise,
                )
                # hand the oracle's event to BOTH the policy (start the recovery) and the recorder
                # (mark the decision moment): the loop calls policy.on_event; the recorder reads the
                # event from on_step. We also forward via a thin monitor wrapper.
                result = run_episode(
                    backend,
                    OperatorStack([op]),
                    oracle,
                    policy,
                    shield,
                    seed=seed,
                    goal_xy=np.asarray(scn.goal_xy),
                    goal_tol_m=0.6,
                    max_time_s=args.max_time,
                    on_step=rec,
                    deep_reset=True,
                )
                outcome = rec.build_outcome(
                    scenario=sname,
                    seed=seed,
                    reached=result.goal_reached,
                    fell=result.fell,
                    final_dist_m=result.final_dist_m,
                    sim_time_s=result.sim_time_s,
                )
                with jpath.open("a") as f:
                    f.write(json.dumps(outcome.__dict__) + "\n")
                n_done += 1
                if n_done % 10 == 0 or n_done < 10:
                    print(
                        f"[a4] {sname:20} {label:16} s{seed} → succ={int(outcome.success)} "
                        f"fell={int(outcome.fell)} cat={int(outcome.catapult)} "
                        f"immob={int(outcome.immobilized)} dist={outcome.final_dist_m:.1f} "
                        f"t={outcome.sim_time_s:.0f}s peakω={outcome.peak_omega:.1f} "
                        f"({n_done} done, {n_skip} skip, {time.time() - t0:.0f}s)",
                        flush=True,
                    )

    # ---- manifest (config hash + commit + A0.1 determinism note) ----
    payload = jpath.read_text() if jpath.exists() else ""
    card = {
        "name": "a4_label_swap_matrix",
        "commit": git_commit(),
        "seeds_per_cell": args.seeds,
        "seed_base": SEED_BASE,
        "lane_y": args.lane_y,
        "scenarios": names,
        "labels": labels,
        "arm_s": args.arm,
        "cruise": args.cruise,
        "max_time_s": args.max_time,
        "determinism": "deep_reset(A0.1) + fixed lane_y ⇒ order-independent, byte-reproducible",
        "n_outcomes": n_done,
        "n_skipped_resume": n_skip,
        "outcome_sha256": hashlib.sha256(payload.encode()).hexdigest()[:12],
    }
    (out_dir / "matrix_manifest.json").write_text(json.dumps(card, indent=2))
    print(f"\n[a4] DONE: {n_done} outcomes (+{n_skip} resumed) → {jpath}", flush=True)
    print(json.dumps(card, indent=2))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
