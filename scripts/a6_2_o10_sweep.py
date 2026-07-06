#!/usr/bin/env python
"""A6.2 — O10 floor-sweep snapshot collection + agent decision-flip eval (Paper-A §4 A6, C4).

Collects O10 (EffortDecay) snapshots at floors {0.4..0.15} (matching A6.1's θ* grid) into
a frozen sweep, then evaluates the agent ROSTER snapshot-level: does it output `continue` (the decay
is mild, low-level suffices) or an INTERVENTION (Switch_Gait/Set_Constraint — decay past θ*)?
The DECISION-FLIP point (floor where continue→intervention) is reported vs A6.1 ground-truth θ*.
This is the agent-independent θ* vs the agent's learned flip — the C4 boundary claim (a real
semantic agent's flip tracks the privileged boundary). Same deterministic harness (deep_reset
+ fixed lane_y) + A0.3 snapshot schema. Run the collector, then the eval (both GPU, one app).

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES KINOVLA_MODEL_ID=… \\
        ~/miniconda3/envs/kinovla/bin/python scripts/a6_2_o10_sweep.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import deque

import numpy as np

FLOORS = (0.4, 0.3, 0.25, 0.2, 0.15)
SEEDS_PER_FLOOR = 6
SEED_BASE = 620


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
    ap = argparse.ArgumentParser(description="A6.2 O10 floor-sweep collect + agent decision-flip")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--out", default="outputs/eval/a6/corpus_o10_sweep")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--binding-t", type=int, default=100)
    ap.add_argument("--arm", type=float, default=0.2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lane-y", type=float, default=4.0)
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from kino_vla.data.snapshot import SnapshotRecorder
    from kino_vla.eval.oracle_trigger import OracleTrigger
    from kino_vla.map.types import SemanticRegion
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import EffortDecay
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.geometry import Rect

    hcfg = load_config("data/hindsight.yaml")
    backend = IsaacPolicyBackend(
        load_config("sim/go2_skeleton.yaml"), np.array([0.0, args.lane_y]), 0.0
    )
    dt = backend.dt
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    spath = out_dir / "samples.jsonl"
    if spath.exists():
        spath.unlink()
    records: list[dict] = []
    batch: dict[str, np.ndarray] = {}
    seeds = range(1 if args.quick else SEEDS_PER_FLOOR)

    def read60() -> np.ndarray:
        o = getattr(backend, "_obs", None)
        obs48 = (
            np.zeros(48, np.float32)
            if o is None
            else np.asarray(o[0].detach().cpu().numpy(), np.float32)
        )
        try:
            tq = backend._robot.data.applied_torque[0].detach().cpu().numpy()
            tau = np.asarray(tq, np.float32)
        except Exception:  # noqa: BLE001
            tau = np.zeros(12, np.float32)
        return np.concatenate([obs48, tau])

    def collect(floor: float, seed: int):
        rect = Rect(cx=3.0, cy=args.lane_y, hx=1.0, hy=1.0)
        op = EffortDecay(decay_rate_per_s=0.6, floor=floor, t_start_s=2.0)
        region = SemanticRegion(rect=rect, appearance_class="solid_ground")
        backend._start_pos = np.array([0.0, args.lane_y])
        backend._start_heading = 0.0
        obs = backend.deep_reset(seed)
        op.on_reset(backend)
        # fire LATE (t≈5) so the snapshot captures the FLOOR-differentiated effort (at t=2.2 the
        # decay just started ⇒ effort ~0.88 for all floors; by t=5 every floor reached its asymptote
        # asymptote ⇒ the snapshot encodes θ, so the agent can decide continue vs intervene on θ).
        oracle = OracleTrigger(onset_time_s=5.0, dt=dt, arm_delay_s=args.arm)
        recorder = SnapshotRecorder(
            hcfg,
            scene=[region],
            operator_name="O10_effort_decay",
            appearance_class="solid_ground",
            privileged_fn=backend.privileged_physics,
            gate_rect=None,
        )
        binding = deque(maxlen=int(args.binding_t))
        for _k in range(args.steps):
            binding.append(read60())
            obs_m = op.transform_obs(obs)
            ev = oracle.step(obs_m)
            recorder.observe(obs_m, ev)
            if recorder.snapshot is not None:
                break
            op.on_step(backend, obs.t)
            obs = backend.step(np.array([0.6, 0.0, 0.0]))
        snap = recorder.snapshot
        if snap is None:
            print(f"[a6.2] floor={floor}/s{seed}: NO fire; skip", flush=True)
            return None
        sid = f"a6o10_floor{floor:g}_s{seed}"
        priv = backend.privileged_physics()
        rec = {
            "sample_id": sid,
            "taxonomy_cell": "T4",
            "operator_name_sweep": "O10_effort_decay",
            "appearance_id": "solid_ground",
            "appearance_split": "train",
            "seed": seed,
            "a6_floor": floor,
            "snapshot": {
                **snap.to_meta(),
                "operator_name": "O10_effort_decay",
                "binding_shape": list(np.asarray(list(binding), np.float32).shape),
            },
            "ground_truth": {
                "category": "effort_decay",
                "theta": {"floor": floor, **{k: float(v) for k, v in priv.items()}},
            },
            "admissible_recovery_set": ["Switch_Gait", "Set_Constraint"],
            "target_theta": [
                priv.get("mu", 0.8),
                priv.get("payload_kg", 0.0),
                priv.get("effort_scale", 1.0),
                priv.get("support_ratio", 1.0),
            ],
            "annotation": None,
            "verdict": {"keep": True, "reason": "A6.2 O10 sweep"},
        }
        records.append(rec)
        with spath.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return {
            f"{sid}__rgb": snap.rgb.astype(np.float32),
            f"{sid}__depth": snap.depth.astype(np.float32),
            f"{sid}__proprio": snap.proprio_window.astype(np.float32),
            f"{sid}__binding": np.asarray(list(binding), np.float32),
        }

    n_fire = 0
    for floor in FLOORS:
        for si in seeds:
            fr = collect(floor, SEED_BASE + si)
            if fr:
                batch.update(fr)
                n_fire += 1
        print(f"[a6.2] floor={floor}: collected (fire {n_fire})", flush=True)
    np.savez_compressed(out_dir / "frames.npz", **batch)
    (out_dir / "collection_card.json").write_text(
        json.dumps(
            {
                "commit": git_commit(),
                "n_snapshots": n_fire,
                "floors": list(FLOORS),
                "seed_base": SEED_BASE,
                "deterministic": "deep_reset(A0.1) + fixed lane_y",
            },
            indent=2,
        )
    )
    print(f"\n[a6.2] DONE: {n_fire} O10-sweep snapshots → {out_dir}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
