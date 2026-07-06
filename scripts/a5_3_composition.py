#!/usr/bin/env python
"""A5.3 — compositional stacking snapshot collection (Paper-A §4 A5.3, serves C4).

Stacks operators per spec §8.3 (parameter-vector concatenation, NO new mechanisms) via
``OperatorStack`` and collects composed snapshots into a frozen extension. Two stacks:

  - **O1+O5** = MuField (ice region) + Payload (16 kg, global): the robot crosses ice UNDER LOAD ⇒
    two concurrent causes (low_friction AND overload). Multi-factor ground truth.
  - **O2+O10** = ComplianceField (mud region) + EffortDecay (global): mud + actuator decay ⇒
    compliant_terrain AND effort_decay.

Same deterministic harness as A0.3/A3 (deep_reset + fixed lane_y) + the A0.3 snapshot schema, so the
A2/A3 eval loaders consume it. The composed ground truth carries BOTH factor categories + the UNION
admissible set; A5.3 eval scores per-factor recall + primitive ∈ composed admissible.

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES \\
        ~/miniconda3/envs/kinovla/bin/python scripts/a5_3_composition.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import deque

import numpy as np

SEEDS_PER_CELL = 8
SEED_BASE = 900


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
    ap = argparse.ArgumentParser(description="A5.3 composition snapshot collection (real Go2)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--out", default="outputs/eval/a5/corpus_composition")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--binding-t", type=int, default=100)
    ap.add_argument("--arm", type=float, default=0.2)
    ap.add_argument("--cruise", type=float, default=0.6)
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from kino_vla.data.snapshot import SnapshotRecorder
    from kino_vla.eval.oracle_trigger import OracleTrigger
    from kino_vla.map.types import SemanticRegion
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import ComplianceField, EffortDecay, MuField, OperatorStack, Payload
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.geometry import Rect

    hcfg = load_config("data/hindsight.yaml")
    lane_y = 4.0
    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, lane_y]), 0.0)
    dt = backend.dt
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    spath = out_dir / "samples.jsonl"
    if spath.exists():
        spath.unlink()
    records: list[dict] = []
    batch: dict[str, np.ndarray] = {}
    seeds = range(1 if args.quick else SEEDS_PER_CELL)

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

    # the two stacks: (name, stack, scene_region, factor_categories, composed_admissible, theta_tag)
    rect = Rect(cx=3.0, cy=lane_y, hx=1.0, hy=1.0)
    stacks = [
        {
            "name": "O1_O5_ice_payload",
            "ops": OperatorStack(
                [
                    MuField(region=rect, mu_s=0.10, mu_d=0.08),
                    Payload(mass_kg=3.0, com_offset_m=(0.0, 0.0)),
                ]
            ),
            "region": SemanticRegion(rect=rect, appearance_class="ice_sheet"),
            "factors": ["low_friction", "overload"],
            "admissible": ["Set_Constraint", "Adjust_Posture", "Hold_and_Request"],
        },
        {
            "name": "O2_O10_mud_decay",
            "ops": OperatorStack(
                [
                    ComplianceField(rect, k_c=14.0, c_c=6.0, d_sink=0.08),
                    EffortDecay(decay_rate_per_s=0.6, floor=0.30, t_start_s=2.0),
                ]
            ),
            "region": SemanticRegion(rect=rect, appearance_class="brown_mud"),
            "factors": ["compliant_terrain", "effort_decay"],
            "admissible": ["Switch_Gait", "Set_Constraint"],
        },
    ]

    def collect(stk, seed):
        backend._start_pos = np.array([0.0, lane_y])
        backend._start_heading = 0.0
        obs = backend.deep_reset(seed)
        stk["ops"].on_reset(backend)
        # region-entry oracle on the (first) region rect — fires when the trunk enters the patch
        oracle = OracleTrigger(rect=stk["region"].rect, dt=dt, arm_delay_s=args.arm)
        recorder = SnapshotRecorder(
            hcfg,
            scene=[stk["region"]],
            operator_name=stk["name"],
            appearance_class=stk["region"].appearance_class,
            privileged_fn=backend.privileged_physics,
            gate_rect=None,
        )
        binding = deque(maxlen=int(args.binding_t))
        for _k in range(args.steps):
            binding.append(read60())
            obs_m = stk["ops"].transform_obs(obs)
            ev = oracle.step(obs_m)
            recorder.observe(obs_m, ev)
            if recorder.snapshot is not None:
                break
            stk["ops"].on_step(backend, obs.t)
            obs = backend.step(np.array([args.cruise, 0.0, 0.0]))
        snap = recorder.snapshot
        if snap is None:
            print(f"[a5.3] {stk['name']}/s{seed}: NO fire; skip", flush=True)
            return None
        sid = f"a5c_{stk['name']}_s{seed}"
        priv = backend.privileged_physics()
        rec = {
            "sample_id": sid,
            "taxonomy_cell": "COMP",
            "appearance_id": stk["region"].appearance_class,
            "appearance_split": "test",
            "seed": seed,
            "operator_name_composed": stk["name"],
            "factors": stk["factors"],  # BOTH causes (multi-factor truth)
            "admissible_recovery_set": sorted(stk["admissible"]),  # UNION admissible
            "ground_truth": {
                "category": stk["factors"][0],
                "factors": stk["factors"],
                "theta": {k: float(v) for k, v in priv.items()},
            },
            "snapshot": {
                **snap.to_meta(),
                "operator_name": stk["name"],
                "binding_shape": list(np.asarray(list(binding), np.float32).shape),
            },
            "annotation": None,
            "verdict": {"keep": True, "reason": "A5.3 composition"},
            "target_theta": [
                priv.get("mu", 0.8),
                priv.get("payload_kg", 0.0),
                priv.get("effort_scale", 1.0),
                priv.get("support_ratio", 1.0),
            ],
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
    for stk in stacks:
        for si in seeds:
            fr = collect(stk, SEED_BASE + si)
            if fr:
                batch.update(fr)
                n_fire += 1
        print(f"[a5.3] {stk['name']}: collected (fire {n_fire})", flush=True)
    np.savez_compressed(out_dir / "frames.npz", **batch)
    card = {
        "commit": git_commit(),
        "n_snapshots": n_fire,
        "stacks": [s["name"] for s in stacks],
        "factors": {s["name"]: s["factors"] for s in stacks},
        "seed_base": SEED_BASE,
        "deterministic": "deep_reset(A0.1) + fixed lane_y",
    }
    (out_dir / "collection_card.json").write_text(json.dumps(card, indent=2))
    print(f"\n[a5.3] DONE: {n_fire} composed snapshots → {out_dir}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
