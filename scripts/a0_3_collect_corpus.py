#!/usr/bin/env python
"""A0.3 — Frozen snapshot-corpus collection (real Go2, deterministic). Paper-A §2 A0.3.

Scales the E2 n=30 ``matched_eval`` to the Paper-A corpus: every taxonomy scenario × its appearance
library (A0.4) × disjoint seeds, captured at the privileged oracle onset (A0.2), on the
DETERMINISTIC backend (A0.1 ``deep_reset`` between every lane in ONE reused app → each lane is
byte-identical and order-independent). Each snapshot carries the richer schema:

  [5×RGB, 5×depth, proprio-11 window (VLA), binding obs48⊕τ12 window T≤100 (C2ST/B1), privileged θ,
   true category + ab_class, admissible set (A0.5), taxonomy cell, appearance-id, pair-id, seed]

Writes ``samples.jsonl`` (extended record; the E2 ``load_suite_sem`` contract is preserved — base
fields + ``{sid}__{rgb,depth,proprio}`` npz keys are unchanged, new fields additive) + per-scenario
``frames_<scenario>.npz`` (durable batches; the freeze step consolidates). Every number traces to
(config hash, seeds, commit). Run scripts/a0_3_freeze.py afterwards to hash + card the corpus.

Run:  python scripts/a0_3_collect_corpus.py --headless [--quick] [--out outputs/eval/a0/corpus]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import deque

import numpy as np

# --- collection plan: registry scenario -> seeds. Appearances come from the A0.4 library for the
#     scenario's appearance class. The load-bearing matched pair is collected ≥100 (per §7). ---
FULL_PLAN: dict[str, int] = {
    "matched_O4": 24,       # × 5 adhesion appearances = 120 (T2 headline)
    "matched_O2": 30,       # × 4 compliant appearances = 120 (T1 headline control)
    "O1_ice": 16,           # × 3 ice = 48 (T1)
    "O8_invisible": 12,     # × 4 solid = 48 (T3)
    "O5_payload_B": 12,     # × 4 solid = 48 (T4)
    "O10_decay_B": 12,      # × 4 solid = 48 (T4)
    "O1_A_nominal": 12,     # × 3 ice = 36 (T5)
    "O2_A_nominal": 10,     # × 4 compliant = 40 (T5)
    "O6_push_A": 10,        # × 4 solid = 40 (T5)
}
QUICK_PLAN: dict[str, int] = {"matched_O4": 1, "matched_O2": 1}  # smoke: 5+4 lanes
SEED_BASE = 500  # disjoint from all training seeds (0–9), per §7


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.3 frozen snapshot-corpus collection")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--out", default="outputs/eval/a0/corpus")
    ap.add_argument("--quick", action="store_true", help="tiny smoke plan (matched pair, 1 seed)")
    ap.add_argument("--only", default="", help="comma-separated scenario names to restrict")
    ap.add_argument("--cruise", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--binding-t", type=int, default=100, help="binding obs48⊕τ window length")
    ap.add_argument("--arm", type=float, default=0.2, help="oracle arm delay (s)")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from kino_vla.data.snapshot import SnapshotRecorder
    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.appearance_library import load_appearance_library
    from kino_vla.eval.oracle_trigger import OracleTrigger
    from kino_vla.eval.registry import load_registry
    from kino_vla.map.types import SemanticRegion
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import OperatorStack
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla import scenarios as S

    reg = load_registry()
    lib = load_appearance_library(register=True)  # registers appearance colours into the §7 render
    tax = FailureTaxonomy(load_config("data/hindsight.yaml"))
    hcfg = load_config("data/hindsight.yaml")

    lane_y = 4.0  # single canonical lane; deep_reset despawns ⇒ fixed geometry deterministic.
    # Every scenario is BUILT at lane_y so its hazard region + oracle rect align with the drive.
    builders = {
        "matched_O4": lambda y: S.o4_tether_matched(y),
        "matched_O2": lambda y: S.o2_compliance_matched(y),
        "O1_ice": lambda y: S.o1_ice(y),
        "O8_invisible": lambda y: S.o8_invisible(y),
        "O5_payload_B": lambda y: S.o5_payload(y),
        "O10_decay_B": lambda y: S.o10_effort_decay(y),
        "O1_A_nominal": lambda y: S.o1_ice_A(y),
        "O2_A_nominal": lambda y: S.o2_compliance_A(y),
        "O6_push_A": lambda y: S.o6_push_A(y),
    }
    plan = QUICK_PLAN if args.quick else dict(FULL_PLAN)
    if args.only:
        keep = set(args.only.split(","))
        plan = {k: v for k, v in plan.items() if k in keep}

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt

    def read60() -> np.ndarray:
        o = getattr(backend, "_obs", None)
        obs48 = (np.zeros(48, np.float32) if o is None
                 else np.asarray(o[0].detach().cpu().numpy(), np.float32))
        try:
            tau = np.asarray(backend._robot.data.applied_torque[0].detach().cpu().numpy(),
                              np.float32)
        except Exception:  # noqa: BLE001
            tau = np.zeros(12, np.float32)
        return np.concatenate([obs48, tau])

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    if not args.quick and not args.only and samples_path.exists():
        samples_path.unlink()  # a full fresh run wipes; a targeted --only run APPENDS (recollect)
    records: list[dict] = []
    n_total = n_fire = 0

    for scn_name, n_seeds in plan.items():
        spec = reg[scn_name]
        base_scn = builders[scn_name](lane_y)
        rect = base_scn.scene_region.rect
        # the appearances to draw from: the library class of this scenario's canonical appearance
        canonical_app = spec.appearance_class
        sem_class = lib.semantic_class(canonical_app) if canonical_app in lib._appearances else None
        if sem_class is None:  # canonical not a library id (e.g. yellow_adhesive) ⇒ map by name
            sem_class = {"yellow_adhesive": "adhesion", "brown_mud": "compliant_terrain",
                         "ice_sheet": "low_friction", "solid_ground": "solid_ground"}[canonical_app]
        appearances = lib.appearances(sem_class)
        gt = tax.ground_truth(spec.operator, spec.theta)
        batch_frames: dict[str, np.ndarray] = {}
        for app in appearances:
            for si in range(n_seeds):
                seed = SEED_BASE + si
                n_total += 1
                scene_region = SemanticRegion(rect=rect, appearance_class=app.id)
                ops = OperatorStack([base_scn.operator])
                backend._start_pos = np.array([0.0, lane_y])
                backend._start_heading = 0.0
                obs = backend.deep_reset(seed)
                ops.on_reset(backend)
                oracle = OracleTrigger.for_scenario(base_scn, dt=dt, arm_delay_s=args.arm)
                recorder = SnapshotRecorder(
                    hcfg, scene=[scene_region], operator_name=spec.operator,
                    appearance_class=app.id, privileged_fn=backend.privileged_physics,
                    gate_rect=None,
                )
                binding = deque(maxlen=int(args.binding_t))
                for _k in range(args.steps):
                    binding.append(read60())
                    obs_m = ops.transform_obs(obs)
                    ev = oracle.step(obs_m)
                    recorder.observe(obs_m, ev)
                    if recorder.snapshot is not None:
                        break
                    ops.on_step(backend, obs.t)
                    obs = backend.step(np.array([args.cruise, 0.0, 0.0]))
                snap = recorder.snapshot
                if snap is None:
                    print(f"[a0.3] {scn_name}/{app.id}/s{seed}: NO oracle fire; skip", flush=True)
                    continue
                n_fire += 1
                sid = f"a0corpus_{scn_name}_{app.id}_s{seed}"
                bind_arr = np.asarray(list(binding), np.float32)  # (T, 60)
                batch_frames[f"{sid}__rgb"] = snap.rgb.astype(np.float32)
                batch_frames[f"{sid}__depth"] = snap.depth.astype(np.float32)
                batch_frames[f"{sid}__proprio"] = snap.proprio_window.astype(np.float32)
                batch_frames[f"{sid}__binding"] = bind_arr
                is_pair = spec.true_category in ("adhesion", "compliant_terrain") and (
                    spec.taxonomy_cell in ("T1", "T2"))
                pair_id = "O4_tether|O2_compliance" if is_pair else None
                # scripted privileged annotation (non-nominal): attribution = true category,
                # primitive = A0.5 canonical. Nominal (`continue`) rows carry ground truth only.
                annotation = None
                if not spec.is_nominal and spec.canonical_recovery.primitive != "continue":
                    annotation = {
                        "thought": f"privileged: {spec.true_category} ({spec.taxonomy_cell})",
                        "attribution": spec.true_category,
                        "attribution_raw": spec.true_category,
                        "action": {"primitive": spec.canonical_recovery.primitive,
                                   "params": dict(spec.canonical_recovery.params)},
                    }
                rec = {
                    "sample_id": sid,
                    "taxonomy_cell": spec.taxonomy_cell,
                    "appearance_id": app.id,
                    "appearance_split": app.split,
                    "seed": seed,
                    "pair_id": pair_id,
                    "ambiguity_pair": pair_id,
                    "success_criterion": spec.success_criterion,
                    "admissible_recovery_set": sorted(spec.admissible_recovery_set),
                    "snapshot": {**snap.to_meta(), "binding_shape": list(bind_arr.shape)},
                    "ground_truth": gt.to_dict(),
                    "annotation": annotation,
                    "verdict": {"keep": True, "reason": "keep", "detail": "A0.3 privileged label"},
                    "target_theta": [float(snap.privileged_theta.get("mu", 0.0)),
                                     float(snap.privileged_theta.get("payload_kg", 0.0)),
                                     float(snap.privileged_theta.get("effort_scale", 1.0)),
                                     float(snap.privileged_theta.get("support_ratio", 1.0))],
                }
                records.append(rec)
                with samples_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        if batch_frames:
            np.savez_compressed(out_dir / f"frames_{scn_name}.npz", **batch_frames)
        print(f"[a0.3] {scn_name}: {len(appearances)} appearances × {n_seeds} seeds → "
              f"{len(batch_frames)//4} snapshots (fire so far {n_fire}/{n_total})", flush=True)

    card = {
        "commit": git_commit(), "seed_base": SEED_BASE, "cruise": args.cruise, "dt": float(dt),
        "binding_t": args.binding_t, "arm_s": args.arm, "n_snapshots": n_fire, "n_lanes": n_total,
        "plan": plan, "deterministic": "deep_reset(A0.1) + fixed lane_y=4.0",
    }
    (out_dir / "collection_card.json").write_text(json.dumps(card, indent=2))
    print(f"\n[a0.3] DONE: {n_fire}/{n_total} snapshots → {out_dir}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
