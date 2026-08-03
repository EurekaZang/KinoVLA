#!/usr/bin/env python
"""A2 — VLA-faithful matched-pair corpus collection (real Go2). Paper-A §4 A2.

Supersedes the A0.3 oracle-onset corpus FOR THE VLA PATH. The A0.3 corpus fails the VLA on two
counts (diagnosed 2026-07-03, see A实验/A2.md §6): (1) it captures at oracle-onset (region-entry +
0.2 s) — BEFORE the adhesion resistance builds — so the proprio-vs-vision conflict the VLA resolves
is not yet present (B5-unshaped and B5-conflict collapse to identical accuracy); (2) deep_reset +
fixed geometry + DR-off yields ZERO seed diversity (24 byte-identical copies per appearance), so
n=120 is pseudo-replication. This collector fixes both while keeping the A0.3 rich schema + A0.4
appearance library + A0.5 registry:

  * TRIGGER = the deployed LearnedMonitor intercept (the SAME point the VLA was trained + deployed
    on, conflict present), not the oracle. The monitor is agent-independent (a fixed model), so the
    fair-comparison intent of R3 is preserved; it is scoped infrastructure, not a contribution.
  * DIVERSITY = per-seed start-pose jitter (x, y, heading). PhysX float determinism depends on the
    ABSOLUTE start coordinates (A0.1 finding), so a seeded jitter makes each seed a genuinely
    different crossing — while O4 and O2 sharing a seed get the SAME jitter + the matched force law,
    so they stay byte-identical to each other (C1 preserved; re-verified by scripts/a2_matched_
    validity.py on the output).
  * DETERMINISM = deep_reset(seed) between lanes ⇒ order-independent + reproducible
    (jitter = f(seed)).

Writes the same load_matched_corpus schema (samples.jsonl + frames_<scn>.npz) as A0.3.

Run:  env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES ~/miniconda3/envs/kinovla/bin/python \
        scripts/a2_collect.py --headless --seeds 24 --out outputs/eval/a2/corpus
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import deque

import numpy as np

SEED_BASE = 500  # eval seeds, disjoint from all training seeds (0–9)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="A2 VLA-faithful matched-pair collection")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--out", default="outputs/eval/a2/corpus")
    ap.add_argument("--seeds", type=int, default=24, help="seeds per (operator, appearance)")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--only", default="", help="restrict to matched_O4 / matched_O2")
    ap.add_argument("--cruise", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--binding-t", type=int, default=100)
    ap.add_argument("--jitter-xy", type=float, default=0.35, help="per-seed start x/y jitter (m)")
    ap.add_argument("--jitter-hdg", type=float, default=0.12, help="per-seed heading jitter (rad)")
    ap.add_argument("--smoke", action="store_true", help="2 seeds, matched pair only")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from kino_vla.data.snapshot import SnapshotRecorder
    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.appearance_library import load_appearance_library
    from kino_vla.eval.registry import load_registry
    from kino_vla.map.types import SemanticRegion
    from kino_vla.monitor.learned_monitor import load_deployed_monitor
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import OperatorStack
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla import scenarios as S

    reg = load_registry()
    lib = load_appearance_library(register=True)
    tax = FailureTaxonomy(load_config("data/hindsight.yaml"))
    hcfg = load_config("data/hindsight.yaml")

    lane_y = 4.0
    builders = {
        "matched_O4": (lambda y: S.o4_tether_matched(y), "adhesion"),
        "matched_O2": (lambda y: S.o2_compliance_matched(y), "compliant_terrain"),
    }
    plan = {"matched_O4": args.seeds, "matched_O2": args.seeds}
    if args.smoke:
        plan = {"matched_O4": 2, "matched_O2": 2}
    if args.only:
        plan = {k: v for k, v in plan.items() if k in set(args.only.split(","))}

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    monitor = load_deployed_monitor(dt)

    def read60() -> np.ndarray:
        o = getattr(backend, "_obs", None)
        obs48 = (np.zeros(48, np.float32) if o is None
                 else np.asarray(o[0].detach().cpu().numpy(), np.float32))
        try:
            tq = backend._robot.data.applied_torque[0].detach().cpu().numpy()
            tau = np.asarray(tq, np.float32)
        except Exception:  # noqa: BLE001
            tau = np.zeros(12, np.float32)
        return np.concatenate([obs48, tau])

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    if not args.only and samples_path.exists():
        samples_path.unlink()
    n_total = n_fire = 0

    for scn_name, n_seeds in plan.items():
        build, sem_class = builders[scn_name]
        spec = reg[scn_name]
        base_scn = build(lane_y)
        rect = base_scn.scene_region.rect
        appearances = lib.appearances(sem_class)
        gt = tax.ground_truth(spec.operator, spec.theta)
        batch_frames: dict[str, np.ndarray] = {}
        for app in appearances:
            for si in range(n_seeds):
                seed = args.seed_base + si
                n_total += 1
                # per-seed start jitter (deterministic f(seed)) — genuine cross-seed diversity via
                # absolute-coordinate PhysX chaos, identical for O4 & O2 at the same seed (matched).
                jr = np.random.default_rng(seed)
                jx = float(jr.uniform(-args.jitter_xy, args.jitter_xy))
                jy = float(jr.uniform(-args.jitter_xy, args.jitter_xy))
                jh = float(jr.uniform(-args.jitter_hdg, args.jitter_hdg))
                scene_region = SemanticRegion(rect=rect, appearance_class=app.id)
                ops = OperatorStack([base_scn.operator])
                backend._start_pos = np.array([float(base_scn.start_xy[0]) + jx, lane_y + jy])
                backend._start_heading = float(base_scn.start_heading) + jh
                obs = backend.deep_reset(seed)
                ops.on_reset(backend)
                monitor.reset()
                recorder = SnapshotRecorder(
                    hcfg, scene=[scene_region], operator_name=spec.operator,
                    appearance_class=app.id, privileged_fn=backend.privileged_physics,
                    gate_rect=None,
                )
                binding: deque = deque(maxlen=int(args.binding_t))
                for _k in range(args.steps):
                    binding.append(read60())
                    obs_m = ops.transform_obs(obs)
                    ev = monitor.step(obs_m)  # deployed-monitor intercept (conflict present)
                    recorder.observe(obs_m, ev)
                    if recorder.snapshot is not None:
                        break
                    ops.on_step(backend, obs.t)
                    obs = backend.step(np.array([args.cruise, 0.0, 0.0]))
                snap = recorder.snapshot
                if snap is None:
                    print(
                        f"[a2.collect] {scn_name}/{app.id}/s{seed}: NO monitor fire; skip",
                        flush=True,
                    )
                    continue
                n_fire += 1
                sid = f"a2_{scn_name}_{app.id}_s{seed}"
                bind_arr = np.asarray(list(binding), np.float32)
                batch_frames[f"{sid}__rgb"] = snap.rgb.astype(np.float32)
                batch_frames[f"{sid}__depth"] = snap.depth.astype(np.float32)
                batch_frames[f"{sid}__proprio"] = snap.proprio_window.astype(np.float32)
                batch_frames[f"{sid}__binding"] = bind_arr
                pair_id = "O4_tether|O2_compliance"
                annotation = {
                    "thought": f"privileged: {spec.true_category} ({spec.taxonomy_cell})",
                    "attribution": spec.true_category,
                    "attribution_raw": spec.true_category,
                    "action": {"primitive": spec.canonical_recovery.primitive,
                               "params": dict(spec.canonical_recovery.params)},
                }
                rec = {
                    "sample_id": sid, "taxonomy_cell": spec.taxonomy_cell,
                    "appearance_id": app.id, "appearance_split": app.split, "seed": seed,
                    "pair_id": pair_id, "ambiguity_pair": pair_id,
                    "jitter": {"x": jx, "y": jy, "hdg": jh},
                    "success_criterion": spec.success_criterion,
                    "admissible_recovery_set": sorted(spec.admissible_recovery_set),
                    "snapshot": {**snap.to_meta(), "binding_shape": list(bind_arr.shape)},
                    "ground_truth": gt.to_dict(), "annotation": annotation,
                    "verdict": {"keep": True, "reason": "keep", "detail": "A2 monitor-intercept"},
                    "target_theta": [float(snap.privileged_theta.get("mu", 0.0)),
                                     float(snap.privileged_theta.get("payload_kg", 0.0)),
                                     float(snap.privileged_theta.get("effort_scale", 1.0)),
                                     float(snap.privileged_theta.get("support_ratio", 1.0))],
                }
                with samples_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        if batch_frames:
            np.savez_compressed(out_dir / f"frames_{scn_name}.npz", **batch_frames)
        print(f"[a2.collect] {scn_name}: {len(appearances)} appearances × {n_seeds} seeds → "
              f"{len(batch_frames)//4} snapshots (fire {n_fire}/{n_total})", flush=True)

    card = {
        "commit": git_commit(), "seed_base": args.seed_base, "cruise": args.cruise, "dt": float(dt),
        "trigger": "deployed_learned_monitor_intercept", "n_snapshots": n_fire, "n_lanes": n_total,
        "jitter_xy": args.jitter_xy, "jitter_hdg": args.jitter_hdg, "plan": plan,
        "note": "VLA-faithful matched corpus (monitor-intercept + per-seed jitter); supersedes "
                "the A0.3 oracle-onset corpus for the VLA path.",
    }
    (out_dir / "collection_card.json").write_text(json.dumps(card, indent=2))
    print(f"\n[a2.collect] DONE: {n_fire}/{n_total} snapshots → {out_dir}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
