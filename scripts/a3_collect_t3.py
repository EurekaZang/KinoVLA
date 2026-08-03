#!/usr/bin/env python
"""A3 — collect the T3 (visual-physics remap) snapshot extension to the frozen corpus (real Go2).

A0.3 froze T1/T2/T4/T5 + O8; T3's O7 was pending its scenario builder (now added:
``o7_visual_remap`` / ``o7_visual_remap_reverse``). This collects the bidirectional T3 battery into
a SEPARATE frozen dir (``outputs/eval/a3/corpus_t3``) so A0.3 stays untouched; A3 eval then merges
them for the full taxonomy × agents heatmap.

Two directions (the bidirectional conflict battery, A3.1/A3.2):

- **O7 looks_safe** (T3, proprio-true): ``solid_ground`` appearance hides a low-friction patch; a
  μ-SWEEP {0.6, 0.3, 0.15, 0.09} feeds BOTH the heatmap (μ=0.09 = canonical low_friction) AND the
  A3.4 dose–response (P(override) vs proprio evidence). Proprio detects the slip; vision is fooled.
- **O7 reverse** (T3 reverse probe, A3.2): hazard-coloured decal on NOMINAL floor (μ=0.8) → correct
  answer ``continue``. Catches a "conflict ⇒ trust camera" vision-dominance shortcut.

Same deterministic harness as A0.3 (deep_reset + fixed lane_y=4.0) + the A0.3 snapshot schema, so
the A2/A3 eval loaders consume it unchanged. Run scripts/a3_freeze_t3.py afterwards to hash + card.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import deque

import numpy as np

# μ sweep for O7 looks_safe (A3.4 dose–response). 0.09 = the canonical low_friction point (heatmap).
O7_MU_SWEEP = (0.6, 0.3, 0.15, 0.09)
SEEDS_PER_CELL = 8           # × appearances; T3 auxiliary (T2 pair carries the ≥100 headline)
SEED_BASE = 500              # disjoint from training 0–9, matches the A0.3 corpus


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="A3 T3 (O7 visual-physics remap) corpus collection")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--out", default="outputs/eval/a3/corpus_t3")
    ap.add_argument("--appearance-config", default="eval/appearance_library.yaml")
    ap.add_argument("--directions", choices=("all", "looks_safe", "reverse"), default="all")
    ap.add_argument("--appearance-split", choices=("all", "train", "test"), default="all")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--seeds-per-cell", type=int, default=SEEDS_PER_CELL)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cruise", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--binding-t", type=int, default=100)
    ap.add_argument("--arm", type=float, default=0.2)
    # Bang-bang square-wave excitation (the M4/hindsight drive, hindsight.yaml drive:*). A flat
    # cruise never excites the O7 slip enough for the proprio channel to carry it (A1.4 bang-bang
    # O7 cnn1d AUC 1.0 vs the A3 cruise failure) -- bang-bang puts a strong slip in the snapshot
    # window AND matches the B1 monitor's bang-bang training distribution (train/test gap closed).
    ap.add_argument("--bang-bang", action="store_true",
                    help="square-wave speed excitation (M4 drive) instead of flat cruise")
    ap.add_argument("--speed-lo", type=float, default=0.1)
    ap.add_argument("--speed-hi", type=float, default=1.0)
    ap.add_argument("--speed-period", type=float, default=1.0)
    ap.add_argument("--speed-duty", type=float, default=0.5)
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from kino_vla.data.snapshot import SnapshotRecorder
    from kino_vla.eval.appearance_library import load_appearance_library
    from kino_vla.eval.oracle_trigger import OracleTrigger
    from kino_vla.eval.registry import load_registry
    from kino_vla.map.types import SemanticRegion
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import OperatorStack
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla import scenarios as S

    reg = load_registry()
    lib = load_appearance_library(args.appearance_config, register=True)
    hcfg = load_config("data/hindsight.yaml")
    lane_y = 4.0

    _APP_CLASS_FALLBACK = {"yellow_adhesive": "adhesion", "brown_mud": "compliant_terrain",
                           "ice_sheet": "low_friction", "solid_ground": "solid_ground"}

    def appearances_for(canonical_app: str):
        """The library appearances to sweep for a scenario's appearance_class. Handles both
        appearance-ids and class-names (e.g. hazard_decal_benign, a class not an id)."""
        if canonical_app in lib._appearances:
            values = lib.appearances(lib.semantic_class(canonical_app))
        elif canonical_app in lib.classes():         # it IS a class name (hazard_decal_benign)
            values = lib.appearances(canonical_app)
        else:
            values = lib.appearances(_APP_CLASS_FALLBACK[canonical_app])
        if args.appearance_split != "all":
            values = [item for item in values if item.split == args.appearance_split]
        return values

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt

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
    if samples_path.exists():
        samples_path.unlink()
    records: list[dict] = []
    n_total = n_fire = 0

    def collect(scn_name: str, base_scn, spec, app_id: str, seed: int, *, mu_label: float | None):
        nonlocal n_total, n_fire
        n_total += 1
        rect = base_scn.scene_region.rect
        scene_region = SemanticRegion(rect=rect, appearance_class=app_id)
        ops = OperatorStack([base_scn.operator])
        backend._start_pos = np.array([0.0, lane_y])
        backend._start_heading = 0.0
        obs = backend.deep_reset(seed)
        ops.on_reset(backend)
        oracle = OracleTrigger.for_scenario(base_scn, dt=dt, arm_delay_s=args.arm)
        recorder = SnapshotRecorder(
            hcfg, scene=[scene_region], operator_name=spec.operator,
            appearance_class=app_id, privileged_fn=backend.privileged_physics, gate_rect=None,
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
            if args.bang_bang:  # square-wave speed (M4 drive): hi then lo each period
                frac = ((obs.t % args.speed_period) / args.speed_period)
                spd = args.speed_hi if frac < args.speed_duty else args.speed_lo
            else:
                spd = args.cruise
            obs = backend.step(np.array([spd, 0.0, 0.0]))
        snap = recorder.snapshot
        if snap is None:
            print(f"[a3] {scn_name}/{app_id}/s{seed}: NO oracle fire; skip", flush=True)
            return
        n_fire += 1
        mu = mu_label if mu_label is not None else float(spec.theta.get("mu_s", 0.0))
        sid = (
            f"a3t3_{scn_name}_{app_id}_mu{mu:g}_s{seed}" if mu_label is not None
            else f"a3t3_{scn_name}_{app_id}_s{seed}"
        )
        bind_arr = np.asarray(list(binding), np.float32)
        # ground truth: the REGISTRY is the pre-registered source of truth (R4/R5). For O7_reverse
        # (nominal) the registry category is `nominal` (tax.ground_truth keys on operator name and
        # would wrongly say low_friction); use spec.true_category + spec.admissible set verbatim.
        is_nominal = spec.is_nominal
        gt_cat = spec.true_category
        annotation = None
        if not is_nominal and spec.canonical_recovery.primitive != "continue":
            annotation = {
                "thought": f"privileged: {gt_cat} ({spec.taxonomy_cell})",
                "attribution": gt_cat,
                "attribution_raw": gt_cat,
                "action": {"primitive": spec.canonical_recovery.primitive,
                           "params": dict(spec.canonical_recovery.params)},
            }
        rec = {
            "sample_id": sid,
            "taxonomy_cell": spec.taxonomy_cell,
            "appearance_id": app_id,
            "appearance_split": lib.split(app_id) if app_id in lib._appearances else "test",
            "seed": seed,
            "pair_id": None,
            "ambiguity_pair": None,
            "success_criterion": spec.success_criterion,
            "admissible_recovery_set": sorted(spec.admissible_recovery_set),
            "snapshot": {**snap.to_meta(), "binding_shape": list(bind_arr.shape)},
            "ground_truth": {"category": gt_cat, "ab_class": spec.ab_class,
                             "theta": {**spec.theta, "mu": mu}},
            "annotation": annotation,
            "verdict": {"keep": True, "reason": "A3 T3 privileged label", "detail": scn_name},
            "target_theta": [mu, 0.0, 1.0, 1.0],
            "a3_mu": mu,                       # the dose–response abscissa (A3.4)
            "a3_direction": "looks_safe" if mu_label is not None else "reverse",
        }
        records.append(rec)
        with samples_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return {f"{sid}__rgb": snap.rgb.astype(np.float32),
                f"{sid}__depth": snap.depth.astype(np.float32),
                f"{sid}__proprio": snap.proprio_window.astype(np.float32),
                f"{sid}__binding": bind_arr}

    batch: dict[str, np.ndarray] = {}
    seeds = range(1 if args.quick else args.seeds_per_cell)

    # O7 looks_safe: μ sweep (dose–response + the μ=0.09 heatmap point)
    if args.directions in {"all", "looks_safe"}:
        spec = reg["O7_looks_safe"]
        for mu in (O7_MU_SWEEP[:1] if args.quick else O7_MU_SWEEP):
            for app in appearances_for(spec.appearance_class):
                for si in seeds:
                    scn = S.o7_visual_remap(lane_y, mu_s=mu, mu_d=mu)
                    fr = collect(
                        "O7_looks_safe",
                        scn,
                        spec,
                        app.id,
                        args.seed_base + si,
                        mu_label=mu,
                    )
                    if fr:
                        batch.update(fr)
            print(f"[a3] O7_looks_safe μ={mu}: collected (fire {n_fire}/{n_total})", flush=True)

    # O7 reverse probe (hazard decal, nominal → continue)
    if args.directions in {"all", "reverse"}:
        spec = reg["O7_reverse"]
        for app in appearances_for(spec.appearance_class):
            for si in seeds:
                scn = S.o7_visual_remap_reverse(lane_y, appearance_class=app.id)
                fr = collect(
                    "O7_reverse", scn, spec, app.id, args.seed_base + si, mu_label=None
                )
                if fr:
                    batch.update(fr)
        print(f"[a3] O7_reverse: collected (fire {n_fire}/{n_total})", flush=True)

    np.savez_compressed(out_dir / "frames.npz", **batch)
    card = {
        "commit": git_commit(), "seed_base": args.seed_base, "cruise": args.cruise,
        "dt": float(dt), "appearance_config": args.appearance_config,
        "appearance_split": args.appearance_split, "directions": args.directions,
        "drive": "bang-bang(M4 speed_hi/lo)" if args.bang_bang else f"cruise({args.cruise})",
        "binding_t": args.binding_t, "arm_s": args.arm, "n_snapshots": n_fire, "n_lanes": n_total,
        "o7_mu_sweep": list(O7_MU_SWEEP), "seeds_per_cell": args.seeds_per_cell,
        "deterministic": "deep_reset(A0.1) + fixed lane_y=4.0",
        "plan": {
            "O7_looks_safe": f"μ∈{O7_MU_SWEEP} × solid_ground × {args.seeds_per_cell} seeds",
            "O7_reverse": f"hazard_decal_benign × {args.seeds_per_cell} seeds",
        },
    }
    (out_dir / "collection_card.json").write_text(json.dumps(card, indent=2))
    print(f"\n[a3] DONE: {n_fire}/{n_total} T3 snapshots → {out_dir}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
