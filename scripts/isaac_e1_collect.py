#!/usr/bin/env python
"""E1 — collect REAL-Go2 observation+history traces for the proprioceptive-indistinguishability
classifier-two-sample test (spec §8.1 P4, strengthened to the joint window distribution).

For each matched ambiguity pair (O4↔O2, O5↔O10, O3↔O1) + the power control (O2 vs O1), drive the
trained policy at the deployment cruise straight through the patch and log, per control step, the
THREE proprioception representations the C2ST tests over:

  * obs48  — the 48-dim policy observation (the raw signal a pure-proprioception RMA baseline sees)
  * tau12  — the 12 raw joint torques (current-inverted effort; the binding claim is obs48+tau)
  * feat12 — the 12-dim deployed monitor feature window (slip/effort/support + kinematics)

tagged by the operator (unit key) and lane. The matched θ live in configs/eval/e1_c2st.yaml; the
pair members reuse hazard_lab.build_scenario (so the onset machinery — region / payload / effort —
is the same tested code the learned monitor collects on). The O4 #49 peel-plateau knob is applied
only when ``o4_shaping.enabled`` (default OFF ⇒ the tether is byte-identical).

Real-stack only (CLAUDE.md §0): one IsaacPolicyBackend, one lane per (unit × seed) at its own y
(Isaac reuses the backend across reset()s, as in scripts/isaac_monitor_data_collect.py).

Run:  python scripts/isaac_e1_collect.py --headless --seeds 8 --tag train
      python scripts/isaac_e1_collect.py --headless --seeds 4 --seed-base 500 --tag test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

LANE_SPACING_M = 4.0


def git_commit() -> str:
    """Short git commit hash for result traceability (QA 5.2); 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------------------
# Plan: flatten the config's pairs + control into deduplicated collection units.
# --------------------------------------------------------------------------------------
def _coerce_theta(theta: dict) -> dict:
    return {k: float(v) for k, v in theta.items()}


def _unit_key(op_id: str, theta: dict, shaping) -> str:
    th = ",".join(f"{k}={theta[k]:g}" for k in sorted(theta))
    sh = "shape" if (shaping and shaping.get("enabled")) else "raw"
    return f"{op_id}[{th}]({sh})"


def e1_plan(cfg) -> tuple[dict, list, dict]:
    """Return (units, pairs, control). ``units`` maps a dedup key -> {op_id, theta, shaping}; a
    pair/control references its members by key so identical (op, θ) configs are collected once."""
    d = cfg.to_dict()
    shaping = d["o4_shaping"]
    units: dict[str, dict] = {}

    def add(op: dict, use_shaping: bool) -> str:
        theta = _coerce_theta(op["theta"])
        sh = shaping if use_shaping else None
        key = _unit_key(op["id"], theta, sh)
        units.setdefault(key, {"op_id": op["id"], "theta": theta, "shaping": sh})
        return key

    pairs = []
    for p in d["pairs"]:
        ka = add(p["op_a"], use_shaping=(p["name"] == "O4_O2" and p["op_a"]["id"] == "O4"))
        kb = add(p["op_b"], use_shaping=False)
        pairs.append(
            {"name": p["name"], "key_a": ka, "key_b": kb, "disambiguator": p["disambiguator"]}
        )
    pc = d["control"]["power_pair"]
    control = {
        "name": pc["name"],
        "key_a": add(pc["op_a"], False),
        "key_b": add(pc["op_b"], False),
    }
    return units, pairs, control


# --------------------------------------------------------------------------------------
# Collection (real Go2)
# --------------------------------------------------------------------------------------
def _read_obs48(backend) -> np.ndarray:
    o = getattr(backend, "_obs", None)
    if o is None:
        return np.zeros(48, dtype=np.float32)
    return np.asarray(o[0].detach().cpu().numpy(), dtype=np.float32)


def _read_torque(backend) -> np.ndarray:
    try:
        return np.asarray(backend._robot.data.applied_torque[0].detach().cpu().numpy(), np.float32)
    except (AttributeError, IndexError, RuntimeError, TypeError) as e:  # surface, don't mask
        print(f"[WARN] _read_torque failed ({type(e).__name__}: {e}); returning zeros", flush=True)
        return np.zeros(12, dtype=np.float32)


def collect_units(
    backend, units: dict, *, seeds: int, seed_base: int, steps: int, cruise: float, dt: float,
    lane_offset: int = 0, interleave: bool = False,
) -> dict:
    """Drive each unit for ``seeds`` rollouts; return key -> list of per-lane trace dicts
    {obs48 (S,48), feat12 (S,12), tau12 (S,12), manifest (S,), t (S,), pen (S,), fell, ...}.

    ``interleave`` alternates the collection ORDER (seed-outer / unit-inner: O4,O2,…,O4,O2,…)
    instead of unit-outer (all O4 then all O2). Any monotonic reused-app physics drift (#51) then
    affects every operator's lanes equally over collection time, so it cancels in the A/B contrast
    and does NOT become an operator-identity confound — validated by the same-operator negative
    control (a valid C2ST needs same-op AUC ≈ 0.5). REQUIRED for the E1/A1 certification."""
    import kino_vla.monitor.hazard_lab as HL
    from kino_vla.monitor.hazard_lab import monitor_features
    from kino_vla.sim.operators import Tether

    results: dict[str, list] = {k: [] for k in units}
    onset_step = int(round(HL.ONSET_T / dt))
    lane_i = lane_offset
    keys = list(units)
    order = ([(k, s) for s in range(seeds) for k in keys] if interleave
             else [(k, s) for k in keys for s in range(seeds)])
    for key, s in order:
        u = units[key]
        op_id, theta, shaping = u["op_id"], u["theta"], u["shaping"]
        seed = seed_base + s
        lane_i += 1
        y = LANE_SPACING_M * lane_i
        sc = HL.build_scenario(op_id, y, theta)
        if op_id == "O4" and shaping and shaping.get("enabled"):
            sc.operator = Tether(
                sc.rect, theta["k"], theta["c"], 0.0, theta["f_break"],
                force_cap_n=float(shaping["force_cap_n"]),
                force_offset_n=float(shaping["force_offset_n"]),
            )
        backend._start_pos = np.array([0.0, y])
        backend._start_heading = 0.0
        obs = backend.reset(seed)
        backend.clear_payload()
        backend.set_effort_scale(1.0)
        HL.install(sc, backend)

        # NOTE (onset alignment): for onset ops (O5 payload / O10 effort) the manifest label
        # leads the observed effect by 1 step — the fault is injected below AFTER the row is
        # recorded, so it first manifests in the next step's obs. We deliberately do NOT patch
        # hazard_lab.hazard_label (it is the deployed-monitor's labeling source, #48). It is a
        # non-issue for E1: the C2ST is op_a-vs-op_b (a PAIR), so the single stale row appears
        # identically in both operators' segments and cancels in the A/B AUC (set
        # arm_steps_after_entry≥1 to drop it entirely).
        obs48, feat12, tau12, manifest, tt, pen = [], [], [], [], [], []
        payload_added = False
        t_in_region = 0.0
        entry = None
        fell = False
        k = 0
        for k in range(steps):
            in_region = sc.rect is not None and sc.rect.contains(obs.pos)
            if in_region:
                t_in_region += dt
            man = HL.hazard_label(sc, obs.pos, float(obs.t), t_in_region)
            obs48.append(_read_obs48(backend))
            tau12.append(_read_torque(backend))
            feat12.append(monitor_features(obs).astype(np.float32))
            manifest.append(int(man))
            tt.append(float(obs.t))
            if in_region:
                entry = obs.pos.copy() if entry is None else entry
                pen.append(float(np.linalg.norm(obs.pos - entry)))
            else:
                entry = None
                pen.append(0.0)

            if sc.onset_kind == "payload" and k == onset_step and not payload_added:
                backend.add_payload(float(sc.theta["mass"]), np.zeros(2))
                payload_added = True
            if sc.operator is not None and sc.onset_kind in ("effort", "push"):
                sc.operator.on_step(backend, float(obs.t))

            obs = backend.step(np.array([cruise, 0.0, 0.0]))
            if obs.fallen:
                fell = True
                break
        backend.clear_payload()
        backend.set_effort_scale(1.0)
        results[key].append(
            {
                "obs48": np.asarray(obs48, dtype=np.float32),
                "feat12": np.asarray(feat12, dtype=np.float32),
                "tau12": np.asarray(tau12, dtype=np.float32),
                "manifest": np.asarray(manifest, dtype=np.int8),
                "t": np.asarray(tt, dtype=np.float32),
                "pen": np.asarray(pen, dtype=np.float32),
                "fell": bool(fell),
                "key": key,
                "op_id": op_id,
                "seed": seed,
                "lane": lane_i,
            }
        )
        print(
            f"[lane {lane_i}] {key:>28} seed{seed}: steps={k + 1} "
            f"manifest={int(np.sum(manifest))} fell={fell}",
            flush=True,
        )
    return results


# --------------------------------------------------------------------------------------
# Trace -> feature-set lane arrays (consumed by the C2ST in isaac_e1_check.py)
# --------------------------------------------------------------------------------------
FEATURE_SET_DIMS = {"feat12": 12, "obs48": 48, "obs48_tau": 60}


def feature_names(feature_set: str) -> list[str]:
    from kino_vla.monitor.hazard_lab import MON_FEATURE_SCHEMA

    if feature_set == "feat12":
        return list(MON_FEATURE_SCHEMA)
    if feature_set == "obs48":
        return [f"obs{i}" for i in range(48)]
    if feature_set == "obs48_tau":
        return [f"obs{i}" for i in range(48)] + [f"tau{j}" for j in range(12)]
    raise ValueError(f"unknown feature_set {feature_set!r}")


def _longest_manifest_run(manifest: np.ndarray) -> tuple[int, int]:
    """Index range [lo, hi) of the longest contiguous manifest==1 run (empty -> (0, 0))."""
    best_lo = best_hi = 0
    i = 0
    n = len(manifest)
    while i < n:
        if manifest[i]:
            j = i
            while j < n and manifest[j]:
                j += 1
            if j - i > best_hi - best_lo:
                best_lo, best_hi = i, j
            i = j
        else:
            i += 1
    return best_lo, best_hi


def lane_trace(trace: dict, feature_set: str, arm_steps: int) -> np.ndarray | None:
    """The (S, F) per-step array for ``feature_set``, restricted to the longest contiguous
    *manifest* segment (the operator's effect is active) and advanced by ``arm_steps`` (the
    regime-B lever that drops the entry ramp). Returns None if the segment is empty."""
    lo, hi = _longest_manifest_run(trace["manifest"])
    lo = lo + max(0, int(arm_steps))
    if hi - lo <= 0:
        return None
    if feature_set == "feat12":
        seg = trace["feat12"][lo:hi]
    elif feature_set == "obs48":
        seg = trace["obs48"][lo:hi]
    elif feature_set == "obs48_tau":
        seg = np.concatenate([trace["obs48"][lo:hi], trace["tau12"][lo:hi]], axis=1)
    else:
        raise ValueError(f"unknown feature_set {feature_set!r}")
    return np.asarray(seg, dtype=np.float64)


def save_npz(results: dict, out_dir: Path, tag: str, dt: float, meta: dict) -> Path:
    """Flatten all lanes' per-step rows into one inspectable npz (QA 5.2 reproducibility).

    ``fell`` is broadcast per-row (constant within a lane) so the offline A1 re-analysis can
    reproduce the check-script's fallen-lane drop (R8) without re-running Isaac."""
    import json

    rows_o, rows_f, rows_t, key_a, lane_a, man_a, t_a, pen_a, fell_a = (
        [], [], [], [], [], [], [], [], []
    )
    for key, traces in results.items():
        for tr in traces:
            n = tr["obs48"].shape[0]
            rows_o.append(tr["obs48"])
            rows_f.append(tr["feat12"])
            rows_t.append(tr["tau12"])
            key_a += [key] * n
            lane_a.append(np.full(n, tr["lane"], dtype=np.int32))
            man_a.append(tr["manifest"])
            t_a.append(tr["t"])
            pen_a.append(tr["pen"])
            fell_a.append(np.full(n, int(tr["fell"]), dtype=np.int8))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"e1_trace_{tag}.npz"
    np.savez_compressed(
        out,
        obs48=np.concatenate(rows_o) if rows_o else np.zeros((0, 48), np.float32),
        feat12=np.concatenate(rows_f) if rows_f else np.zeros((0, 12), np.float32),
        tau12=np.concatenate(rows_t) if rows_t else np.zeros((0, 12), np.float32),
        key=np.asarray(key_a),
        lane=np.concatenate(lane_a) if lane_a else np.zeros(0, np.int32),
        manifest=np.concatenate(man_a) if man_a else np.zeros(0, np.int8),
        t=np.concatenate(t_a) if t_a else np.zeros(0, np.float32),
        pen=np.concatenate(pen_a) if pen_a else np.zeros(0, np.float32),
        fell=np.concatenate(fell_a) if fell_a else np.zeros(0, np.int8),
        dt=np.float32(dt),
        meta=np.asarray(json.dumps(meta)),
    )
    print(f"[collect] wrote {out}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 real-Go2 obs+history trace collection")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--config", default="eval/e1_c2st.yaml")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="0 = use the config regime.steps")
    ap.add_argument("--tag", default="train")
    ap.add_argument("--out", default="outputs/eval/e1")
    ap.add_argument("--o4-shaping", action="store_true",
                    help="enable the #49 O4 peel-plateau matched preset (A1.1/A1.3 certificate)")
    ap.add_argument("--o4-cap", type=float, default=0.0, help="force_cap_n for --o4-shaping")
    ap.add_argument("--o4-offset", type=float, default=14.0, help="force_offset_n for --o4-shaping")
    ap.add_argument("--interleave", action="store_true",
                    help="alternate collection order (seed-outer) so reused-app drift (#51) "
                         "cancels in the A/B contrast — REQUIRED for the A1 certification")
    args = ap.parse_args()
    app = AppLauncher(args).app

    from kino_vla.utils.config import load_config

    overrides = {}
    if args.o4_shaping:  # make O4's forward grip a constant drag = O2's (matched preset, #49)
        overrides = {"o4_shaping.enabled": True, "o4_shaping.force_cap_n": float(args.o4_cap),
                     "o4_shaping.force_offset_n": float(args.o4_offset)}
    cfg = load_config(args.config, overrides=overrides)
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    units, pairs, control = e1_plan(cfg)
    steps = int(args.steps or cfg.regime.steps)
    cruise = float(cfg.regime.cruise_mps)
    print(
        f"[collect] tag={args.tag} dt={dt:.4f} units={list(units)} "
        f"seeds={args.seeds} (base {args.seed_base}) steps={steps} cruise={cruise}",
        flush=True,
    )
    results = collect_units(
        backend, units, seeds=args.seeds, seed_base=args.seed_base, steps=steps,
        cruise=cruise, dt=dt, interleave=args.interleave,
    )
    meta = {"tag": args.tag, "seeds": args.seeds, "seed_base": args.seed_base, "steps": steps,
            "cruise": cruise, "config": args.config, "commit": git_commit(),
            "o4_shaping": cfg.to_dict()["o4_shaping"], "interleave": bool(args.interleave)}
    save_npz(results, Path(args.out), args.tag, dt, meta)

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
