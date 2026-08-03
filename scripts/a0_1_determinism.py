#!/usr/bin/env python
"""A0.1 determinism gate — cross-ordering byte-identity + same-op validity gate (real Go2).

The #51/#52 residue is an operator-ORDER effect: on a REUSED Isaac app, lane N's PhysX state
depends on which operators ran in lanes 0..N-1 (accumulating collider prims reorder broadphase/
warm-start; payload/effort/wrench/collapse-material survive ``reset``). A high-capacity temporal
classifier then "separates" byte-identical operators (A1 #52); closed-loop reach becomes order-
sensitive (E4 reconcile); the θ* curve gets spurious dips/flips.

This script measures the residue directly and tests the fix. It runs a FIXED set of ``(op, seed)``
lanes in TWO different orderings inside ONE reused app, under a chosen reset mechanism, and asks:
is each lane's proprio trajectory (obs48 ⊕ τ12 — the binding representation) IDENTICAL regardless
of what ran before it?

  --mechanism naive : reset(seed) + clear_payload + set_effort_scale(1) + install   (the confound)
  --mechanism deep  : deep_reset(seed) + install                                     (the A0.1 fix)

Acceptance (design_doc §2 A0.1): under ``deep`` every same-(op,seed) lane is cross-ordering
identical (max|Δ| = 0), so the operator-order residue is gone; a structured same-operator C2ST
(odd/even lane split) then returns AUC ≈ 0.5 (the validity gate). The fresh-app reference
(scripts/isaac_e1_singlelane.py, one app per lane) is byte-identical by construction (A1).

Run:  python scripts/a0_1_determinism.py --headless --mechanism deep  --seeds 10
      python scripts/a0_1_determinism.py --headless --mechanism naive --seeds 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

# The matched pair + power control, θ mirroring configs/eval/e1_c2st.yaml / A1 (O4↔O2 identical).
THETA = {
    "O4": {"k": 14.0, "c": 6.0, "f_break": 1.0e9},
    "O2": {"k": 14.0, "c": 6.0, "d_sink": 0.08},
    "O1": {"mu": 0.09},
}
# Two orderings over the SAME 7 (op, seed) lanes; B permutes so each lane has other predecessors.
ORDER_A = [("O4", 0), ("O4", 1), ("O4", 2), ("O2", 0), ("O2", 1), ("O2", 2), ("O1", 0)]
ORDER_B = [("O1", 0), ("O2", 2), ("O4", 0), ("O2", 0), ("O4", 2), ("O2", 1), ("O4", 1)]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.1 determinism gate (cross-ordering identity)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--mechanism", choices=["naive", "deep"], required=True)
    ap.add_argument("--seeds", type=int, default=10, help="same-op validity-gate seeds per op")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lane-y", type=float, default=4.0,
                    help="FIXED lateral lane for every lane, so predecessor-history is the ONLY "
                         "variable across orderings (deep_reset despawns, so same-y is safe)")
    ap.add_argument("--cruise", type=float, default=0.6)
    ap.add_argument("--o4-cap", type=float, default=0.0)
    ap.add_argument("--o4-offset", type=float, default=14.0)
    ap.add_argument("--out", default="outputs/eval/a0/determinism")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    import kino_vla.monitor.hazard_lab as HL
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import Tether
    from kino_vla.utils.config import REPO_ROOT, load_config

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt

    def _tau() -> np.ndarray:
        try:
            return np.asarray(backend._robot.data.applied_torque[0].detach().cpu().numpy(),
                              np.float32)
        except Exception:  # noqa: BLE001
            return np.zeros(12, np.float32)

    def drive(op: str, seed: int) -> np.ndarray:
        """Run one lane under the chosen reset mechanism; return the (S, 60) obs48⊕τ trajectory.

        Every lane uses a FIXED y, so predecessor-history is the ONLY variable across orderings —
        the accumulated PhysX residue is exactly what the mechanism must scrub."""
        y = float(args.lane_y)
        theta = THETA[op]
        sc = HL.build_scenario(op, y, theta)
        if op == "O4":
            sc.operator = Tether(sc.rect, theta["k"], theta["c"], 0.0, theta["f_break"],
                                 force_cap_n=float(args.o4_cap),
                                 force_offset_n=float(args.o4_offset))
        backend._start_pos = np.array([0.0, y])
        backend._start_heading = 0.0
        if args.mechanism == "deep":
            obs = backend.deep_reset(seed)
            print(f"[a0.1/deep] {op}_{seed} despawned={backend._last_deep_reset_removed} prims",
                  flush=True)
        else:  # naive reused-app path (mirrors isaac_e1_collect: reset + manual clears + install)
            obs = backend.reset(seed)
            backend.clear_payload()
            backend.set_effort_scale(1.0)
        HL.install(sc, backend)
        rows = []
        for _k in range(args.steps):
            o = getattr(backend, "_obs", None)
            obs48 = (np.zeros(48, np.float32) if o is None
                     else np.asarray(o[0].detach().cpu().numpy(), np.float32))
            rows.append(np.concatenate([obs48, _tau()]))
            if sc.operator is not None and sc.onset_kind in ("effort", "push"):
                sc.operator.on_step(backend, float(obs.t))
            obs = backend.step(np.array([args.cruise, 0.0, 0.0]))
            if obs.fallen:
                break
        return np.asarray(rows, np.float32)

    # ---- Part 1: cross-ordering identity over the fixed lane set --------------------------------
    traj_a = {f"{op}_{s}": drive(op, s) for op, s in ORDER_A}
    traj_b = {f"{op}_{s}": drive(op, s) for op, s in ORDER_B}
    cross = {}
    for key in traj_a:
        a, b = traj_a[key], traj_b[key]
        n = min(len(a), len(b))
        d = float(np.abs(a[:n] - b[:n]).max()) if n else float("nan")
        cross[key] = {"max_abs_delta": d, "len_a": int(len(a)), "len_b": int(len(b)),
                      "identical": bool(d == 0.0)}
        print(f"[a0.1/{args.mechanism}] cross-order {key:6}: max|Δ|={d:.6g} "
              f"(len {len(a)}/{len(b)})", flush=True)

    # ---- Part 2: same-op validity-gate lanes (O4 & O2 across seeds, interleaved) ----------------
    # Interleaving makes each op's lanes fall at odd/even collection positions, so the structured
    # odd/even C2ST split (offline) probes exactly the operator-ORDER residue A1 §0 caught.
    same_op = {"O4": [], "O2": []}
    for s in range(args.seeds):
        for op in ("O4", "O2"):
            same_op[op].append(drive(op, s))

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"lanes_{args.mechanism}.npz"
    payload: dict = {}
    for tag, d in (("A", traj_a), ("B", traj_b)):
        for k, v in d.items():
            payload[f"cross_{tag}_{k}"] = v
    for op, lst in same_op.items():
        for i, v in enumerate(lst):
            payload[f"sameop_{op}_{i}"] = v
    np.savez_compressed(npz, **payload)

    summary = {
        "mechanism": args.mechanism, "commit": git_commit(), "dt": float(dt),
        "steps": args.steps, "cruise": args.cruise, "seeds": args.seeds,
        "o4_preset": {"cap": args.o4_cap, "offset": args.o4_offset},
        "cross_ordering": cross,
        "all_identical": bool(all(v["identical"] for v in cross.values())),
        "max_cross_delta": float(max(v["max_abs_delta"] for v in cross.values())),
        "npz": str(npz.relative_to(REPO_ROOT)),
    }
    (out_dir / f"summary_{args.mechanism}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[a0.1/{args.mechanism}] ALL_IDENTICAL={summary['all_identical']} "
          f"max_cross_delta={summary['max_cross_delta']:.6g}", flush=True)
    print(f"[a0.1/{args.mechanism}] wrote {npz} + summary_{args.mechanism}.json", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
