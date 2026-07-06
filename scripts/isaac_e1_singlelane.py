#!/usr/bin/env python
"""A1 determinism null-test — collect ONE lane in a FRESH Isaac app (no cross-lane residual).

The E1/A1 C2ST is run on a REUSED Isaac app across lanes; #51 physics non-determinism leaves an
operator-ORDER residual (each lane's PhysX state depends on the previous lane's operator), which a
high-capacity temporal classifier exploits to "separate" operators whose applied trunk force is
algebraically identical. This script collects exactly ONE (operator, seed) lane per process — so
every lane is a fresh-app first-lane with no predecessor — to produce a confound-free collection.
Run it once per (op, seed) via a shell loop; each writes lane_<op>_<seed>.npz. The gold-standard
A0.1 control: if fresh-app O4↔O2 C2ST is at chance while the reused-app one is 1.0, the reused-app
separation is the confound (and E1's matched-pair claim holds under determinism).

Run:  python scripts/isaac_e1_singlelane.py --headless --op O4 --seed 0 --o4-shaping
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="collect one E1 lane in a fresh Isaac app")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--op", required=True, help="O4 | O2 | O1 (the matched-pair members + control)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=460)
    ap.add_argument("--cruise", type=float, default=0.6)
    ap.add_argument("--o4-shaping", action="store_true", help="#49 matched preset for O4")
    ap.add_argument("--o4-cap", type=float, default=0.0)
    ap.add_argument("--o4-offset", type=float, default=14.0)
    ap.add_argument("--out", default="outputs/eval/a1/traces_fresh")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    import kino_vla.monitor.hazard_lab as HL
    from kino_vla.monitor.hazard_lab import monitor_features
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import Tether
    from kino_vla.utils.config import REPO_ROOT, load_config

    # matched θ (mirror configs/eval/e1_c2st.yaml pairs) + the A1.2/A1.4 operators.
    # O7 = looks-safe/is-slippery (appearance solid_ground, μ=0.09); O8 = invisible collider;
    # clean = the benign proprio look-alike (nominal ground, no operator).
    theta = {"O4": {"k": 14.0, "c": 6.0, "f_break": 1.0e9},
             "O2": {"k": 14.0, "c": 6.0, "d_sink": 0.08},
             "O1": {"mu": 0.09},
             "O3": {"mu_collapsed": 0.09, "dwell": 0.45},
             "O7": {"mu": 0.09},
             "O8": {},
             "clean": {"cruise": 0.6}}[args.op]

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    y = 4.0  # fixed lane y (single lane per app ⇒ no y-band confound either)
    sc = HL.build_scenario(args.op, y, theta)
    if args.op == "O4" and args.o4_shaping:
        sc.operator = Tether(sc.rect, theta["k"], theta["c"], 0.0, theta["f_break"],
                             force_cap_n=float(args.o4_cap), force_offset_n=float(args.o4_offset))
    backend._start_pos = np.array([0.0, y])
    backend._start_heading = 0.0
    obs = backend.reset(args.seed)
    backend.clear_payload()
    backend.set_effort_scale(1.0)
    HL.install(sc, backend)

    def _obs48():
        o = getattr(backend, "_obs", None)
        return (np.zeros(48, np.float32) if o is None
                else np.asarray(o[0].detach().cpu().numpy(), np.float32))

    def _tau():
        try:
            return np.asarray(backend._robot.data.applied_torque[0].detach().cpu().numpy(),
                              np.float32)
        except Exception:  # noqa: BLE001
            return np.zeros(12, np.float32)

    obs48, feat12, tau12, manifest, tt, pen, posx = [], [], [], [], [], [], []
    t_in_region = 0.0
    entry = None
    fell = False
    for _k in range(args.steps):
        in_region = sc.rect is not None and sc.rect.contains(obs.pos)
        if in_region:
            t_in_region += dt
        man = HL.hazard_label(sc, obs.pos, float(obs.t), t_in_region)
        obs48.append(_obs48())
        tau12.append(_tau())
        feat12.append(monitor_features(obs).astype(np.float32))
        manifest.append(int(man))
        tt.append(float(obs.t))
        posx.append(float(obs.pos[0]))  # x-position for position-based windowing (A1.4 clean lanes)
        if in_region:
            entry = obs.pos.copy() if entry is None else entry
            pen.append(float(np.linalg.norm(obs.pos - entry)))
        else:
            entry = None
            pen.append(0.0)
        if sc.operator is not None and sc.onset_kind in ("effort", "push"):
            sc.operator.on_step(backend, float(obs.t))
        obs = backend.step(np.array([args.cruise, 0.0, 0.0]))
        if obs.fallen:
            fell = True
            break

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"lane_{args.op}_{args.seed}.npz"
    np.savez_compressed(
        out, obs48=np.asarray(obs48, np.float32), feat12=np.asarray(feat12, np.float32),
        tau12=np.asarray(tau12, np.float32), manifest=np.asarray(manifest, np.int8),
        t=np.asarray(tt, np.float32), pen=np.asarray(pen, np.float32),
        posx=np.asarray(posx, np.float32),
        fell=np.int8(fell), op=args.op, seed=np.int32(args.seed),
        meta=np.asarray(json.dumps({"op": args.op, "seed": args.seed, "fresh_app": True})),
    )
    print(f"[singlelane] wrote {out} steps={len(obs48)} manifest={int(np.sum(manifest))} "
          f"fell={fell}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
