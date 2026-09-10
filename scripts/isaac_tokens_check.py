#!/usr/bin/env python
"""M4 Kino-Tokens on the Isaac Go2 (spec §4, §6.5) — STRICT four-channel θ gate.

The extractor must regress *all four*
privileged channels below their ``configs/tolerances.yaml`` bars on the physically-
simulated Go2 (μ 0.10, payload 1.5, effort 0.10, support 0.12), not μ alone at a relaxed
bar. Each channel is excited in its own single-operator lane/phase on the real Go2:

  - μ        — friction-patch lanes of several μ (ice…firm); μ is observable while slipping.
  - support  — graded-height ridges (O9 high-centering); target is the trailing-window
               average support fraction (gait-phase noise smoothed; spec §6 / #22).
  - effort   — set_effort_scale levels, driven bang-bang fast so intermediate cuts bind.
  - payload  — ascending add_payload (irreversible in-process, so collected last).

The per-step rollouts are saved to ``outputs/tokens/isaac_rollouts.npz`` and the extractor
is trained + gated on CPU (kino_vla/tokens/isaac_gate.py). Collection and gating are
decoupled (``--collect-only`` / scripts/isaac_tokens_gate.py): the GPU pass runs once,
the gate iterates offline — the machine has frozen on heavy interactive runs, so training
never shares the GPU-holding loop with an unbounded thread pool.

Run:  python scripts/isaac_tokens_check.py --headless                 (collect + gate)
      python scripts/isaac_tokens_check.py --headless --collect-only  (save npz only)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 Isaac Kino-Tokens strict gate")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="drive the Go2 and save the rollouts npz, then exit (no CPU training)",
    )
    parser.add_argument("--npz", default="outputs/tokens/isaac_rollouts.npz")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.types import FrictionRegion, SupportLossRegion
    from kino_vla.tokens.features import obs_to_features, physics_to_target
    from kino_vla.tokens.isaac_gate import (
        Rollout,
        print_report,
        save_rollouts,
        train_and_gate,
    )
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.geometry import Rect

    tok_cfg = load_config("tokens/extractor_v0.yaml")
    tol = load_config("tolerances.yaml")
    cc = tok_cfg.isaac_collect
    dt = None  # set after backend init

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    period = float(cc.speed_period_s)
    duty = float(cc.speed_duty)

    def bangbang(hi: float, lo: float):
        def f(k: int) -> float:
            return hi if (((k * dt) / period) % 1.0) < duty else lo

        return f

    def drive(y: float, speed_fn, regime: int, channel: str, level: float, seed: int) -> Rollout:
        """Teleport to lane (0, y), settle to cruise, then record n_steps of (feats, tgts)."""
        backend._start_pos = np.array([0.0, y])
        backend._start_heading = 0.0
        obs = backend.reset(seed)
        rng = np.random.default_rng(seed + 7)
        for _ in range(int(cc.settle_steps)):
            obs = backend.step(np.array([speed_fn(0), 0.0, 0.0]))
        feats, tgts = [], []
        for k in range(int(cc.n_steps)):
            if obs.fallen:
                break
            feats.append(obs_to_features(obs))
            tgts.append(physics_to_target(backend.privileged_physics()))
            cmd = np.array(
                [speed_fn(k), 0.03 * rng.standard_normal(), 0.03 * rng.standard_normal()]
            )
            obs = backend.step(cmd)
        print(
            f"[isaac_tokens] {channel:8s} y={y:+5.1f} level={level:5.2f}: "
            f"{len(feats):3d} steps (fell={obs.fallen})"
        )
        return Rollout(
            channel=channel,
            level=float(level),
            regime=int(regime),
            feats=np.asarray(feats, dtype=np.float64),
            tgts=np.asarray(tgts, dtype=np.float64),
        )

    rollouts: list[Rollout] = []

    # --- μ phase (payload=0, effort=1.0): friction lanes ice…firm -----------------------
    mu_fn = bangbang(float(cc.mu_speed_hi), float(cc.mu_speed_lo))
    for y, mu in [(float(a), float(b)) for a, b in cc.mu_lanes]:
        if mu < 0.75:  # ice patch straddling the driven path; firm lane drives bare ground
            backend.add_friction_regions(
                [
                    FrictionRegion(
                        rect=Rect(float(cc.patch_cx), y, float(cc.patch_hx), float(cc.patch_hy)),
                        mu_s=mu,
                        mu_d=mu,
                    )
                ]
            )
        regime = 1 if mu < 0.55 else 0  # low_friction vs normal
        rollouts.append(drive(y, mu_fn, regime, "mu", mu, seed=int(abs(y) * 10) + 1))

    # --- support phase (payload=0, effort=1.0): graded-height ridges (O9) ----------------
    for y, h in [(float(a), float(b)) for a, b in cc.support_lanes]:
        if h > 0.0:
            backend.add_support_loss_regions(
                [
                    SupportLossRegion(
                        rect=Rect(float(cc.patch_cx), y, float(cc.ridge_hx), float(cc.ridge_hy)),
                        residual_support=0.3,
                    )
                ],
                height_m=h,
            )
        regime = 4 if h > 0.0 else 0  # high_centered vs normal
        rollouts.append(
            drive(
                y, lambda k: float(cc.support_speed), regime, "support", h, seed=int(abs(y)) + 200
            )
        )

    # --- effort phase (payload=0): set_effort_scale levels, driven bang-bang fast --------
    eff_fn = bangbang(float(cc.effort_speed_hi), float(cc.effort_speed_lo))
    for scale in [float(s) for s in cc.effort_levels]:
        backend.set_effort_scale(1.0)  # settle at nominal, then cut so the decay hits mid-stride
        backend._start_pos = np.array([0.0, float(cc.effort_lane_y)])
        backend._start_heading = 0.0
        obs = backend.reset(int(scale * 100) + 300)
        for _ in range(int(cc.settle_steps)):
            obs = backend.step(np.array([eff_fn(0), 0.0, 0.0]))
        backend.set_effort_scale(scale)
        rng = np.random.default_rng(int(scale * 100) + 307)
        feats, tgts = [], []
        for k in range(int(cc.n_steps)):
            if obs.fallen:
                break
            feats.append(obs_to_features(obs))
            tgts.append(physics_to_target(backend.privileged_physics()))
            cmd = np.array([eff_fn(k), 0.03 * rng.standard_normal(), 0.03 * rng.standard_normal()])
            obs = backend.step(cmd)
        regime = 3 if scale < 1.0 else 0  # actuator_decay vs normal
        print(
            f"[isaac_tokens] effort   y={float(cc.effort_lane_y):+5.1f} level={scale:5.2f}: "
            f"{len(feats):3d} steps (fell={obs.fallen})"
        )
        rollouts.append(Rollout("effort", scale, regime, np.asarray(feats), np.asarray(tgts)))
    backend.set_effort_scale(1.0)  # restore before the payload phase

    # --- payload phase (effort=1.0): ascending add_payload (irreversible) ----------------
    pay_fn = bangbang(float(cc.payload_speed_hi), float(cc.payload_speed_lo))
    prev = 0.0
    for level in [float(p) for p in cc.payload_levels]:
        add = level - prev
        if add > 0.0:
            backend.add_payload(add, np.array([0.0, 0.0]))
            prev = level
        regime = 2 if level > 0.0 else 0  # overload vs normal
        rollouts.append(
            drive(float(cc.payload_lane_y), pay_fn, regime, "payload", level, seed=int(level) + 400)
        )

    npz_path = REPO_ROOT / str(args.npz)
    save_rollouts(npz_path, rollouts)
    print(f"[isaac_tokens] saved {len(rollouts)} rollouts → {npz_path}")

    passed = True
    if not args.collect_only:
        report = train_and_gate(rollouts, tok_cfg, tol)
        print_report(report)
        passed = report.passed
    else:
        print("PASS: M4 rollouts collected (--collect-only)")

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
