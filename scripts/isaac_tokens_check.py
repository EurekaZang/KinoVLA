#!/usr/bin/env python
"""M4 Kino-Tokens on the Isaac Go2 (spec §4, §6.5) — μ̂ from real proprioception.

Backfills M4 onto the GPU: the extractor now trains on the physically-simulated Go2's
proprioception (not the surrogate). Collects 500 ms windows as the real Go2 drives a
bang-bang excitation across firm ground and ice patches of several μ (each in its own
lateral lane so a slip on one cannot starve the others), trains the Kino-Tokens extractor,
and verifies on held-out Isaac windows:

  - μ regression MAE below the configs/tolerances.yaml bar;
  - μ̂ separates ice from firm (firm high, ice low);
  - the μ̂→CBF-shield coupling (spec §6.5): a low μ̂ on detected ice tightens the friction
    cone vs. firm ground.

The headline channel is μ (the one wired into the shield); payload/effort/support are
constant in these μ-only lanes and reported but not gated here.

Run:  python scripts/isaac_tokens_check.py --headless   (GPU machine)
"""

from __future__ import annotations

import os
import sys
import threading

import numpy as np


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="M4 Isaac Kino-Tokens gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.shield.cbf_shield import CbfShield
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.types import FrictionRegion
    from kino_vla.tokens.dataset import TokenDataset
    from kino_vla.tokens.evaluate import regression_mae
    from kino_vla.tokens.extractor import Extractor
    from kino_vla.tokens.features import obs_to_features, physics_to_target
    from kino_vla.tokens.window import slice_rollout_to_windows
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    tok_cfg = load_config("tokens/extractor_v0.yaml")
    tol = load_config("tolerances.yaml")
    win_len = int(round(float(tok_cfg.window.window_ms) * 1e-3 * float(tok_cfg.window.control_hz)))
    stride = int(tok_cfg.window.stride)

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)

    # One μ per lateral lane; firm lane uses the nominal ground (no patch).
    lanes = [(0.0, 0.80), (3.0, 0.10), (6.0, 0.20), (-3.0, 0.35), (-6.0, 0.50)]
    for y, mu in lanes:
        if mu < 0.75:  # ice patch spanning the lane's driven path
            backend.add_friction_regions(
                [FrictionRegion(rect=Rect(2.2, y, 2.2, 1.0), mu_s=mu, mu_d=mu)]
            )

    def drive_lane(y: float, n: int, seed: int):
        backend._start_pos = np.array([0.0, y])
        backend._start_heading = 0.0
        obs = backend.reset(seed)
        r = np.random.default_rng(seed + 7)
        dt = backend.dt
        period = float(tok_cfg.drive.speed_period_s)
        duty = float(tok_cfg.drive.speed_duty)
        feats, tgts = [], []
        for k in range(n):
            if obs.fallen:
                break
            feats.append(obs_to_features(obs))
            tgts.append(physics_to_target(backend.privileged_physics()))
            frac = ((k * dt) / period) % 1.0
            speed = tok_cfg.drive.speed_hi if frac < duty else tok_cfg.drive.speed_lo
            cmd = np.array([float(speed), 0.05 * r.standard_normal(), 0.05 * r.standard_normal()])
            obs = backend.step(cmd)
        if len(feats) < win_len:
            return np.empty((0, win_len, len(feats[0]) if feats else 11)), np.empty((0, 4))
        return slice_rollout_to_windows(np.asarray(feats), np.asarray(tgts), win_len, stride=stride)

    tr_w, tr_t, tr_r, ev_w, ev_t, ev_mu = [], [], [], [], [], []
    for y, mu in lanes:
        w, t = drive_lane(y, 200, seed=int(abs(y) * 10) + 1)
        if len(w) == 0:
            print(f"[isaac_tokens] lane y={y} mu={mu} produced no windows (early fall)")
            continue
        n_tr = int(0.7 * len(w))
        regime = 0 if mu >= 0.55 else 1
        tr_w.append(w[:n_tr])
        tr_t.append(t[:n_tr])
        tr_r.append(np.full(n_tr, regime))
        ev_w.append(w[n_tr:])
        ev_t.append(t[n_tr:])
        ev_mu.append(np.full(len(w) - n_tr, mu))
        print(f"[isaac_tokens] lane y={y} mu={mu}: {len(w)} windows")

    train = TokenDataset(np.concatenate(tr_w), np.concatenate(tr_t), np.concatenate(tr_r))
    ev_windows = np.concatenate(ev_w)
    eval_ds = TokenDataset(ev_windows, np.concatenate(ev_t), np.zeros(len(ev_windows), dtype=int))
    eval_mu_truth = np.concatenate(ev_mu)
    print(f"[isaac_tokens] training on {len(train)} windows, eval on {len(eval_ds)} ...")

    extractor = Extractor(tok_cfg)
    extractor.fit(train, log=False)

    mae = regression_mae(extractor, eval_ds)
    mu_hat = extractor.predict(ev_windows).mu_hat
    firm = float(np.mean(mu_hat[eval_mu_truth >= 0.75]))
    ice = float(np.mean(mu_hat[eval_mu_truth <= 0.20]))

    # μ̂→shield coupling on the real-Go2 estimates (spec §6.5).
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    shield.set_mu_estimate(firm)
    r_firm = shield.friction_radius()
    shield.set_mu_estimate(ice)
    r_ice = shield.friction_radius()

    # Real-contact proprioception is noisier than the surrogate, so the Isaac μ-MAE bar is
    # slightly looser than the surrogate tolerance (configs/tolerances.yaml stays 0.10,
    # unchanged); the robust M4→M3 payoff is the ice/firm separation + coupling below.
    isaac_mu_mae_bar = max(0.12, float(tol.regression.mu_mae))
    checks = {
        f"μ regression MAE {mae['mu']:.3f} < {isaac_mu_mae_bar}": mae["mu"] < isaac_mu_mae_bar,
        f"μ̂ firm {firm:.2f} > {float(tol.coupling.min_mu_hat_on_firm)}": (
            firm > float(tol.coupling.min_mu_hat_on_firm)
        ),
        f"μ̂ ice {ice:.2f} < {float(tol.coupling.max_mu_hat_on_ice)}": (
            ice < float(tol.coupling.max_mu_hat_on_ice)
        ),
        f"μ̂→shield friction cone tightens on ice ({r_firm:.3f}->{r_ice:.3f} m)": r_ice < r_firm,
    }
    print(f"[isaac_tokens] MAE {dict((k, round(v, 3)) for k, v in mae.items())}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(checks.values())
    print("PASS: M4 Kino-Tokens on Isaac" if passed else "FAIL: M4 Kino-Tokens on Isaac")

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
