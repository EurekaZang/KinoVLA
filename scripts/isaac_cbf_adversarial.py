#!/usr/bin/env python
"""M3 CBF adversarial gate on the Isaac Go2 (spec §6.6) — the shield clamps hostile cmds.

Streams hostile Sport-Client velocity commands (max-out dashes, sign-flipping
oscillation, spin-dashes) at the physically-simulated Go2 with the command cap raised so
a hostile command *exceeds* the capture-safe envelope. The CBF-QP shield runs in the live
control loop and adjudicates every command.

HONEST SCOPE (deviations #9/#13): the trained Go2 policy is inherently command-robust —
velocity-command hostility alone does not topple it (it filters commands into stable
gaits), and the CBF's "zero falls" guarantee is a property of the reduced LIP model, not
the full contact dynamics (that fall-count contrast lives in the surrogate adversarial
gate, where the fall model is falsifiable). What IS demonstrable on the real Go2 is that
the shield is *actively intervening*: it clamps each hostile command onto the capture-safe
set, so the commanded velocity it actually issues stays bounded well below the raw hostile
command. This gate asserts: the shield intervenes on most steps AND its issued command
speed is bounded below the hostile input. It also reports the (push-perturbed) fall counts
as a diagnostic.

Run:  python scripts/isaac_cbf_adversarial.py --headless   (GPU machine)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 Isaac CBF adversarial gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.shield.adversarial import HOSTILE_PROFILES
    from kino_vla.shield.cbf_shield import CbfShield
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    # Raise the command cap so hostile commands exceed the policy's trained envelope.
    cfg = load_config("sim/go2_skeleton.yaml", {"max_speed_mps": 2.5, "max_yaw_rate_radps": 3.0})
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    profiles = ["max_forward", "oscillate", "spin_dash"]
    n_steps = 150
    hostile_v, hostile_w = 2.5, 3.0

    def run_episode(profile: str, shielded: bool, seed: int) -> dict:
        backend._start_pos = np.array([0.0, 0.0])
        obs = backend.reset(seed)
        shield = CbfShield(load_config("shield/cbf_v0.yaml")) if shielded else None
        rng = np.random.default_rng(seed)
        cmd_fn = HOSTILE_PROFILES[profile]
        intervened, hostile_speed, issued_speed, fell = 0, [], [], False
        for k in range(n_steps):
            cmd = cmd_fn(k, rng, hostile_v, hostile_w)
            if shield is not None:
                dec = shield.filter(cmd, obs)
                out = dec.cmd
                intervened += int(dec.intervened)
            else:
                out = cmd
            hostile_speed.append(float(np.linalg.norm(cmd[:2])))
            issued_speed.append(float(np.linalg.norm(out[:2])))
            obs = backend.step(out)
            if obs.fallen:
                fell = True
                break
        return {
            "intervened": intervened,
            "steps": k + 1,
            "max_hostile": max(hostile_speed),
            "max_issued": max(issued_speed),
            "fell": fell,
        }

    rows, total_intervened, total_steps, max_issued_shielded = [], 0, 0, 0.0
    falls = {"shielded": 0, "bypassed": 0}
    for profile in profiles:
        for shielded in (True, False):
            r = run_episode(profile, shielded, seed=0)
            tag = "shielded" if shielded else "bypassed"
            falls[tag] += int(r["fell"])
            if shielded:
                total_intervened += r["intervened"]
                total_steps += r["steps"]
                max_issued_shielded = max(max_issued_shielded, r["max_issued"])
            rows.append(
                f"{profile:12s} {tag:9s} intervened={r['intervened']:3d}/{r['steps']:3d} "
                f"max_issued={r['max_issued']:.2f} (hostile {r['max_hostile']:.2f}) "
                f"fell={r['fell']}"
            )

    print("[isaac_cbf_adversarial] per-episode:")
    for row in rows:
        print(f"  {row}")
    intervene_frac = total_intervened / max(1, total_steps)
    print(
        f"[isaac_cbf_adversarial] shield intervened on {intervene_frac:.0%} of steps; "
        f"max issued speed {max_issued_shielded:.2f} m/s vs hostile {hostile_v:.2f} m/s; "
        f"falls shielded={falls['shielded']} bypassed={falls['bypassed']}"
    )
    # The shield is demonstrably adjudicating the real Go2's command stream: it intervenes
    # on most steps and bounds the issued command well below the hostile input.
    passed = intervene_frac > 0.5 and max_issued_shielded < hostile_v - 0.5
    print(
        "PASS: shield actively clamps hostile commands on the real Go2"
        if passed
        else "FAIL: shield did not bound the hostile commands"
    )

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
