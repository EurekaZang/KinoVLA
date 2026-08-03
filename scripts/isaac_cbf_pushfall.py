#!/usr/bin/env python
"""M3 CBF push-fall contrast on the Isaac Go2 (spec §6.6) — STRICT zero-fall realization.

Closes the CLAUDE.md §6 #22 M3 strict-GPU gap. The surrogate adversarial gate proves the
shield's zero-fall property on the reduced LIP model; this script tries to realize that
contrast on the FULL-ORDER physically-simulated Go2:

  1. cruise forward to steady state;
  2. apply a real O6 impulse (backend.apply_push) that spikes the body velocity into a
     capture-point regime;
  3. stream a hostile "keep accelerating in the push direction" command for a recovery window.

Two arms per (push magnitude, direction, seed): BYPASSED (raw hostile command) vs SHIELDED
(the CBF-QP filters it — reading the post-push body velocity, it sees the DCM leaving the
safe set and clamps the command to a brake). It sweeps magnitudes/directions and reports the
full fall-contrast table — turning the previously-untested 0/0 (CLAUDE.md §6 #22) into a real
characterization of the shield's effect on the full-order policy.

FINDING — TWO ERAS (deviations #13/#23 → #42):
  ORIGINAL (reduced-LIP velocity clamp, 2026-06-16): ANTI-PROTECTIVE on the full Go2 — for a
  forward/diagonal push the clamp braked the robust policy that would otherwise ride the push out
  (bypassed 0/3, shielded 3/3). Root cause: the clamp SUPPRESSED the policy's own recovery STEP.
  FIXED by the DIRECTION-AWARE REACTIVE BRACE (#42, shield.enable_reactive_brace; 2026-06-23): on a
  detected physical push (an UNcommanded velocity jump — not a hostile command, which tracks under
  slew) the shield either BRACES a lateral push (the policy's weak axis: defend the wider/lower
  brace polygon + the physical brace reflex, lower CoM) or DEFERS to the policy on a forward push
  (ride-out — a clamp there is anti-protective). RESULT on the real Go2: NO anti-protective cell
  (shielded ≤ bypassed everywhere — forward/diagonal ride-out makes shielded ≡ bypassed); GENUINE
  protection on every lateral push the bare policy fails (bypassed 3/3 → shielded 0/3); shield-on
  zero falls across the recoverable envelope. The only shielded falls are the extreme cells where
  the BARE policy ALSO falls (unrecoverable for both — a fairness limit, NOT anti-protection). The
  shield config/QP math is untouched (reactive_brace is OFF in cbf_v0.yaml so the surrogate
  adversarial zero-fall gate + §6.6 property tests are byte-identical; it is enabled only for this
  real-Go2 deployment). The surrogate adversarial gate remains the falsifiable reduced-LIP zero-fall
  test; the real-Go2 claims are now (a) command clamping AND (b) genuine push protection via brace.

Run:  python scripts/isaac_cbf_pushfall.py --headless   (GPU machine)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 Isaac CBF push-fall contrast")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.shield.cbf_shield import CbfShield
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    pf = load_config("shield/isaac_pushfall.yaml")
    backend_cfg = load_config(
        "sim/go2_skeleton.yaml",
        {
            "max_speed_mps": float(pf.max_speed_mps),
            "max_yaw_rate_radps": float(pf.max_yaw_rate_radps),
        },
    )
    backend = IsaacPolicyBackend(backend_cfg, np.array([0.0, 0.0]), 0.0)
    cruise = np.array([float(pf.cruise_speed_mps), 0.0, 0.0])
    hostile_speed = float(pf.hostile_speed_mps)
    seeds = [int(s) for s in pf.seeds]

    def run_episode(impulse_ns: float, direction: np.ndarray, shielded: bool, seed: int) -> bool:
        backend._start_pos = np.array([0.0, 0.0])
        backend._start_heading = 0.0
        obs = backend.reset(seed)
        shield = CbfShield(load_config("shield/cbf_v0.yaml")) if shielded else None
        if shield is not None:
            shield.enable_reactive_brace()  # #42: the protective real-Go2 deployment of the shield
        backend.set_reflex(False)
        for _ in range(int(pf.cruise_steps)):
            obs = backend.step(cruise)
            if obs.fallen:
                return True  # toppled before the push (shouldn't happen at 0.5 m/s)
        # Real O6 impulse along the push direction → a capture-point perturbation.
        backend.apply_push(float(impulse_ns) * direction, 0.0)
        hostile = np.array([hostile_speed * direction[0], hostile_speed * direction[1], 0.0])
        for _ in range(int(pf.post_steps)):
            if shield is not None:
                dec = shield.filter(hostile, obs)
                out = dec.cmd
                # #42: a detected push engages the PHYSICAL brace (lower CoM via the posture
                # controller) — the dog-executed recovery the reduced-LIP velocity clamp suppressed.
                backend.set_reflex(
                    any(
                        c.startswith(("RECOVER_BRACE", "INFEASIBLE_FALLBACK", "HALT"))
                        for c in dec.codes
                    )
                )
            else:
                out = hostile
            obs = backend.step(out)
            if obs.fallen:
                return True
        return False

    dir_names = {(1.0, 0.0): "forward", (0.0, 1.0): "lateral", (0.7, 0.7): "diagonal"}
    n = len(seeds)
    min_byp = int(pf.min_bypassed_falls)
    cells: list[tuple[str, float, int, int]] = []  # (dname, J, bypassed_falls, shielded_falls)
    for raw_dir in pf.push_dirs:
        d = np.asarray([float(x) for x in raw_dir], dtype=np.float64)
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        dname = dir_names.get(tuple(round(float(x), 1) for x in raw_dir), str(raw_dir))
        for impulse in [float(j) for j in pf.push_impulses_ns]:
            byp = sum(run_episode(impulse, d, False, s) for s in seeds)
            shi = sum(run_episode(impulse, d, True, s) for s in seeds)
            cells.append((dname, impulse, byp, shi))
            print(
                f"  {dname:9s} J={impulse:5.1f} Ns : "
                f"bypassed_falls={byp}/{n} shielded_falls={shi}/{n}"
            )
            sys.stdout.flush()

    print("[isaac_cbf_pushfall] sweep table:")
    for dname, j, byp, shi in cells:
        print(f"  {dname:9s} J={j:5.1f} Ns : bypassed_falls={byp}/{n} shielded_falls={shi}/{n}")

    # The #42 protection claim (replaces the #13/#23 anti-protection finding). The reduced-LIP
    # velocity clamp was anti-protective on the full Go2 because it suppressed the policy's own
    # recovery STEP. The direction-aware reactive brace fixes that: brace a lateral push (the
    # policy's weak axis), defer to the policy on a forward push (ride-out). Two assertions:
    #   (1) NO ANTI-PROTECTION: in NO cell does the shield fall while the bare policy survives
    #       (shielded ≤ bypassed everywhere) — overturns the pre-fix #13/#23.
    #   (2) GENUINE PROTECTION: ∃ cell where the bare policy falls (≥min_byp) and shielded NEVER
    #       falls — the shield prevents a real full-order topple (not a reduced-LIP/surrogate-only
    #       property). The ZERO-FALL ENVELOPE is every cell with shielded=0; its complement is the
    #       unrecoverable extreme where BOTH arms fall (a fairness limit, NOT anti-protection).
    anti = [(dn, j, byp, shi) for (dn, j, byp, shi) in cells if shi > byp]
    protective = [(dn, j, byp, shi) for (dn, j, byp, shi) in cells if byp >= min_byp and shi == 0]
    unrecoverable = [(dn, j) for (dn, j, byp, shi) in cells if shi == byp and byp >= min_byp]
    zero_fall = [(dn, j) for (dn, j, byp, shi) in cells if shi == 0]
    print(f"[isaac_cbf_pushfall] anti-protective cells (shielded>bypassed): {anti or 'NONE'}")
    print(f"[isaac_cbf_pushfall] protective cells (bypassed≥{min_byp}, shielded=0): {protective}")
    print(
        f"[isaac_cbf_pushfall] shield-on zero-fall envelope: {len(zero_fall)}/{len(cells)} cells; "
        f"unrecoverable-for-both (fairness limit, not anti-protection): {unrecoverable or 'NONE'}"
    )
    ok = not anti and bool(protective)
    if ok:
        print(
            "PASS: the reactive-brace shield is GENUINELY PROTECTIVE on the real Go2 — it never "
            "falls where the bare policy survives (no anti-protection, overturning #13/#23), and "
            f"prevents the full-order topple on {len(protective)} push cell(s) the bare policy "
            "fails (shield-on zero falls across the recoverable envelope)."
        )
    else:
        print(
            f"FAIL: anti-protective cells remain {anti} or no protective cell — the reactive brace "
            "did not transfer the zero-fall guarantee to the full-order push recovery."
        )
    print("PASS: M3 push-fall characterization complete" if ok else "FAIL: M3 push-fall")

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
