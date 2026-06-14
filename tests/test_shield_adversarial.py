"""Adversarial-command gate (spec §6.6, M3 exit criterion).

The headline shield claim: **zero falls with the shield active vs. non-zero falls
with it bypassed**, under hostile velocity-command streams. Scope (honest boundary):
the guarantee is over the *capture point / topple* failure mode on dry ground — the
formal §6.6 invariant-set claim. Falls from sustained *slip* on extreme low-μ ground
are a friction phenomenon governed by the planner's speed choice (Set_Constraint,
M7) and the μ̂ head (M4), not by the capture-point CBF; those are exercised
informationally by ``kino_vla.shield.adversarial`` but not gated here.

Per QA 5.2, any edit under ``kino_vla/shield/`` must re-run this suite before merge.
"""

from __future__ import annotations

from kino_vla.shield.adversarial import (
    HOSTILE_PROFILES,
    run_adversarial_episode,
    run_adversarial_suite,
)


def test_dry_ground_zero_falls_shielded_nonzero_bypassed():
    summary = run_adversarial_suite(seeds=[0, 1, 2, 3], include_ice=False, n_steps=600)
    assert summary.count(True) == summary.count(False) == len(HOSTILE_PROFILES) * 4
    assert summary.falls(True) == 0, "the shield must prevent every fall on dry ground"
    assert summary.falls(False) > 0, "bypassed, hostile commands must topple the robot"


def test_every_profile_topples_when_bypassed():
    # Non-vacuity: each hostile profile genuinely topples the unshielded robot, so
    # the shielded zero-falls result is earned per-profile, not by an easy mix.
    for profile in HOSTILE_PROFILES:
        bypass = run_adversarial_episode(profile, seed=0, shielded=False, n_steps=600)
        assert bypass.fell, f"profile {profile!r} should topple the bypassed robot"
        shielded = run_adversarial_episode(profile, seed=0, shielded=True, n_steps=600)
        assert not shielded.fell, f"profile {profile!r} toppled despite the shield"
        assert shielded.max_speed_mps < 1.2  # held below the nominal capture speed


def test_ice_is_outside_the_capture_point_guarantee():
    # Honest boundary (NOT a shield win): on extreme low-μ ice the failure is sustained
    # *slip*, not a capture-point topple, so the §6.6 CBF does not prevent it — both the
    # shielded and bypassed robot fall (the shield can even be anti-protective here via
    # its halt-then-creep cycle). Full ice safety is the planner's job: Set_Constraint
    # (low speed) / Hold_and_Request (M7) fed by the μ̂ head (M4). This test pins that
    # honest scope so the ice result is never mis-sold as a shield guarantee.
    shielded = run_adversarial_episode(
        "max_forward", seed=0, shielded=True, n_steps=600, ice_mu=0.10
    )
    bypassed = run_adversarial_episode(
        "max_forward", seed=0, shielded=False, n_steps=600, ice_mu=0.10
    )
    assert shielded.fell and bypassed.fell  # capture-point CBF does not cover slip-falls
    assert shielded.n_halt > 0  # but the §6.5 friction coupling DID engage the fallback
