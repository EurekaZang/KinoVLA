"""M2 operator gates per QA 5.2: θ-application, determinism, composability.

O3 Collapse, O5 Payload, O8 Invisible-Collider, O9 High-Centering, O10 Effort-Decay,
on the CPU surrogate backend (the Isaac-side PhysX checks live in
tests/test_sim_operators.py behind @pytest.mark.sim). Each operator gets:
(a) θ-application — set a parameter, measure its effect in sim, assert tolerance;
(b) determinism — same seed ⇒ same trajectory hash;
(c) composability — stacks with another operator without crashing and θ concatenates.
"""

from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.operators import (
    Collapse,
    EffortDecay,
    HighCentering,
    InvisibleCollider,
    MuField,
    OperatorStack,
    Payload,
)
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.utils.config import load_config
from kino_vla.utils.geometry import Rect
from kino_vla.utils.seeding import trajectory_hash

TOL = load_config("operators/apply_tolerances.yaml")
SIM_CFG = load_config("sim/surrogate.yaml")
ICE = Rect(cx=3.0, cy=0.0, hx=1.0, hy=1.0)


def make_backend(start=(0.0, 0.0), heading=0.0):
    return SurrogateBackend(SIM_CFG, np.asarray(start, dtype=np.float64), heading)


def run_stack(stack, seed, n_steps=300, cmd=(0.6, 0.0, 0.0), start=(0.0, 0.0)):
    backend = make_backend(start)
    obs = backend.reset(seed)
    stack.on_reset(backend)
    positions, velocities = [], []
    for _ in range(n_steps):
        stack.on_step(backend, obs.t)
        obs = backend.step(np.asarray(cmd, dtype=np.float64))
        positions.append(obs.pos)
        velocities.append(obs.vel_body)
    return trajectory_hash(np.asarray(positions), np.asarray(velocities)), obs


def drive(backend, op, cmd, n_steps, start_obs=None):
    """Apply one operator's hooks while driving a constant command; return last obs."""
    obs = start_obs
    for _ in range(n_steps):
        t = 0.0 if obs is None else obs.t
        op.on_step(backend, t)
        obs = backend.step(np.asarray(cmd, dtype=np.float64))
    return obs


# ---------------------------------------------------------------- O3 Collapse


def test_o3_theta_applied_friction_collapses_after_dwell():
    region = Rect(cx=1.0, cy=0.0, hx=1.0, hy=1.0)
    op = Collapse(region=region, mu_collapsed=0.08, trigger_dwell_s=0.4, mu_intact=0.8)
    backend = make_backend(start=(1.0, 0.0))
    backend.reset(0)
    op.on_reset(backend)
    tol = float(TOL.o3_collapse.mu_abs_tol)
    # Intact before any dwell.
    assert backend.friction_at(np.array([1.0, 0.0])) == pytest.approx(0.8, abs=tol)
    # Dwell in place (near-zero command keeps the robot inside the region).
    drive(backend, op, cmd=(0.0, 0.0, 0.0), n_steps=int(0.6 / backend.dt))
    assert backend.friction_at(np.array([1.0, 0.0])) == pytest.approx(0.08, abs=tol)


def test_o3_no_collapse_without_dwell():
    region = Rect(cx=5.0, cy=0.0, hx=0.4, hy=0.4)
    op = Collapse(region=region, mu_collapsed=0.05, trigger_dwell_s=2.0, mu_intact=0.8)
    backend = make_backend(start=(0.0, 0.0))
    backend.reset(0)
    op.on_reset(backend)
    # Never enter the far region: it must stay intact.
    drive(backend, op, cmd=(0.0, 0.0, 0.0), n_steps=200)
    assert backend.friction_at(np.array([5.0, 0.0])) == pytest.approx(0.8, abs=0.02)


def test_o3_determinism_gate():
    stack = OperatorStack([Collapse(region=ICE, mu_collapsed=0.08, trigger_dwell_s=0.3)])
    h1, _ = run_stack(stack, seed=42)
    h2, _ = run_stack(stack, seed=42)
    h3, _ = run_stack(stack, seed=43)
    assert h1 == h2
    assert h1 != h3


# ---------------------------------------------------------------- O5 Payload


def test_o5_theta_applied_mass_increases():
    backend = make_backend()
    backend.reset(0)
    base = backend.mass_kg
    Payload(mass_kg=8.0, com_offset_m=(0.1, 0.0)).on_reset(backend)
    assert backend.mass_kg == pytest.approx(base + 8.0, abs=float(TOL.o5_payload.mass_abs_tol))


def test_o5_theta_applied_push_response_attenuated():
    # Same impulse: Δv = J/m, so the payload's added mass shrinks the velocity jump.
    impulse = np.array([15.0, 0.0])
    light = make_backend()
    light.reset(0)
    heavy = make_backend()
    heavy.reset(0)
    Payload(mass_kg=15.0, com_offset_m=(0.0, 0.0)).on_reset(heavy)
    light.apply_push(impulse, 0.0)
    heavy.apply_push(impulse, 0.0)
    dv_light = np.linalg.norm(light.step(np.zeros(3)).vel_body)
    dv_heavy = np.linalg.norm(heavy.step(np.zeros(3)).vel_body)
    # 15 kg base, +15 kg payload -> half the Δv.
    assert dv_heavy == pytest.approx(0.5 * dv_light, rel=float(TOL.o5_payload.dv_rel_tol))


def test_o5_overload_saturates_effort_not_friction():
    # A heavy payload on good ground: torque saturates (effort_ratio > 0) while
    # friction stays fine (slip ~ 0) — the O5↔O10 ambiguity symptom.
    backend = make_backend()
    backend.reset(0)
    Payload(mass_kg=30.0, com_offset_m=(0.15, 0.0)).on_reset(backend)
    obs = drive(
        backend, Payload(mass_kg=0.0, com_offset_m=(0.0, 0.0)), cmd=(1.0, 0.0, 0.0), n_steps=80
    )
    assert obs.effort_ratio > 0.0
    assert obs.slip_ratio == pytest.approx(0.0, abs=1e-6)


def test_o5_determinism_gate():
    stack = OperatorStack([Payload(mass_kg=10.0, com_offset_m=(0.1, 0.0))])
    h1, _ = run_stack(stack, seed=5)
    h2, _ = run_stack(stack, seed=5)
    assert h1 == h2


# ---------------------------------------------------------------- O8 Invisible Collider


def test_o8_theta_applied_robot_halts_at_wall():
    wall = Rect(cx=2.0, cy=0.0, hx=0.3, hy=2.0)  # near edge at x = 1.7
    op = InvisibleCollider(region=wall)
    backend = make_backend(start=(0.0, 0.0))
    backend.reset(0)
    op.on_reset(backend)
    obs = drive(backend, op, cmd=(0.8, 0.0, 0.0), n_steps=400)
    near_edge = wall.cx - wall.hx
    tol = float(TOL.o8_invisible_collider.pos_abs_tol)
    # Reached the wall but never crossed it.
    assert obs.pos[0] <= near_edge + tol
    assert obs.pos[0] > near_edge - 0.4
    # The wall is invisible: only the command/measured velocity gap reveals it.
    tracking_error = float(np.linalg.norm(obs.cmd_prev[:2] - obs.vel_body))
    assert tracking_error > 0.35


def test_o8_no_wall_robot_passes():
    backend = make_backend(start=(0.0, 0.0))
    backend.reset(0)
    obs = None
    for _ in range(400):
        obs = backend.step(np.array([0.8, 0.0, 0.0]))
    assert obs.pos[0] > 2.0  # without the wall it sails past x = 2


def test_o8_collision_disabled_counterfactual_retains_visual_parameters():
    wall = Rect(cx=2.0, cy=0.0, hx=0.03, hy=2.0)
    op = InvisibleCollider(
        region=wall,
        height_m=0.22,
        collision_enabled=False,
        geometry_kind="transparent_acrylic",
        optical_transmission=0.96,
    )
    backend = make_backend(start=(0.0, 0.0))
    backend.reset(0)
    op.on_reset(backend)
    obs = drive(backend, op, cmd=(0.8, 0.0, 0.0), n_steps=400)
    assert obs.pos[0] > 2.0
    theta = op.get_privileged_state()
    assert theta["height_m"] == pytest.approx(0.22)
    assert theta["collision_enabled"] == pytest.approx(0.0)
    assert theta["rendered"] == pytest.approx(1.0)
    assert theta["optical_transmission"] == pytest.approx(0.96)


def test_o8_determinism_gate():
    stack = OperatorStack([InvisibleCollider(region=Rect(cx=2.0, cy=0.0, hx=0.3, hy=2.0))])
    h1, _ = run_stack(stack, seed=9, cmd=(0.8, 0.0, 0.0))
    h2, _ = run_stack(stack, seed=9, cmd=(0.8, 0.0, 0.0))
    assert h1 == h2


# ---------------------------------------------------------------- O9 High-Centering


def test_o9_theta_applied_support_loss_inside_region():
    region = Rect(cx=2.0, cy=0.0, hx=1.0, hy=1.0)
    op = HighCentering(region=region, residual_support=0.1)
    backend = make_backend(start=(2.0, 0.0))  # start inside the ridge footprint
    backend.reset(0)
    op.on_reset(backend)
    obs = backend.step(np.array([0.8, 0.0, 0.0]))
    tol = float(TOL.o9_high_centering.support_abs_tol)
    assert obs.support_ratio == pytest.approx(0.1, abs=tol)
    # Beached: a full forward command yields a tiny fraction of nominal accel.
    free = make_backend(start=(2.0, 0.0))
    free.reset(0)
    free_obs = free.step(np.array([0.8, 0.0, 0.0]))
    assert abs(obs.vel_body[0]) < 0.2 * abs(free_obs.vel_body[0])


def test_o9_support_nominal_outside_region():
    region = Rect(cx=5.0, cy=0.0, hx=0.5, hy=0.5)
    backend = make_backend(start=(0.0, 0.0))
    backend.reset(0)
    HighCentering(region=region, residual_support=0.1).on_reset(backend)
    obs = backend.step(np.array([0.5, 0.0, 0.0]))
    assert obs.support_ratio == pytest.approx(1.0)


def test_o9_determinism_gate():
    stack = OperatorStack([HighCentering(region=Rect(cx=2.0, cy=0.0, hx=1.0, hy=1.0))])
    h1, _ = run_stack(stack, seed=4)
    h2, _ = run_stack(stack, seed=4)
    assert h1 == h2


# ---------------------------------------------------------------- O10 Effort-Decay


def test_o10_schedule_is_exact():
    op = EffortDecay(decay_rate_per_s=0.5, floor=0.2, t_start_s=1.0)
    tol = float(TOL.o10_effort_decay.scale_abs_tol)
    assert op.scale_at(0.5) == pytest.approx(1.0, abs=tol)  # before t_start
    assert op.scale_at(1.0) == pytest.approx(1.0, abs=tol)  # at t_start
    assert op.scale_at(2.0) == pytest.approx(0.5, abs=tol)  # 1 s of 0.5/s decay
    assert op.scale_at(10.0) == pytest.approx(0.2, abs=tol)  # floored


def test_o10_theta_applied_saturates_effort_not_friction():
    # Severe decay on good ground: torque saturates (effort_ratio > 0), friction fine.
    op = EffortDecay(decay_rate_per_s=5.0, floor=0.25, t_start_s=0.0)
    backend = make_backend()
    backend.reset(0)
    op.on_reset(backend)
    obs = drive(backend, op, cmd=(0.8, 0.0, 0.0), n_steps=120)
    assert obs.effort_ratio > 0.0
    assert obs.slip_ratio == pytest.approx(0.0, abs=1e-6)


def test_o10_no_decay_is_inert():
    op = EffortDecay(decay_rate_per_s=0.0, floor=0.2, t_start_s=0.0)
    backend = make_backend()
    backend.reset(0)
    op.on_reset(backend)
    obs = drive(backend, op, cmd=(0.6, 0.0, 0.0), n_steps=120)
    assert obs.effort_ratio == pytest.approx(0.0, abs=1e-9)


def test_o10_determinism_gate():
    stack = OperatorStack([EffortDecay(decay_rate_per_s=0.4, floor=0.3, t_start_s=0.5)])
    h1, _ = run_stack(stack, seed=8)
    h2, _ = run_stack(stack, seed=8)
    assert h1 == h2


# ---------------------------------------------------------------- composability


def test_compose_o5_o10_ambiguity_pair_stacks():
    # The two torque-saturation operators stack (Suite-Comp groundwork): θ concatenates
    # and the run stays finite.
    stack = OperatorStack(
        [
            Payload(mass_kg=8.0, com_offset_m=(0.1, 0.0)),
            EffortDecay(decay_rate_per_s=0.3, floor=0.3, t_start_s=1.0),
        ]
    )
    h1, obs = run_stack(stack, seed=11, n_steps=300, cmd=(0.8, 0.0, 0.0))
    assert np.isfinite(obs.pos).all()
    theta = stack.get_privileged_state()
    assert "op0.O5_payload.mass_kg" in theta
    assert "op1.O10_effort_decay.decay_rate_per_s" in theta
    h2, _ = run_stack(stack, seed=11, n_steps=300, cmd=(0.8, 0.0, 0.0))
    assert h1 == h2


def test_compose_o3_o1_friction_operators_stack():
    stack = OperatorStack(
        [
            MuField(region=Rect(cx=1.0, cy=0.0, hx=0.5, hy=0.5), mu_s=0.3, mu_d=0.25),
            Collapse(
                region=Rect(cx=4.0, cy=0.0, hx=0.5, hy=0.5), mu_collapsed=0.08, trigger_dwell_s=0.3
            ),
        ]
    )
    h1, obs = run_stack(stack, seed=13, n_steps=300)
    assert np.isfinite(obs.pos).all()
    theta = stack.get_privileged_state()
    assert "op0.O1_mu_field.mu_d" in theta
    assert "op1.O3_collapse.mu_collapsed" in theta
    h2, _ = run_stack(stack, seed=13, n_steps=300)
    assert h1 == h2
