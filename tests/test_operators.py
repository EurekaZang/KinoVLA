"""Operator gates per QA 5.2: θ-application, determinism, composability (O1, O6, O11).

These run against the CPU surrogate backend; the Isaac-side θ-application checks
(PhysX material readback, root-velocity impulse) live in tests/test_sim_operators.py
behind @pytest.mark.sim and are verified manually on the GPU machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.operators import OPERATORS, MuField, ObsBias, OperatorStack, Push
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
    """Drive a constant command through the stack; return (trajectory hash, last obs)."""
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


# ---------------------------------------------------------------- framework


def test_registry_contains_operators():
    # M1: O1, O6, O11; M2 adds O3, O5, O8, O9, O10; M5 adds O2, O4, O7 (spec §8.2).
    assert set(OPERATORS) == {
        "O1_mu_field",
        "O2_compliance",
        "O3_collapse",
        "O4_tether",
        "O5_payload",
        "O6_push",
        "O7_visual_remap",
        "O8_invisible_collider",
        "O9_high_centering",
        "O10_effort_decay",
        "O11_obs_bias",
    }
    for name, cls in OPERATORS.items():
        assert cls.name == name
        assert cls.axis  # every operator declares its mechanism axis (spec §8.2)


def test_stack_privileged_state_namespacing():
    stack = OperatorStack(
        [
            MuField(region=ICE, mu_s=0.12, mu_d=0.1),
            Push(impulse_xy_ns=np.array([7.5, 0.0]), t_push_s=1.0),
        ]
    )
    theta = stack.get_privileged_state()
    assert theta["op0.O1_mu_field.mu_d"] == pytest.approx(0.1)
    assert theta["op1.O6_push.t_push_s"] == pytest.approx(1.0)
    assert all(isinstance(v, float) for v in theta.values())


# ---------------------------------------------------------------- O1 μ-Field


def test_o1_theta_applied_friction_lookup():
    backend = make_backend()
    backend.reset(0)
    MuField(region=ICE, mu_s=0.12, mu_d=0.1).on_reset(backend)
    tol = float(TOL.o1_mu_field.mu_abs_tol)
    assert backend.friction_at(np.array([3.0, 0.0])) == pytest.approx(0.1, abs=tol)
    assert backend.friction_at(np.array([0.0, 0.0])) == pytest.approx(
        float(SIM_CFG.mu_nominal), abs=tol
    )


def test_o1_theta_applied_behavioral_accel_cap():
    # Full-throttle from rest *on* the patch: first-step acceleration ≈ μ_d · g.
    backend = make_backend(start=(3.0, 0.0))
    backend.reset(0)
    MuField(region=ICE, mu_s=0.12, mu_d=0.1).on_reset(backend)
    obs = backend.step(np.array([1.0, 0.0, 0.0]))
    measured_accel = float(obs.vel_body[0]) / backend.dt
    expected = 0.1 * float(SIM_CFG.gravity)
    assert measured_accel == pytest.approx(expected, rel=float(TOL.o1_mu_field.accel_rel_tol))
    # Same command from rest off the patch is not friction-limited.
    backend.reset(0)
    obs = backend.step(np.array([1.0, 0.0, 0.0]))
    assert float(obs.vel_body[0]) / backend.dt > expected * 3


def test_o1_determinism_gate():
    stack = OperatorStack([MuField(region=ICE, mu_s=0.12, mu_d=0.1)])
    h1, _ = run_stack(stack, seed=42)
    h2, _ = run_stack(stack, seed=42)
    h3, _ = run_stack(stack, seed=43)
    assert h1 == h2
    assert h1 != h3


def test_o1_two_instances_stack():
    # Append semantics: two μ-regions coexist; first-added wins on overlap.
    backend = make_backend()
    backend.reset(0)
    near = MuField(region=Rect(cx=1.0, cy=0.0, hx=0.5, hy=0.5), mu_s=0.3, mu_d=0.25)
    far = MuField(region=Rect(cx=5.0, cy=0.0, hx=0.5, hy=0.5), mu_s=0.06, mu_d=0.05)
    near.on_reset(backend)
    far.on_reset(backend)
    assert backend.friction_at(np.array([1.0, 0.0])) == pytest.approx(0.25)
    assert backend.friction_at(np.array([5.0, 0.0])) == pytest.approx(0.05)


# ---------------------------------------------------------------- O6 Push


def test_o6_theta_applied_velocity_jump():
    impulse = np.array([7.5, 0.0])  # Δv = J/m = 0.5 m/s at 15 kg
    push = Push(impulse_xy_ns=impulse, t_push_s=1.0)
    backend = make_backend()
    obs = backend.reset(0)
    push.on_reset(backend)
    assert not push.fired
    while obs.t < 2.0:
        was_fired = push.fired
        push.on_step(backend, obs.t)
        obs = backend.step(np.zeros(3))
        if push.fired and not was_fired:
            # At rest under zero command, velocity was exactly 0 before the impulse.
            # One control step of first-order decay separates impulse and observation:
            # expected |v| = (J/m) * (1 - dt/τ).
            dv = float(np.linalg.norm(impulse)) / backend.mass_kg
            expected = dv * (1.0 - backend.dt / float(SIM_CFG.tau_track_s))
            assert float(np.linalg.norm(obs.vel_body)) == pytest.approx(
                expected, rel=float(TOL.o6_push.dv_rel_tol)
            )
            return
    pytest.fail("push never fired")


def test_o6_fires_once_and_rearms_on_reset():
    push = Push(impulse_xy_ns=np.array([7.5, 0.0]), t_push_s=0.5)
    backend = make_backend()
    obs = backend.reset(0)
    push.on_reset(backend)
    for _ in range(100):
        push.on_step(backend, obs.t)
        obs = backend.step(np.zeros(3))
    assert push.fired
    # Velocity decays back toward zero after the single impulse — no re-push.
    assert float(np.linalg.norm(obs.vel_body)) < 0.05
    push.on_reset(backend)
    assert not push.fired


def test_o6_finite_pulse_routes_duration_and_application_point():
    push = Push(
        impulse_xy_ns=np.array([0.0, 6.0]),
        t_push_s=1.0,
        duration_s=0.12,
        application_point_body_m=np.array([0.0, -0.085, 0.075]),
    )
    backend = make_backend()
    calls = []

    def record_pulse(impulse, yaw_impulse, duration, application_point):
        calls.append((impulse.copy(), yaw_impulse, duration, application_point.copy()))

    def reject_legacy(*_args):
        pytest.fail("finite O6 must not use the legacy instantaneous-velocity path")

    backend.start_push_pulse = record_pulse
    backend.apply_push = reject_legacy
    push.on_reset(backend)
    push.on_step(backend, 1.0)
    push.on_step(backend, 2.0)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], [0.0, 6.0])
    assert calls[0][1] == pytest.approx(0.0)
    assert calls[0][2] == pytest.approx(0.12)
    np.testing.assert_allclose(calls[0][3], [0.0, -0.085, 0.075])
    theta = push.get_privileged_state()
    assert theta["duration_s"] == pytest.approx(0.12)
    assert theta["application_point_body_y_m"] == pytest.approx(-0.085)


def test_o6_determinism_gate():
    stack = OperatorStack([Push(impulse_xy_ns=np.array([0.0, 7.5]), t_push_s=1.0)])
    h1, _ = run_stack(stack, seed=7)
    h2, _ = run_stack(stack, seed=7)
    assert h1 == h2


# ---------------------------------------------------------------- O11 Obs-Bias


def test_o11_theta_applied_exact_bias_and_drift():
    op = ObsBias(bias={"vel_body": np.array([0.2, -0.1]), "tilt": 0.05}, drift_per_s={"tilt": 0.01})
    backend = make_backend()
    backend.reset(0)
    obs = backend.step(np.array([0.5, 0.0, 0.0]))
    biased = op.transform_obs(obs)
    tol = float(TOL.o11_obs_bias.abs_tol)
    np.testing.assert_allclose(biased.vel_body, obs.vel_body + [0.2, -0.1], atol=tol)
    assert biased.tilt == pytest.approx(obs.tilt + 0.05 + 0.01 * obs.t, abs=tol)
    # Truth preserved: untouched fields identical, backend state never modified.
    np.testing.assert_array_equal(biased.pos, obs.pos)
    assert biased.t == obs.t


def test_o11_rejects_non_imu_fields():
    with pytest.raises(ValueError, match="cannot bias"):
        ObsBias(bias={"pos": 1.0})


def test_o11_privileged_state_flattening():
    op = ObsBias(bias={"vel_body": np.array([0.2, -0.1])}, drift_per_s={"tilt": 0.01})
    theta = op.get_privileged_state()
    assert theta["bias.vel_body.0"] == pytest.approx(0.2)
    assert theta["bias.vel_body.1"] == pytest.approx(-0.1)
    assert theta["drift_per_s.tilt"] == pytest.approx(0.01)


def test_o11_determinism_gate():
    stack = OperatorStack([ObsBias(bias={"vel_body": np.array([0.1, 0.0])})])
    h1, _ = run_stack(stack, seed=3)
    h2, _ = run_stack(stack, seed=3)
    assert h1 == h2


# ---------------------------------------------------------------- composability


def test_compose_o1_o6_smoke():
    stack = OperatorStack(
        [
            MuField(region=ICE, mu_s=0.12, mu_d=0.1),
            Push(impulse_xy_ns=np.array([0.0, 7.5]), t_push_s=2.0),
        ]
    )
    h1, obs = run_stack(stack, seed=11, n_steps=400)
    assert np.isfinite(obs.pos).all()
    theta = stack.get_privileged_state()
    assert "op0.O1_mu_field.mu_d" in theta and "op1.O6_push.impulse_norm_ns" in theta
    h2, _ = run_stack(stack, seed=11, n_steps=400)
    assert h1 == h2


def test_compose_obs_transform_chains():
    stack = OperatorStack(
        [
            ObsBias(bias={"tilt": 0.05}),
            ObsBias(bias={"tilt": 0.02, "yaw_rate": 0.1}),
        ]
    )
    backend = make_backend()
    obs = backend.reset(0)
    biased = stack.transform_obs(obs)
    assert biased.tilt == pytest.approx(obs.tilt + 0.07)
    assert biased.yaw_rate == pytest.approx(obs.yaw_rate + 0.1)
