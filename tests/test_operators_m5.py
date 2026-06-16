"""M5 operator gates per QA 5.2: θ-application, determinism, composability.

O2 Compliance-Field, O4 Tether/Adhesion, O7 Visual-Physics Remap, on the CPU surrogate.
Each operator gets: (a) θ-application — set a parameter, measure its effect in sim,
assert tolerance; (b) determinism — same seed ⇒ same trajectory hash; (c) composability
— stacks with another operator and θ concatenates.
"""

from __future__ import annotations

import numpy as np

from kino_vla.sim.operators import (
    ComplianceField,
    MuField,
    OperatorStack,
    Tether,
    VisualPhysicsRemap,
)
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.utils.config import load_config
from kino_vla.utils.geometry import Rect
from kino_vla.utils.seeding import trajectory_hash

TOL = load_config("operators/apply_tolerances.yaml")
SIM_CFG = load_config("sim/surrogate.yaml")
REGION = Rect(cx=2.0, cy=0.0, hx=1.0, hy=1.5)


def make_backend(start=(0.0, 0.0)):
    return SurrogateBackend(SIM_CFG, np.asarray(start, dtype=np.float64), 0.0)


def run_stack(stack, seed, n_steps=250, cmd=(0.8, 0.0, 0.0)):
    backend = make_backend()
    obs = backend.reset(seed)
    stack.on_reset(backend)
    positions, velocities = [], []
    for _ in range(n_steps):
        stack.on_step(backend, obs.t)
        obs = backend.step(np.asarray(cmd, dtype=np.float64))
        positions.append(obs.pos)
        velocities.append(obs.vel_body)
    return trajectory_hash(np.asarray(positions), np.asarray(velocities)), obs


def distance_travelled(op, n_steps=250, cmd=(0.8, 0.0, 0.0)):
    backend = make_backend()
    obs = backend.reset(0)
    op.on_reset(backend)
    for _ in range(n_steps):
        op.on_step(backend, obs.t)
        obs = backend.step(np.asarray(cmd, dtype=np.float64))
    return float(obs.pos[0])


# ---------------------------------------------------------------- O2 Compliance-Field


def test_o2_theta_roundtrip_and_sink():
    op = ComplianceField(REGION, k_c=14.0, c_c=6.0, d_sink=0.08)
    state = op.get_privileged_state()
    tol = float(TOL.o2_compliance.theta_abs_tol)
    assert abs(state["k_c"] - 14.0) <= tol
    assert abs(state["c_c"] - 6.0) <= tol
    assert abs(state["d_sink"] - 0.08) <= tol
    # Measured base-height drop inside the patch equals d_sink.
    backend = make_backend()
    obs = backend.reset(0)
    op.on_reset(backend)
    nominal_h = float(SIM_CFG.base_height_m)
    min_h = nominal_h
    for _ in range(150):
        obs = backend.step(np.array([0.8, 0.0, 0.0]))
        min_h = min(min_h, obs.base_height)
    assert abs((nominal_h - min_h) - 0.08) <= float(TOL.o2_compliance.base_height_abs_tol)


def test_o2_higher_stiffness_slows_more():
    soft = distance_travelled(ComplianceField(REGION, k_c=6.0, c_c=3.0, d_sink=0.03))
    stiff = distance_travelled(ComplianceField(REGION, k_c=30.0, c_c=12.0, d_sink=0.18))
    margin = float(TOL.o2_compliance.resistance_monotone_margin_m)
    assert soft - stiff >= margin  # more resistance ⇒ less distance covered


def test_o2_determinism():
    h1, _ = run_stack(OperatorStack([ComplianceField(REGION, 14.0, 6.0, 0.08)]), seed=3)
    h2, _ = run_stack(OperatorStack([ComplianceField(REGION, 14.0, 6.0, 0.08)]), seed=3)
    assert h1 == h2


def test_o2_composable_with_mu_field():
    stack = OperatorStack(
        [ComplianceField(REGION, 14.0, 6.0, 0.08), MuField(Rect(4.0, 0.0, 0.5, 0.5), 0.1, 0.1)]
    )
    state = stack.get_privileged_state()
    assert any("O2_compliance" in k for k in state)
    assert any("O1_mu_field" in k for k in state)
    run_stack(stack, seed=1)  # no crash


# ---------------------------------------------------------------- O4 Tether/Adhesion


def test_o4_theta_roundtrip():
    op = Tether(REGION, k=14.0, d=6.0, l0=0.0, f_break=1.0e9, d_sink=0.08)
    state = op.get_privileged_state()
    tol = float(TOL.o4_tether.theta_abs_tol)
    assert abs(state["k"] - 14.0) <= tol
    assert abs(state["d"] - 6.0) <= tol
    assert abs(state["L_0"]) <= tol
    assert abs(state["F_break"] - 1.0e9) <= 1.0  # large value, loose abs tol


def test_o4_tether_breaks_and_speed_recovers():
    # A weak break force snaps under load; afterwards the robot speeds back up.
    op = Tether(REGION, k=40.0, d=8.0, l0=0.0, f_break=20.0)
    backend = make_backend()
    obs = backend.reset(0)
    op.on_reset(backend)
    speeds = []
    for _ in range(250):
        obs = backend.step(np.array([0.8, 0.0, 0.0]))
        if REGION.contains(obs.pos):
            speeds.append(float(obs.vel_body[0]))
    speeds = np.asarray(speeds)
    # Speed dips while the tether loads, then recovers after it breaks.
    margin = float(TOL.o4_tether.break_speedup_margin_mps)
    assert speeds[-1] - speeds.min() >= margin


def test_o4_determinism():
    h1, _ = run_stack(OperatorStack([Tether(REGION, 14.0, 6.0, 0.0, 1e9)]), seed=5)
    h2, _ = run_stack(OperatorStack([Tether(REGION, 14.0, 6.0, 0.0, 1e9)]), seed=5)
    assert h1 == h2


def test_o4_composable():
    stack = OperatorStack(
        [Tether(REGION, 14.0, 6.0, 0.0, 1e9), MuField(Rect(4.0, 0.0, 0.5, 0.5), 0.1, 0.1)]
    )
    state = stack.get_privileged_state()
    assert any("O4_tether" in k for k in state)
    run_stack(stack, seed=2)


# ---------------------------------------------------------------- O7 Visual-Physics Remap


def test_o7_physics_friction_applied():
    op = VisualPhysicsRemap(REGION, mu_s=0.12, mu_d=0.10, depth_bias_m=0.6)
    backend = make_backend()
    backend.reset(0)
    op.on_reset(backend)
    tol = float(TOL.o7_visual_remap.mu_abs_tol)
    assert abs(backend.friction_at(np.array([2.0, 0.0])) - 0.10) <= tol


def test_o7_appearance_decoupled_from_physics():
    op = VisualPhysicsRemap(REGION, mu_s=0.12, mu_d=0.10, depth_bias_m=0.6)
    region = op.scene_region()
    # Looks like benign solid ground (low visual cost) despite being physically ice.
    assert region.appearance_class == "solid_ground"
    assert region.visual_cost == 0.0
    tol = float(TOL.o7_visual_remap.depth_bias_abs_tol)
    assert abs(region.depth_bias_m - 0.6) <= tol
    assert abs(op.get_privileged_state()["mu_d"] - 0.10) <= tol + 0.02


def test_o7_determinism():
    h1, _ = run_stack(OperatorStack([VisualPhysicsRemap(REGION, 0.12, 0.10, 0.6)]), seed=4)
    h2, _ = run_stack(OperatorStack([VisualPhysicsRemap(REGION, 0.12, 0.10, 0.6)]), seed=4)
    assert h1 == h2


def test_o7_composable():
    stack = OperatorStack(
        [VisualPhysicsRemap(REGION, 0.12, 0.10, 0.6), MuField(Rect(5.0, 0.0, 0.5, 0.5), 0.2, 0.2)]
    )
    state = stack.get_privileged_state()
    assert any("O7_visual_remap" in k for k in state)
    run_stack(stack, seed=6)
