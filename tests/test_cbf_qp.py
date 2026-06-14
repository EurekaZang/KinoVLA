"""CBF-QP solver gates (spec §6.5) — exact projection + the <1 ms p99 budget (QA 5.3).

Safety-critical (QA 5.2): the QP must return a point that satisfies *every* assembled
constraint to machine precision (no ADMM slack), detect an empty feasible set rather
than silently returning an infeasible point, and solve within the spec §6.5 / QA 5.3
performance budget. These are property-style tests over randomised constraint sets.
"""

from __future__ import annotations

import numpy as np

from kino_vla.shield.modes import load_modes
from kino_vla.shield.qp import build_constraints, project_onto_polytope, solve_cbf_qp
from kino_vla.utils.config import load_config


def test_interior_point_is_unchanged():
    a = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    u, feas = project_onto_polytope([0.2, -0.3], a, b)
    assert feas
    np.testing.assert_allclose(u, [0.2, -0.3])


def test_projects_to_box_corner():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 1.0])
    u, feas = project_onto_polytope([5.0, 5.0], a, b)
    assert feas
    np.testing.assert_allclose(u, [1.0, 1.0], atol=1e-9)


def test_projects_to_single_edge():
    # Only the x<=1 face is violated; projection drops x to 1, keeps y.
    a = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    b = np.array([1.0, 1.0, 2.0, 2.0])
    u, feas = project_onto_polytope([3.0, 0.4], a, b)
    assert feas
    np.testing.assert_allclose(u, [1.0, 0.4], atol=1e-9)


def test_empty_polytope_is_detected():
    # x <= 0 and x >= 1 simultaneously: no feasible point.
    a = np.array([[1.0, 0.0], [-1.0, 0.0]])
    b = np.array([0.0, -1.0])
    _, feas = project_onto_polytope([0.5, 0.0], a, b)
    assert not feas


def test_no_constraints_returns_point():
    u, feas = project_onto_polytope([0.7, -0.2], np.empty((0, 2)), np.empty((0,)))
    assert feas
    np.testing.assert_allclose(u, [0.7, -0.2])


def test_returned_point_satisfies_all_constraints_random():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        m = int(rng.integers(2, 14))
        ang = rng.uniform(0, 2 * np.pi, m)
        a = np.column_stack([np.cos(ang), np.sin(ang)])  # unit normals
        b = rng.uniform(0.2, 1.5, m)  # origin always interior -> feasible
        point = rng.uniform(-3, 3, 2)
        u, feas = project_onto_polytope(point, a, b)
        assert feas
        assert np.all(a @ u <= b + 1e-7), "projection must satisfy every constraint"


def test_projection_is_the_nearest_feasible_point_random():
    # Cross-check the analytic projection against a dense grid search.
    rng = np.random.default_rng(11)
    for _ in range(60):
        m = int(rng.integers(3, 8))
        ang = rng.uniform(0, 2 * np.pi, m)
        a = np.column_stack([np.cos(ang), np.sin(ang)])
        b = rng.uniform(0.4, 1.2, m)
        point = rng.uniform(-2, 2, 2)
        u, feas = project_onto_polytope(point, a, b)
        assert feas
        gx, gy = np.meshgrid(np.linspace(-2, 2, 161), np.linspace(-2, 2, 161))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        ok = np.all(a @ grid.T <= b[:, None] + 1e-9, axis=0)
        gfeas = grid[ok]
        best = gfeas[np.argmin(np.sum((gfeas - point) ** 2, axis=1))]
        # Analytic optimum must be at least as close as the best grid point.
        assert np.linalg.norm(u - point) <= np.linalg.norm(best - point) + 1e-6


def test_constraint_groups_counted():
    cfg = load_config("shield/cbf_v0.yaml")
    mode = load_modes(cfg)["trot"]
    a_s, b_s = mode.support_polygon()
    cons = build_constraints(
        np.zeros(2),
        np.zeros(2),
        a_s,
        b_s,
        omega=mode.omega(cfg.gravity),
        alpha=8.0,
        delta=mode.delta,
        delta_u=mode.delta_u,
        mu=0.8,
        z_c=mode.z_c,
        n_friction_facets=8,
    )
    assert cons.n_cbf == 4 and cons.n_zmp == 4 and cons.n_friction == 8
    assert cons.a_mat.shape == (16, 2)
    assert cons.b_vec.shape == (16,)


def test_qp_p99_solve_time_under_budget():
    # QA 5.3 / spec §6.5: QP solve < 1 ms p99. Measured over a realistic mix of
    # feasible and clamping states with the production constraint count (16 rows).
    cfg = load_config("shield/cbf_v0.yaml")
    mode = load_modes(cfg)["trot"]
    a_s, b_s = mode.support_polygon()
    omega = mode.omega(cfg.gravity)
    rng = np.random.default_rng(3)
    times = []
    for _ in range(5000):
        xi = rng.uniform(-0.25, 0.25, 2)
        u_nom = rng.uniform(-1.0, 1.0, 2)
        cons = build_constraints(
            xi,
            np.zeros(2),
            a_s,
            b_s,
            omega=omega,
            alpha=8.0,
            delta=mode.delta,
            delta_u=mode.delta_u,
            mu=0.8,
            z_c=mode.z_c,
            n_friction_facets=8,
        )
        times.append(solve_cbf_qp(u_nom, cons).solve_time_s)
    p99_ms = float(np.percentile(times, 99) * 1e3)
    assert p99_ms < 1.0, f"QP p99 {p99_ms:.3f} ms exceeds the 1 ms budget"
