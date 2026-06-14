"""CBF Safety Shield gates (spec §6) — the strictest QA bar (QA 5.2 safety-critical).

Central property (spec §6.6 forward invariance): starting inside the safe set 𝒞, NO
command sequence — however hostile — drives the capture point out of the support
polygon (a fall). Also checks transparency at cruise, the §6.5 back-solve identity,
the §6.5 perception↔safety friction coupling, and the §6.8 infeasibility fallback.

QA 5.2: never weaken a safety assertion to make a test pass.
"""

from __future__ import annotations

import numpy as np

from kino_vla.shield.cbf_shield import CbfShield
from kino_vla.shield.lip import nominal_zmp, step_dcm
from kino_vla.shield.modes import load_modes
from kino_vla.shield.qp import build_constraints, solve_cbf_qp
from kino_vla.sim.types import Obs
from kino_vla.utils.config import load_config

CFG = "shield/cbf_v0.yaml"


def _obs(vx: float = 0.0, vy: float = 0.0) -> Obs:
    return Obs(
        t=0.0,
        pos=np.zeros(2),
        heading=0.0,
        vel_body=np.array([vx, vy]),
        yaw_rate=0.0,
        cmd_prev=np.zeros(3),
        slip_ratio=0.0,
        base_height=0.31,
        tilt=0.0,
        fallen=False,
    )


def test_cruise_command_is_transparent():
    # A cruise command well inside the trot safe set passes through untouched.
    sh = CbfShield(load_config(CFG))
    d = sh.filter(np.array([0.8, 0.0, 0.3]), _obs(0.8, 0.0))
    assert not d.intervened
    np.testing.assert_allclose(d.cmd, [0.8, 0.0, 0.3], atol=1e-6)
    assert d.codes == ()


def test_back_solve_identity_when_feasible():
    # spec §6.5: when the QP does not bind, v_cmd* == v_cmd exactly.
    sh = CbfShield(load_config(CFG))
    for vx, cmdx in [(0.0, 0.3), (0.4, 0.5), (0.6, 0.7), (0.5, 0.2)]:
        d = sh.filter(np.array([cmdx, 0.0, 0.0]), _obs(vx, 0.0))
        if not d.intervened:
            assert abs(d.cmd[0] - cmdx) < 1e-6


def test_yaw_rate_passes_through():
    sh = CbfShield(load_config(CFG))
    d = sh.filter(np.array([0.5, 0.0, 1.234]), _obs(0.5, 0.0))
    assert d.cmd[2] == 1.234


def test_forward_invariance_under_hostile_commands():
    # spec §6.6: the headline guarantee. Stream hostile random velocity commands
    # through the QP and integrate the (divergent) DCM dynamics against a *fixed*
    # planted-feet support polygon; the capture point must never leave the TRUE
    # polygon (the fall condition), and the contracted barrier must stay >= 0.
    cfg = load_config(CFG)
    g, alpha, k_xi = float(cfg.gravity), float(cfg.alpha), float(cfg.k_xi)
    mode = load_modes(cfg)["trot"]
    omega = mode.omega(g)
    a_s, b_s = mode.support_polygon()
    dt = 0.002  # 500 Hz shield rate (spec §6.5)
    rng = np.random.default_rng(0)
    worst_true = -1e9
    worst_h = 1e9
    for _ in range(60):
        xi = rng.uniform(-1, 1, 2) * (b_s[0] - mode.delta) * 0.4  # start interior
        p = np.zeros(2)
        for _ in range(1500):
            v_cmd = rng.uniform(-4.0, 4.0, 2)
            xi_des = p + v_cmd / omega
            u_nom = nominal_zmp(xi, xi_des, k_xi)
            cons = build_constraints(
                xi,
                p,
                a_s,
                b_s,
                omega=omega,
                alpha=alpha,
                delta=mode.delta,
                delta_u=mode.delta_u,
                mu=float(cfg.mu_nominal),
                z_c=mode.z_c,
            )
            qp = solve_cbf_qp(u_nom, cons)
            assert qp.feasible, "planted-feet interior QP must stay feasible"
            p = p + omega * (xi - p) * dt
            xi = step_dcm(xi, qp.u_star, omega, dt)
            worst_true = max(worst_true, float(np.max(a_s @ xi - b_s)))
            worst_h = min(worst_h, float((b_s - mode.delta - a_s @ xi).min()))
    assert worst_true <= 1e-6, f"capture point left the support polygon (fall): {worst_true:.4f}"
    assert worst_h >= -1e-6, f"contracted barrier went negative: {worst_h:.4f}"


def test_forward_invariance_in_velocity_loop_ideal_tracker():
    # spec §6.6 transferred to a velocity-tracking plant. The QP bounds the ZMP, but
    # the deployed input is the back-solved VELOCITY; an IDEAL (instant, uncapped)
    # tracker reaches the commanded equilibrium immediately. The realized capture
    # point ξ = v/ω must never leave the contracted trot safe set C = {|ξ| ≤ lx − δ}.
    # (The surrogate's max-speed clip masks this; an ideal tracker does not.)
    cfg = load_config(CFG)
    trot = load_modes(cfg)["trot"]
    omega = trot.omega(float(cfg.gravity))
    c_bound = trot.lx - trot.delta
    profiles = {
        "max_forward": lambda k, r: np.array([10.0, 0.0, 0.0]),
        "diagonal": lambda k, r: np.array([10.0, 10.0, 0.0]),
        "spin_dash": lambda k, r: np.array([10.0, 0.0, 9.0]),
        "random": lambda k, r: np.append(r.uniform(-10, 10, 2), r.uniform(-9, 9)),
    }
    for fn in profiles.values():
        sh = CbfShield(load_config(CFG))
        rng = np.random.default_rng(1)
        v = np.zeros(2)
        for k in range(500):
            d = sh.filter(fn(k, rng), _obs(v[0], v[1]))
            v = d.cmd[:2].copy()  # ideal tracker: instantly achieve the commanded velocity
            xi = v / omega
            assert np.max(np.abs(xi)) <= c_bound + 1e-6, (
                f"realized capture point {np.max(np.abs(xi)):.4f} left C ({c_bound:.4f})"
            )


def test_nonfinite_command_halts():
    # spec §6.6 covers "any hallucinated command" — NaN/Inf must HALT, not propagate.
    sh = CbfShield(load_config(CFG))
    for bad in ([np.nan, 0.0, 0.0], [0.0, 0.0, np.nan], [np.inf, 0.0, 0.0]):
        d = sh.filter(np.array(bad), _obs(0.5, 0.0))
        assert np.all(np.isfinite(d.cmd))
        np.testing.assert_allclose(d.cmd, [0.0, 0.0, 0.0])
        assert "HALT" in d.codes and "NONFINITE_COMMAND" in d.codes
        assert d.intervened
    # A non-finite measured velocity must also halt rather than corrupt the state.
    bad_obs = Obs(
        t=0.0,
        pos=np.zeros(2),
        heading=0.0,
        vel_body=np.array([np.nan, 0.0]),
        yaw_rate=0.0,
        cmd_prev=np.zeros(3),
        slip_ratio=0.0,
        base_height=0.31,
        tilt=0.0,
        fallen=False,
    )
    d = sh.filter(np.array([0.5, 0.0, 0.0]), bad_obs)
    assert "HALT" in d.codes and np.all(np.isfinite(d.cmd))


def test_committed_mode_switch_round_trip():
    # The admission→commit path (spec §6.7): an admitted target is committed via
    # set_mode() (the M7 planner's job); confirm the round trip actually changes mode.
    sh = CbfShield(load_config(CFG))
    assert sh.mode == "trot"
    adm = sh.admit_mode_switch("crawl", _obs(0.2, 0.0))
    assert adm.approved
    sh.set_mode(adm.target)
    assert sh.mode == "crawl"


def test_hostile_command_clamped_below_capture_speed():
    # A sustained max command must settle below the surrogate's nominal capture
    # speed (1.2 m/s) under the trot stance (no fallback engaged).
    sh = CbfShield(load_config(CFG))
    v = 0.0
    for _ in range(400):
        d = sh.filter(np.array([5.0, 0.0, 0.0]), _obs(v, 0.0))
        v += (d.cmd[0] - v) * 0.2  # first-order command tracking
    assert v < 1.2, f"shielded speed {v:.3f} reached the capture limit"
    assert d.intervened


def test_friction_coupling_tightens_with_low_mu():
    # spec §6.5: a lower μ̂ shrinks the realizable ZMP set, so the same hostile
    # command yields a smaller (or infeasible→halt) output. Compare high vs low μ̂.
    cmd = np.array([3.0, 0.0, 0.0])
    obs = _obs(0.6, 0.0)
    hi = CbfShield(load_config(CFG))
    hi.set_mu_estimate(0.8)
    lo = CbfShield(load_config(CFG))
    lo.set_mu_estimate(0.08)
    d_hi = hi.filter(cmd, obs)
    d_lo = lo.filter(cmd, obs)
    assert abs(d_lo.cmd[0]) <= abs(d_hi.cmd[0]) + 1e-9


def test_infeasibility_triggers_fallback_then_halt():
    # spec §6.8: extreme low μ̂ at speed makes even the brace mode infeasible, so the
    # shield halts (zero command) and emits structured codes — no slack softening.
    sh = CbfShield(load_config(CFG))
    sh.set_mu_estimate(0.02)
    d = sh.filter(np.array([4.0, 0.0, 0.0]), _obs(1.0, 0.0))
    assert "HALT" in d.codes and "EXECUTION_REPORT" in d.codes
    np.testing.assert_allclose(d.cmd[:2], [0.0, 0.0])
    assert sh.stats.n_halt >= 1


def test_stats_and_reset():
    sh = CbfShield(load_config(CFG))
    for _ in range(10):
        sh.filter(np.array([5.0, 0.0, 0.0]), _obs(1.0, 0.0))
    assert sh.stats.n_calls == 10
    assert len(sh.stats.qp_times_s) == 10
    sh.reset()
    assert sh.stats.n_calls == 0
    assert sh.mode == "trot"
