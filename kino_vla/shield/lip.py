"""Linear-inverted-pendulum / DCM reduced-order model (spec §6.1, §6.4).

The shield adjudicates at the ZMP level of the LIP. State is the CoM horizontal
position/velocity ``x = [p, v]``; the divergent component (DCM / capture point)
``ξ = p + v/ω`` carries the only unstable mode, so constraining ξ inside the support
polygon (0-step capturability, spec §6.2) constrains the whole instability.

This module is frame-agnostic: the shield evaluates it instantaneously in a
body frame pinned at the CoM (p = 0, so ξ = v/ω), while the forward-invariance test
(spec §6.6) integrates the closed DCM dynamics ``ξ̇ = ω(ξ − u)`` against a *fixed*
support polygon to verify ξ never leaves it.
"""

from __future__ import annotations

import math

import numpy as np


def dcm(p: np.ndarray, v: np.ndarray, omega: float) -> np.ndarray:
    """Divergent component of motion ξ = p + v/ω (spec §6.1)."""
    p = np.asarray(p, dtype=np.float64).reshape(2)
    v = np.asarray(v, dtype=np.float64).reshape(2)
    return p + v / omega


def nominal_zmp(xi: np.ndarray, xi_des: np.ndarray, k_xi: float) -> np.ndarray:
    """DCM tracking law u_nom = ξ + K_ξ(ξ − ξ_des) (spec §6.4).

    Under u_nom the DCM obeys ξ̇ = −ω K_ξ (ξ − ξ_des): it converges exponentially to
    ξ_des, i.e. the robot cruises at the commanded velocity.
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(2)
    xi_des = np.asarray(xi_des, dtype=np.float64).reshape(2)
    return xi + k_xi * (xi - xi_des)


def back_solve_v_cmd(
    v: np.ndarray, xi: np.ndarray, u_star: np.ndarray, omega: float, k_xi: float
) -> np.ndarray:
    """Filtered Sport-Client velocity from the safe ZMP: v* = v + (ω/K_ξ)(ξ − u*) (spec §6.5)."""
    v = np.asarray(v, dtype=np.float64).reshape(2)
    xi = np.asarray(xi, dtype=np.float64).reshape(2)
    u_star = np.asarray(u_star, dtype=np.float64).reshape(2)
    return v + (omega / k_xi) * (xi - u_star)


def omega_of(z_c: float, gravity: float) -> float:
    """LIP natural frequency ω = √(g / z_c) (spec §6.1)."""
    return math.sqrt(gravity / z_c)


def step_dcm(xi: np.ndarray, u: np.ndarray, omega: float, dt: float) -> np.ndarray:
    """Exact one-step integration of ξ̇ = ω(ξ − u) under a zero-order-hold ZMP u.

    Closed-form (u constant over [0, dt]): ξ(dt) = u + e^{ω dt}(ξ₀ − u). Used by the
    forward-invariance property test (spec §6.6) — it is the divergent half of the
    LIP, so a too-fast capture point provably runs away unless the CBF reins u in.
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(2)
    u = np.asarray(u, dtype=np.float64).reshape(2)
    return u + math.exp(omega * dt) * (xi - u)
