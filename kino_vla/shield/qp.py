"""Exact CBF-QP solver and constraint assembly (spec §6.5).

The CBF Safety Shield's QP is, by construction, the *Euclidean projection* of the
nominal ZMP ``u_nom`` onto the intersection of half-planes formed by the CBF safety
constraints, the ZMP-realizability constraints, and the friction-cone facets:

    u* = argmin ½‖u − u_nom‖²   s.t.   A u ≤ b                              (spec §6.5)

with ``u ∈ ℝ²``. Because the problem is two-dimensional, its optimum is the closest
of: ``u_nom`` itself (if feasible), the foot of the perpendicular onto each single
constraint line, or each pairwise constraint intersection (a polytope vertex). We
enumerate these candidates and pick the feasible minimiser. This is *exact* (the
returned ``u*`` satisfies every constraint to machine precision — no ADMM tolerance
slack), deterministic, dependency-free, and trivially within the <1 ms budget for the
O(M) constraints of a quadruped support polygon. An empty feasible set (e.g. support
polygon collapse under violent slip) is detected here and routed to the §6.8 fallback
chain by the shield — the QP never softens a safety constraint with a slack variable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


def project_onto_polytope(
    point: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    feas_tol: float = 1e-7,
) -> tuple[np.ndarray, bool]:
    """Euclidean projection of ``point`` onto ``{u ∈ ℝ² : a_mat u ≤ b_vec}``.

    Returns ``(u_star, feasible)``. ``feasible`` is ``False`` iff the polytope is
    empty, in which case ``u_star`` is ``point`` unchanged (the caller falls back).
    Exact for the 2-D projection: candidates are the point, the perpendicular foot
    on each constraint line, and every pairwise line intersection; the nearest
    feasible candidate is the projection.
    """
    point = np.asarray(point, dtype=np.float64).reshape(2)
    a_mat = np.asarray(a_mat, dtype=np.float64)
    b_vec = np.asarray(b_vec, dtype=np.float64)
    # Defense-in-depth: a non-finite target has no projection — report infeasible so
    # the shield routes it to the halt fallback rather than to an arbitrary vertex
    # (vertices are point-independent, so a NaN point would otherwise pass silently).
    if not np.all(np.isfinite(point)):
        return point.copy(), False
    m = a_mat.shape[0]
    if m == 0:
        return point.copy(), True

    cands = [point.reshape(1, 2)]

    # Perpendicular foot onto each constraint line a_i·u = b_i.
    nrm2 = np.einsum("ij,ij->i", a_mat, a_mat)
    good = nrm2 > 1e-12
    if good.any():
        ag, bg, ng = a_mat[good], b_vec[good], nrm2[good]
        viol = (ag @ point - bg) / ng
        cands.append(point[None, :] - ag * viol[:, None])

    # Pairwise line intersections (polytope vertices).
    if m >= 2:
        ii, jj = np.triu_indices(m, k=1)
        a_i, a_j, b_i, b_j = a_mat[ii], a_mat[jj], b_vec[ii], b_vec[jj]
        det = a_i[:, 0] * a_j[:, 1] - a_i[:, 1] * a_j[:, 0]
        nz = np.abs(det) > 1e-12
        if nz.any():
            d = det[nz]
            vx = (b_i[nz] * a_j[nz, 1] - b_j[nz] * a_i[nz, 1]) / d
            vy = (a_i[nz, 0] * b_j[nz] - a_j[nz, 0] * b_i[nz]) / d
            cands.append(np.column_stack([vx, vy]))

    cand = np.vstack(cands)
    feasible_mask = np.all(a_mat @ cand.T <= b_vec[:, None] + feas_tol, axis=0)
    if not feasible_mask.any():
        return point.copy(), False
    cf = cand[feasible_mask]
    d2 = np.einsum("ij,ij->i", cf - point, cf - point)
    return cf[int(np.argmin(d2))].copy(), True


@dataclass(frozen=True)
class CbfConstraints:
    """Assembled half-plane constraints ``A u ≤ b`` for one control step.

    ``n_cbf`` / ``n_zmp`` / ``n_friction`` record how many rows each physical group
    contributes (CBF safety, ZMP-realizability, friction cone — spec §6.5), and
    ``h`` holds the per-edge barrier values ``h_j(x)`` (spec §6.2) at the current
    state for diagnostics and the forward-invariance check.
    """

    a_mat: np.ndarray  # (M, 2)
    b_vec: np.ndarray  # (M,)
    h: np.ndarray  # (n_support,) barrier values h_j(x)
    n_cbf: int
    n_zmp: int
    n_friction: int


def build_constraints(
    xi: np.ndarray,
    p: np.ndarray,
    support_a: np.ndarray,
    support_b: np.ndarray,
    *,
    omega: float,
    alpha: float,
    delta: float,
    delta_u: float,
    mu: float,
    z_c: float,
    n_friction_facets: int = 8,
) -> CbfConstraints:
    """Assemble the CBF-QP constraint set (spec §6.3, §6.5).

    ``support_a`` / ``support_b`` are the support-polygon half-planes
    ``a_j·y ≤ b_j`` (unit outward normals). ``xi`` is the current DCM / capture
    point and ``p`` the CoM ground projection, both in the same planar frame as the
    polygon. ``mu`` is the (online-estimated, spec §6.5) friction coefficient that
    couples perception to the shield — lower ``mu`` shrinks the realizable ZMP set.
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(2)
    p = np.asarray(p, dtype=np.float64).reshape(2)
    support_a = np.asarray(support_a, dtype=np.float64)
    support_b = np.asarray(support_b, dtype=np.float64)
    n_s = support_a.shape[0]

    # Barrier values h_j(x) = (b_j − δ) − a_j·ξ  (spec §6.2).
    h = (support_b - delta) - support_a @ xi

    # CBF safety:  ω a_j·u ≥ ω a_j·ξ − α h_j  ⟹  (−ω a_j)·u ≤ α h_j − ω a_j·ξ.
    a_xi = support_a @ xi
    cbf_a = -omega * support_a
    cbf_b = alpha * h - omega * a_xi

    # ZMP realizability: the ZMP must lie inside the support polygon (margin δ_u).
    zmp_a = support_a.copy()
    zmp_b = support_b - delta_u

    # Friction cone ‖p − u‖ ≤ μ z_c (spec §6.5), inner regular-polygon approximation
    # (conservative: feasible set ⊆ disk). Facet k: d_k·(u − p) ≤ r cos(π/N).
    n = max(3, int(n_friction_facets))
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    r = max(0.0, float(mu) * float(z_c))
    fric_a = dirs
    fric_b = r * np.cos(np.pi / n) + dirs @ p

    a_mat = np.vstack([cbf_a, zmp_a, fric_a])
    b_vec = np.concatenate([cbf_b, zmp_b, fric_b])
    return CbfConstraints(a_mat=a_mat, b_vec=b_vec, h=h, n_cbf=n_s, n_zmp=n_s, n_friction=n)


@dataclass(frozen=True)
class QpResult:
    """One CBF-QP solve outcome (spec §6.5, §6.8 intervention metrics)."""

    u_star: np.ndarray  # (2,) filtered ZMP
    u_nom: np.ndarray  # (2,) nominal ZMP
    feasible: bool
    intervention: float  # ‖u* − u_nom‖ (spec §6.8 metric)
    h_min: float  # min_j h_j(x), the tightest barrier (spec §6.2)
    n_constraints: int
    solve_time_s: float


def solve_cbf_qp(u_nom: np.ndarray, constraints: CbfConstraints) -> QpResult:
    """Project ``u_nom`` onto the safe set; time the solve (spec §6.5, §6.9)."""
    u_nom = np.asarray(u_nom, dtype=np.float64).reshape(2)
    t0 = time.perf_counter()
    u_star, feasible = project_onto_polytope(u_nom, constraints.a_mat, constraints.b_vec)
    dt = time.perf_counter() - t0
    return QpResult(
        u_star=u_star,
        u_nom=u_nom.copy(),
        feasible=feasible,
        intervention=float(np.linalg.norm(u_star - u_nom)),
        h_min=float(constraints.h.min()) if constraints.h.size else float("inf"),
        n_constraints=int(constraints.a_mat.shape[0]),
        solve_time_s=dt,
    )
