"""CBF-QP Safety Shield (spec §6) — LLM proposes, the barrier adjudicates.

Replaces the M1 ``PassThroughShield`` stub. Every Sport-Client velocity command is
projected, at the ZMP level of the LIP reduced-order model, onto the safe set 𝒞_σ of
the current posture/gait mode so the capture point can never leave the contracted
support polygon (0-step capturability — spec §6.2, §6.6). Pipeline per call:

    obs → LIP state (ξ, v) → ξ_des from v_cmd → u_nom (DCM tracking, §6.4)
        → CBF-QP project u_nom onto 𝒞_σ (§6.5) → back-solve v_cmd* (§6.5)

On QP infeasibility no slack variable softens the safety constraints (§6.8): the
shield escalates through a fallback mode chain (brace: lower CoM, widen stance) and,
failing that, halts (zero command + Execution Report). Discrete mode switches are
gated by the admission rule (§6.7). Intervention magnitude ‖u*−u_nom‖, activation
frequency, and per-stage latency are logged for the §6.8 / §6.9 metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from kino_vla.shield.lip import back_solve_v_cmd, dcm, nominal_zmp
from kino_vla.shield.modes import Mode, load_modes
from kino_vla.shield.passthrough import ShieldDecision
from kino_vla.shield.qp import QpResult, build_constraints, project_onto_polytope, solve_cbf_qp
from kino_vla.sim.types import Obs
from kino_vla.utils.config import Config


@dataclass(frozen=True)
class AdmissionResult:
    """Outcome of a discrete mode-switch admission check (spec §6.7)."""

    approved: bool
    target: str
    margin: float  # min_j h_j^{σ'}(x) − ε_switch ; ≥ 0 ⟺ approved
    code: str


@dataclass
class ShieldStats:
    """Running CBF intervention / fallback / latency metrics (spec §6.8, §6.9)."""

    n_calls: int = 0
    n_intervened: int = 0
    n_infeasible: int = 0
    n_halt: int = 0
    interventions: list[float] = field(default_factory=list)
    qp_times_s: list[float] = field(default_factory=list)
    shield_times_s: list[float] = field(default_factory=list)

    def intervention_rate(self) -> float:
        return self.n_intervened / self.n_calls if self.n_calls else 0.0

    def mean_intervention(self) -> float:
        return float(np.mean(self.interventions)) if self.interventions else 0.0

    def qp_p99_ms(self) -> float:
        return float(np.percentile(self.qp_times_s, 99) * 1e3) if self.qp_times_s else 0.0

    def shield_p99_ms(self) -> float:
        return float(np.percentile(self.shield_times_s, 99) * 1e3) if self.shield_times_s else 0.0


class CbfShield:
    """CBF-QP command filter implementing the M3 ``Shield`` protocol (spec §6)."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._g = float(cfg.gravity)
        self._alpha = float(cfg.alpha)
        self._k_xi = float(cfg.k_xi)
        self._n_facets = int(cfg.friction_facets)
        self._mu = float(cfg.mu_nominal)
        self._eps_switch = float(cfg.epsilon_switch)
        self._modes: dict[str, Mode] = load_modes(cfg)
        self._mode_name = str(cfg.default_mode)
        if self._mode_name not in self._modes:
            raise ValueError(f"default_mode {self._mode_name!r} not in modes {list(self._modes)}")
        self._fallback_modes = [str(m) for m in cfg.get("fallback_modes", [])]
        self.stats = ShieldStats()

    # ------------------------------------------------------------------ state
    @property
    def mode(self) -> str:
        return self._mode_name

    def modes_table(self) -> dict[str, Mode]:
        """The configured posture/gait modes (read by the Primitive Compiler)."""
        return dict(self._modes)

    def set_mode(self, name: str) -> None:
        """Force the active mode (used after an admitted switch — spec §6.7)."""
        if name not in self._modes:
            raise ValueError(f"unknown mode {name!r}; have {list(self._modes)}")
        self._mode_name = name

    def set_mu_estimate(self, mu: float) -> None:
        """Update the friction estimate feeding the cone constraint (spec §6.5).

        The perception↔safety coupling point: the Kino-Tokens μ̂ head (M4) calls this
        so a detected ice patch tightens the realizable-ZMP set and the shield turns
        conservative. Until M4 the shield runs on the config nominal μ.
        """
        self._mu = max(1e-3, float(mu))

    def reset(self) -> None:
        self._mode_name = str(self._cfg.default_mode)
        self._mu = float(self._cfg.mu_nominal)
        self.stats = ShieldStats()

    # ------------------------------------------------------------- core solve
    def _solve_for_mode(
        self, mode: Mode, v: np.ndarray, v_cmd: np.ndarray
    ) -> tuple[np.ndarray, QpResult]:
        """Run the full LIP→QP→back-solve pipeline for one mode; return (v_cmd*, qp)."""
        omega = mode.omega(self._g)
        p = np.zeros(2)  # body frame pinned at the CoM ground projection
        xi = dcm(p, v, omega)
        xi_des = dcm(p, v_cmd, omega)
        u_nom = nominal_zmp(xi, xi_des, self._k_xi)
        a_s, b_s = mode.support_polygon()
        cons = build_constraints(
            xi,
            p,
            a_s,
            b_s,
            omega=omega,
            alpha=self._alpha,
            delta=mode.delta,
            delta_u=mode.delta_u,
            mu=self._mu,
            z_c=mode.z_c,
            n_friction_facets=self._n_facets,
        )
        qp = solve_cbf_qp(u_nom, cons)
        v_cmd_star = back_solve_v_cmd(v, xi, qp.u_star, omega, self._k_xi)
        # Steady-state safety — transfer §6.6 to the velocity-tracking plant. The QP
        # bounds dh/dt at the *measured* state, but the back-solved velocity drives a
        # tracker toward the equilibrium ξ_ss = v_cmd*/ω, which (since δ > δ_u) the ZMP
        # constraint alone permits as far as b − δ_u — i.e. into the δ−δ_u margin
        # *outside* the contracted safe set C = {a_j·ξ ≤ b_j − δ}. Project ξ_ss back
        # onto C (using the barrier margin δ, not δ_u) so the COMMANDED equilibrium can
        # never leave C; transparent at cruise (ξ_ss already deep inside C).
        xi_ss = v_cmd_star / omega
        xi_ss_safe, _ = project_onto_polytope(xi_ss, a_s, b_s - mode.delta)
        v_cmd_star = omega * xi_ss_safe
        return v_cmd_star, qp

    def filter(self, cmd: np.ndarray, obs: Obs) -> ShieldDecision:
        """Project ``cmd`` onto the safe set; fall back on infeasibility (spec §6.5, §6.8)."""
        t0 = time.perf_counter()
        cmd = np.asarray(cmd, dtype=np.float64).reshape(3)
        v = np.asarray(obs.vel_body, dtype=np.float64).reshape(2)

        # Non-finite input is in the hostile-command space the spec §6.6 claim covers
        # ("无论 LLM 输出任何幻觉指令"). A NaN/Inf u_nom would still project to a finite
        # polytope vertex (vertices do not depend on the point) and silently propagate,
        # so reject it up front to a HALT (zero command) — never softened.
        if not (np.all(np.isfinite(cmd)) and np.all(np.isfinite(v))):
            self.stats.n_calls += 1
            self.stats.n_halt += 1
            self.stats.qp_times_s.append(0.0)
            self.stats.interventions.append(0.0)
            self.stats.n_intervened += 1
            self.stats.shield_times_s.append(time.perf_counter() - t0)
            return ShieldDecision(
                cmd=np.zeros(3, dtype=np.float64),
                intervened=True,
                codes=("HALT", "NONFINITE_COMMAND", "EXECUTION_REPORT"),
            )

        v_cmd = cmd[:2]
        wz = float(cmd[2])

        codes: list[str] = []
        mode = self._modes[self._mode_name]
        v_star, qp = self._solve_for_mode(mode, v, v_cmd)

        # Infeasibility fallback chain (spec §6.8) — no constraint softening.
        if not qp.feasible:
            self.stats.n_infeasible += 1
            for fb_name in self._fallback_modes:
                fb = self._modes.get(fb_name)
                if fb is None:
                    continue
                v_star, qp = self._solve_for_mode(fb, v, v_cmd)
                if qp.feasible:
                    codes.append(f"INFEASIBLE_FALLBACK:{fb_name}")
                    break
            if not qp.feasible:  # exhausted → damping halt + Execution Report
                self.stats.n_halt += 1
                codes.append("HALT")
                codes.append("EXECUTION_REPORT")
                v_star = np.zeros(2)
                wz = 0.0  # a halt is a full stop — do not let hostile yaw pass through

        out = np.array([v_star[0], v_star[1], wz], dtype=np.float64)
        intervened = bool(np.linalg.norm(out[:2] - v_cmd) > 1e-6)
        if intervened and "HALT" not in codes:
            codes.append("CBF_INTERVENED")

        self.stats.n_calls += 1
        self.stats.qp_times_s.append(qp.solve_time_s)
        self.stats.interventions.append(qp.intervention)
        if intervened:
            self.stats.n_intervened += 1
        self.stats.shield_times_s.append(time.perf_counter() - t0)
        return ShieldDecision(cmd=out, intervened=intervened, codes=tuple(codes))

    # ----------------------------------------------------- mode-switch admission
    def admit_mode_switch(self, target: str, obs: Obs) -> AdmissionResult:
        """Admit σ→σ' iff h_j^{σ'}(x) ≥ ε_switch ∀j, else reject with a code (spec §6.7).

        The current state must already lie inside the *new* mode's contracted safe
        set before the switch is allowed — this is what makes the piecewise-switched
        system inductively forward-invariant.
        """
        if target not in self._modes:
            return AdmissionResult(False, target, float("-inf"), f"REJECT: unknown mode {target!r}")
        tgt = self._modes[target]
        omega = tgt.omega(self._g)
        xi = dcm(np.zeros(2), np.asarray(obs.vel_body, dtype=np.float64).reshape(2), omega)
        h = tgt.barriers(xi)
        h_min = float(h.min())
        margin = h_min - self._eps_switch
        if margin >= 0.0:
            return AdmissionResult(
                True, target, margin, f"ADMIT: x ∈ C_{target}, margin={margin:+.3f}m"
            )
        return AdmissionResult(
            False,
            target,
            margin,
            f"REJECT: x ∉ C_{target}, h_min={h_min:+.3f}m < ε={self._eps_switch:.3f}",
        )
