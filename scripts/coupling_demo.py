"""μ̂ → CBF-shield coupling demo — the M4 exit criterion (spec §6.5).

Drives the surrogate Go2 straight across an O1 ice patch with the Kino-Tokens μ̂ head
feeding ``CbfShield.set_mu_estimate`` through the anomaly-gated coupler (spec §4 #4).
Logs, per control step, the monitor arousal, the extractor's μ̂, and the shield's
friction-cone radius ``r = μ̂·z_c`` — demonstrating that a detected ice patch TIGHTENS
the QP friction bound (the shield turns conservative) while firm ground leaves it at
nominal. Off the ice the gate is closed and the extractor compute is skipped, so μ̂
relaxes to nominal and the bound is the M3 baseline; the coupling only bites on ice.

Artifacts: outputs/tokens/coupling_demo.json (full per-step trace) and .md (summary
table). Importable: tests/test_extractor-style gates call ``run_coupling_demo``.

Usage:
    python scripts/coupling_demo.py [--seed N] [--speed 0.6]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kino_vla.loop import run_episode
from kino_vla.monitor.event import MonitorEvent
from kino_vla.monitor.learned_monitor import load_deployed_monitor
from kino_vla.shield.cbf_shield import CbfShield
from kino_vla.sim.operators import MuField, OperatorStack
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.sim.types import Obs
from kino_vla.tokens.coupler import MuEstimateCoupler
from kino_vla.tokens.extractor import Extractor
from kino_vla.utils.config import REPO_ROOT, load_config
from kino_vla.utils.geometry import Rect

DEFAULT_CKPT = "outputs/tokens/extractor"


class _StraightDrive:
    """Constant forward cruise: crosses the hazard (no FSM detour) so the coupling is
    actually exercised — the robot must experience the ice for μ̂ to react."""

    def __init__(self, speed: float) -> None:
        self._cmd = np.array([speed, 0.0, 0.0], dtype=np.float64)

    def on_event(self, event: MonitorEvent) -> bool:  # noqa: ARG002 — straight driver ignores events
        return False

    def step(self, obs: Obs) -> np.ndarray:  # noqa: ARG002 — open-loop forward command
        return self._cmd.copy()


@dataclass
class CouplingDemoResult:
    """Summary of the coupling demo (firm vs ice μ̂ and friction radius)."""

    mu_hat_firm: float  # mean μ̂ on firm ground (gate closed ⇒ nominal)
    mu_hat_ice: float  # min μ̂ reached on the ice (gate open ⇒ extractor estimate)
    r_firm: float  # friction-cone radius r=μ̂·z_c on firm ground [m]
    r_ice: float  # friction-cone radius on ice [m] (tightened)
    fired: bool  # monitor fired on the ice
    n_ice_steps: int
    rows: list[dict] = field(default_factory=list)

    @property
    def tightening_ratio(self) -> float:
        """How much the friction bound shrank on ice (r_firm / r_ice)."""
        return self.r_firm / self.r_ice if self.r_ice > 0 else float("inf")


def run_coupling_demo(
    *,
    seed: int = 0,
    ckpt: str = DEFAULT_CKPT,
    speed: float = 0.6,
    max_time_s: float = 12.0,
    ice_mu_d: float = 0.08,
    monitor: object | None = None,
) -> CouplingDemoResult:
    """Run the straight ice crossing with μ̂→shield coupling; return firm-vs-ice stats."""
    ext_cfg = load_config("tokens/extractor_v0.yaml")
    sim_cfg = load_config("sim/surrogate.yaml")
    backend = SurrogateBackend(sim_cfg, np.zeros(2), 0.0)
    ice_rect = Rect(cx=2.5, cy=0.0, hx=1.0, hy=1.2)
    operators = OperatorStack([MuField(region=ice_rect, mu_s=ice_mu_d + 0.02, mu_d=ice_mu_d)])
    monitor = monitor if monitor is not None else load_deployed_monitor(backend.dt)
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt
    extractor = Extractor.load(ext_cfg, ckpt_path, device="cpu")
    coupler = MuEstimateCoupler(ext_cfg, extractor, shield)
    policy = _StraightDrive(speed)

    rows: list[dict] = []

    def on_step(obs: Obs, event: MonitorEvent | None, cmd: np.ndarray, decision) -> None:  # noqa: ANN001
        # Called after the shield filters (μ̂ already pushed): read the LIVE shield state.
        on_ice = bool(ice_rect.contains(obs.pos))
        rows.append(
            {
                "t": round(float(obs.t), 3),
                "x": round(float(obs.pos[0]), 3),
                "on_ice": on_ice,
                "slip": round(float(obs.slip_ratio), 3),
                "anomaly": round(float(monitor.anomaly_score), 3),
                "mu_hat": round(float(shield.mu_estimate), 4),
                "r_friction_m": round(float(shield.friction_radius()), 4),
                "cmd_vx": round(float(cmd[0]), 3),
                "vx_shielded": round(float(decision.cmd[0]), 3),
            }
        )

    episode = run_episode(
        backend,
        operators,
        monitor,
        policy,
        shield,
        seed=seed,
        goal_xy=np.array([8.0, 0.0]),
        goal_tol_m=0.3,
        max_time_s=max_time_s,
        on_step=on_step,
        coupler=coupler,
    )

    firm = [r for r in rows if r["x"] < ice_rect.cx - ice_rect.hx]  # strictly before the patch
    ice = [r for r in rows if r["on_ice"]]
    mu_firm = float(np.mean([r["mu_hat"] for r in firm])) if firm else float("nan")
    mu_ice = float(min((r["mu_hat"] for r in ice), default=float("nan")))
    r_firm = float(np.mean([r["r_friction_m"] for r in firm])) if firm else float("nan")
    r_ice = float(min((r["r_friction_m"] for r in ice), default=float("nan")))
    return CouplingDemoResult(
        mu_hat_firm=mu_firm,
        mu_hat_ice=mu_ice,
        r_firm=r_firm,
        r_ice=r_ice,
        fired=episode.monitor_fired,
        n_ice_steps=len(ice),
        rows=rows,
    )


def _write_artifacts(res: CouplingDemoResult, tol) -> bool:  # noqa: ANN001
    out_dir = REPO_ROOT / "outputs" / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coupling_demo.json").write_text(
        json.dumps(
            {
                "mu_hat_firm": res.mu_hat_firm,
                "mu_hat_ice": res.mu_hat_ice,
                "r_firm_m": res.r_firm,
                "r_ice_m": res.r_ice,
                "tightening_ratio": res.tightening_ratio,
                "monitor_fired": res.fired,
                "rows": res.rows,
            },
            indent=2,
        )
    )
    max_ice = float(tol.coupling.max_mu_hat_on_ice)
    min_firm = float(tol.coupling.min_mu_hat_on_firm)
    ok = (
        res.fired
        and res.mu_hat_ice <= max_ice
        and res.mu_hat_firm >= min_firm
        and res.r_ice < res.r_firm
    )
    lines = [
        "# μ̂ → CBF-shield coupling demo (M4, spec §6.5)",
        "",
        "Straight ice crossing; the Kino-Tokens μ̂ head feeds the shield friction cone.",
        "",
        "| quantity | firm ground | on ice | gate |",
        "| --- | --- | --- | --- |",
        f"| μ̂ (friction estimate) | {res.mu_hat_firm:.3f} | {res.mu_hat_ice:.3f} | "
        f"≤{max_ice} ice / ≥{min_firm} firm |",
        f"| friction radius r=μ̂·z_c [m] | {res.r_firm:.4f} | {res.r_ice:.4f} | r_ice < r_firm |",
        "",
        f"- monitor fired on ice: **{res.fired}**",
        f"- friction bound tightened by **{res.tightening_ratio:.1f}×** on detected ice",
        f"- ice steps logged: {res.n_ice_steps}",
        "",
        f"**{'PASS' if ok else 'FAIL'}: μ̂→shield coupling**",
    ]
    (out_dir / "coupling_demo.md").write_text("\n".join(lines) + "\n")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="μ̂→shield coupling demo (M4)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.6)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = parser.parse_args()

    tol = load_config("tolerances.yaml")
    res = run_coupling_demo(seed=args.seed, ckpt=args.ckpt, speed=args.speed)
    ok = _write_artifacts(res, tol)
    print("=== μ̂ → CBF-shield coupling (spec §6.5) ===")
    print(f"  monitor fired on ice : {res.fired}")
    print(f"  μ̂  firm / ice        : {res.mu_hat_firm:.3f} / {res.mu_hat_ice:.3f}")
    print(f"  friction radius firm/ice [m] : {res.r_firm:.4f} / {res.r_ice:.4f}")
    print(f"  bound tightened       : {res.tightening_ratio:.1f}x on detected ice")
    print("  artifacts             : outputs/tokens/coupling_demo.(json|md)")
    print("PASS: coupling_demo" if ok else "FAIL: coupling_demo")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
