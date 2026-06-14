"""End-to-end latency budget instrumentation (spec §6.9).

The two-closed-loop architecture only holds if the fast safety loop keeps the state
in the safe set 𝒞 long enough for the slow VLA loop to think (spec §6.9: Reflex pins
the state, CBF guarantees the VLA's command cannot push it out). This module records
per-stage wall-clock samples and renders the §6.9 budget table comparing measured
p50/p99 against each stage's budget. Stages not yet implemented (Kino-Token window,
VLA inference) carry their spec budgets with no samples until M4/M7 fill them in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Spec §6.9 latency budget, in milliseconds. ``None`` budget = informational stage.
BUDGETS_MS: dict[str, float] = {
    "kino_monitor": 5.0,  # anomaly detection
    "reflex": 20.0,  # self-stabilising intervention
    "kino_token_window": 500.0,  # sliding-window fill (M4)
    "vla_inference": 1000.0,  # ~1 s VLA reasoning (M7)
    "cbf_qp": 1.0,  # QP solve (the M3 hard gate)
    "cbf_compile": 10.0,  # CBF adjudication + primitive compile
}

# Display order and labels for the rendered table.
_ORDER = [
    ("kino_monitor", "Kino-Monitor detect"),
    ("reflex", "Reflex self-stab"),
    ("kino_token_window", "Kino-Token window (M4)"),
    ("vla_inference", "VLA inference (M7)"),
    ("cbf_qp", "CBF-QP solve"),
    ("cbf_compile", "CBF adjudicate + compile"),
]


@dataclass
class LatencyBudget:
    """Collects per-stage timing samples and renders the spec §6.9 budget table."""

    samples_ms: dict[str, list[float]] = field(default_factory=dict)

    def record(self, stage: str, seconds: float) -> None:
        self.samples_ms.setdefault(stage, []).append(float(seconds) * 1e3)

    def record_ms(self, stage: str, milliseconds: float) -> None:
        self.samples_ms.setdefault(stage, []).append(float(milliseconds))

    def extend(self, stage: str, seconds_list: list[float]) -> None:
        self.samples_ms.setdefault(stage, []).extend(float(s) * 1e3 for s in seconds_list)

    def stats(self, stage: str) -> dict[str, float]:
        s = np.asarray(self.samples_ms.get(stage, []), dtype=np.float64)
        if s.size == 0:
            return {"n": 0, "p50": float("nan"), "p99": float("nan"), "mean": float("nan")}
        return {
            "n": int(s.size),
            "p50": float(np.percentile(s, 50)),
            "p99": float(np.percentile(s, 99)),
            "mean": float(s.mean()),
        }

    def within_budget(self, stage: str) -> bool | None:
        """True/False if the stage's p99 meets its budget; None if no samples."""
        st = self.stats(stage)
        if st["n"] == 0 or stage not in BUDGETS_MS:
            return None
        return st["p99"] <= BUDGETS_MS[stage]

    def table(self) -> str:
        """Render the §6.9 latency budget as a markdown table."""
        rows = [
            "| Stage | Budget (ms) | p50 (ms) | p99 (ms) | n | status |",
            "|---|---|---|---|---|---|",
        ]
        for key, label in _ORDER:
            budget = BUDGETS_MS.get(key)
            st = self.stats(key)
            if st["n"] == 0:
                status = "— (pending)"
                p50 = p99 = "—"
            else:
                p50 = f"{st['p50']:.3f}"
                p99 = f"{st['p99']:.3f}"
                ok = self.within_budget(key)
                status = "PASS" if ok else "FAIL"
            budget_s = f"{budget:.1f}" if budget is not None else "—"
            rows.append(f"| {label} | {budget_s} | {p50} | {p99} | {st['n']} | {status} |")
        return "\n".join(rows)
