"""M2 Kino-Monitor ROC + step-time budget gates (spec §12 metric, QA 5.3).

The monitor must separate failure rollouts from nominal (high AUC) and its per-step
cost must stay within the 1 ms budget (it is designed to run at the proprioception
rate, the spec §6.9 "1 kHz Monitor").
"""

from __future__ import annotations

import time

import numpy as np

from kino_vla.monitor.roc import compute_monitor_roc, roc_curve
from kino_vla.monitor.rule_monitor import RuleMonitor
from kino_vla.sim.types import Obs
from kino_vla.utils.config import load_config


def test_monitor_roc_separates_failures():
    roc = compute_monitor_roc(n_per_class=20)
    assert roc.auc >= 0.9
    # Clean margin: the worst failure scores above the best nominal alarm.
    failure = roc.scores[roc.labels == 1]
    nominal = roc.scores[roc.labels == 0]
    assert failure.min() > nominal.max()


def test_roc_curve_endpoints():
    # Perfectly separable scores -> AUC 1, curve spans the unit square corners.
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    labels = np.array([0, 0, 1, 1])
    roc = roc_curve(scores, labels)
    assert roc.auc == 1.0
    assert roc.tpr.max() == 1.0
    assert roc.fpr.max() == 1.0
    assert roc.tpr.min() == 0.0


def test_monitor_step_within_budget():
    # QA 5.3: Kino-Monitor step < 1 ms (amortized over a representative run).
    cfg = load_config("monitor/rule_v0.yaml")
    monitor = RuleMonitor(cfg, dt=0.001)
    obs = Obs(
        t=1.0,
        pos=np.array([1.0, 0.0]),
        heading=0.0,
        vel_body=np.array([0.5, 0.0]),
        yaw_rate=0.0,
        cmd_prev=np.array([0.6, 0.0, 0.0]),
        slip_ratio=0.3,
        base_height=0.31,
        tilt=0.0,
        fallen=False,
        effort_ratio=0.1,
    )
    n = 5000
    start = time.perf_counter()
    for _ in range(n):
        monitor.step(obs)
    per_step_ms = (time.perf_counter() - start) / n * 1e3
    assert per_step_ms < 1.0, f"monitor step {per_step_ms:.3f} ms exceeds 1 ms budget"
