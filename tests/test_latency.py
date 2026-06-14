"""Latency-budget instrumentation gates (spec §6.9, M3 exit criterion).

The §6.9 table must be auto-generated from recorded samples (no hand-edited numbers,
QA 5.4) and correctly flag whether each implemented stage meets its budget.
"""

from __future__ import annotations

from kino_vla.shield.latency import BUDGETS_MS, LatencyBudget


def test_budget_constants_match_spec():
    # spec §6.9 budgets (ms).
    assert BUDGETS_MS["kino_monitor"] == 5.0
    assert BUDGETS_MS["reflex"] == 20.0
    assert BUDGETS_MS["cbf_qp"] == 1.0
    assert BUDGETS_MS["cbf_compile"] == 10.0


def test_within_budget_pass_and_fail():
    lb = LatencyBudget()
    lb.extend("cbf_qp", [0.0001, 0.0002, 0.00015])  # 0.1–0.2 ms, under 1 ms
    assert lb.within_budget("cbf_qp") is True
    lb.record_ms("reflex", 50.0)  # over the 20 ms budget
    assert lb.within_budget("reflex") is False
    assert lb.within_budget("vla_inference") is None  # no samples yet


def test_stats_percentiles():
    lb = LatencyBudget()
    lb.samples_ms["cbf_qp"] = [float(i) for i in range(1, 101)]  # 1..100 ms
    st = lb.stats("cbf_qp")
    assert st["n"] == 100
    assert 49.0 <= st["p50"] <= 51.0
    assert st["p99"] >= 98.0


def test_table_renders_all_stages():
    lb = LatencyBudget()
    lb.extend("cbf_qp", [0.0001] * 10)
    table = lb.table()
    assert "| Stage |" in table
    assert "CBF-QP solve" in table
    assert "VLA inference (M7)" in table  # pending stage shown
    assert "PASS" in table
    assert "pending" in table  # stages without samples flagged
