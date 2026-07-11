import math

import pytest

from kino_vla.eval.a7_ablation import (
    aggregate_dose_curves,
    aggregate_encoder_grid,
    auroc,
    bootstrap_mean_ci,
    conformal_risk_control,
    dose_curve_summary,
    encoder_cell_key,
    encoder_grid_keys,
    ers_regret_from_agent_rows,
    expected_calibration_error,
    grounding_summary,
    heldout_threshold_selection,
    mcnemar,
    posterior_mean_variance,
    primitive_counts,
    proportion_cell,
    rationale_grounding,
    risk_coverage_auc,
    risk_coverage_curve,
    status_counts,
    tail_window_rows,
    wilson,
)
from scripts.a7_eval import _ensemble_variance, _posterior_source_paths, _score_from_posteriors


def test_wilson_and_proportion_cell():
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(10, 10)[0] > 0.7
    rows = [{"attr_ok": True}, {"attr_ok": False}, {"attr_ok": True}]
    cell = proportion_cell(rows)
    assert cell["n"] == 3
    assert cell["k"] == 2
    assert cell["rate"] == 0.667


def test_mcnemar_exact_pairing():
    lo = [True, False, False, False]
    hi = [True, True, True, False]
    res = mcnemar(lo, hi)
    assert res["b_a_right_b_wrong"] == 0
    assert res["c_a_wrong_b_right"] == 2
    assert res["b_better"] is True
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


def test_auroc_ties_and_degenerate_classes():
    assert math.isnan(auroc([0.1, 0.2], [1, 1]))
    assert auroc([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 1.0
    assert auroc([0.2, 0.2, 0.2, 0.2], [0, 1, 0, 1]) == 0.5


def test_risk_coverage_curve_abstains_high_scores_to_safe_cost():
    rows = [
        {"score": 0.1, "cost_agent": 0.0, "cost_safe": 1.0},
        {"score": 0.2, "cost_agent": 0.0, "cost_safe": 1.0},
        {"score": 0.8, "cost_agent": 4.0, "cost_safe": 1.0},
        {"score": 0.9, "cost_agent": 4.0, "cost_safe": 1.0},
    ]
    curve = risk_coverage_curve(rows, score_field="score")
    assert curve[0]["coverage"] == 0.0
    assert curve[0]["expected_cost"] == 1.0
    assert curve[-1]["coverage"] == 1.0
    assert curve[-1]["expected_cost"] == 2.0
    best = min(curve, key=lambda r: r["expected_cost"])
    assert best["coverage"] == 0.5
    assert best["expected_cost"] == 0.5
    assert risk_coverage_auc(curve) == 1.0


def test_risk_coverage_curve_preserves_exact_thresholds_for_traceability():
    rows = [
        {"score": 0.0221901, "cost_agent": 0.0, "cost_safe": 1.0},
        {"score": 0.0221902, "cost_agent": 4.0, "cost_safe": 1.0},
    ]

    curve = risk_coverage_curve(rows, score_field="score")

    assert curve[0]["tau"] < 0.0221901
    assert curve[1]["tau"] == 0.0221901
    assert curve[2]["tau"] == 0.0221902
    assert len({point["tau"] for point in curve}) == len(curve)


def test_heldout_threshold_selection_selects_on_calibration_only():
    rows = [
        {"sample_id": "a", "score": 0.1, "cost_agent": 0.0, "cost_safe": 1.0, "cost_continue": 4.0},
        {"sample_id": "b", "score": 0.2, "cost_agent": 0.0, "cost_safe": 1.0, "cost_continue": 4.0},
        {"sample_id": "c", "score": 0.8, "cost_agent": 4.0, "cost_safe": 1.0, "cost_continue": 4.0},
        {"sample_id": "d", "score": 0.9, "cost_agent": 4.0, "cost_safe": 1.0, "cost_continue": 4.0},
        {"sample_id": "e", "score": 0.15, "cost_agent": 0.0, "cost_safe": 1.0, "cost_continue": 4.0},
        {"sample_id": "f", "score": 0.85, "cost_agent": 4.0, "cost_safe": 1.0, "cost_continue": 4.0},
    ]
    out = heldout_threshold_selection(rows, score_field="score", split_seed=0, calibration_frac=0.5)
    assert set(out) >= {"tau", "calibration", "test", "splits"}
    assert 0.0 <= out["test"]["coverage"] <= 1.0
    assert out["test"]["expected_cost_ci"][0] <= out["test"]["expected_cost"] <= out["test"]["expected_cost_ci"][1]
    assert set(out["splits"]["calibration_ids"]).isdisjoint(out["splits"]["test_ids"])


def test_conformal_risk_control_returns_threshold_below_risk_budget():
    rows = [
        {"score": 0.1, "cost_agent": 0.0, "cost_safe": 1.0},
        {"score": 0.2, "cost_agent": 0.0, "cost_safe": 1.0},
        {"score": 0.8, "cost_agent": 4.0, "cost_safe": 1.0},
        {"score": 0.9, "cost_agent": 4.0, "cost_safe": 1.0},
    ]
    out = conformal_risk_control(rows, score_field="score", risk_budget=1.0)
    assert out["risk_budget"] == 1.0
    assert out["selected"]["expected_cost"] <= 1.0
    assert out["selected"]["coverage"] == 0.5


def test_expected_calibration_error_bins_confidence_correctness():
    rows = [
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.8, "correct": True},
        {"confidence": 0.2, "correct": False},
        {"confidence": 0.1, "correct": False},
    ]
    out = expected_calibration_error(rows, n_bins=2)
    assert out["n"] == 4
    assert out["ece"] == 0.15
    assert len(out["bins"]) == 2
    assert out["bins"][0]["accuracy"] == 0.0
    assert out["bins"][1]["accuracy"] == 1.0


def test_posterior_mean_variance_scores_seed_disagreement():
    posteriors = [
        {"x": {"a": 0.9, "b": 0.1}},
        {"x": {"a": 0.7, "b": 0.3}},
        {"x": {"a": 0.8, "b": 0.2}},
    ]
    assert posterior_mean_variance(posteriors, "x") == 0.007
    assert posterior_mean_variance(posteriors, "missing") == 0.0


def test_ensemble_variance_attaches_score_when_multiple_posteriors_exist():
    points = [{"sample_id": "x"}, {"sample_id": "y"}]
    posteriors = [
        {"x": {"a": 0.9, "b": 0.1}, "y": {"a": 0.5, "b": 0.5}},
        {"x": {"a": 0.7, "b": 0.3}, "y": {"a": 0.5, "b": 0.5}},
        {"x": {"a": 0.8, "b": 0.2}, "y": {"a": 0.5, "b": 0.5}},
    ]

    assert _ensemble_variance(points, posteriors)
    assert points[0]["ensemble_variance"] == 0.007
    assert points[1]["ensemble_variance"] == 0.0


def test_ensemble_variance_reports_unavailable_for_single_posterior():
    points = [{"sample_id": "x"}]
    posteriors = [{"x": {"a": 0.9, "b": 0.1}}]

    assert not _ensemble_variance(points, posteriors)
    assert "ensemble_variance" not in points[0]


def test_posterior_source_paths_include_all_used_cache_sidecars():
    paths = _posterior_source_paths(
        {
            "main": "posterior_scores.json",
            "seed1": "posterior_scores_seed1.json",
            "seed2": "posterior_scores_seed2.json",
        }
    )

    assert paths == {
        "posterior_scores": "posterior_scores.json",
        "posterior_scores_meta": "posterior_scores_meta.json",
        "posterior_scores_seed1": "posterior_scores_seed1.json",
        "posterior_scores_seed1_meta": "posterior_scores_seed1_meta.json",
        "posterior_scores_seed2": "posterior_scores_seed2.json",
        "posterior_scores_seed2_meta": "posterior_scores_seed2_meta.json",
    }


def test_score_from_posteriors_calibrates_posterior_argmax_not_executed_attr():
    points = [
        {"sample_id": "x", "truth": "adhesion", "attr_ok": False},
        {"sample_id": "y", "truth": "adhesion", "attr_ok": True},
    ]
    posteriors = {
        "x": {"adhesion": 0.9, "nominal": 0.1},
        "y": {"adhesion": 0.4, "nominal": 0.6},
    }

    _score_from_posteriors(points, posteriors)

    assert points[0]["posterior_argmax"] == "adhesion"
    assert points[0]["posterior_correct"] is True
    assert points[0]["posterior_confidence"] == 0.9
    assert points[1]["posterior_argmax"] == "nominal"
    assert points[1]["posterior_correct"] is False
    assert points[1]["posterior_confidence"] == 0.6


def test_bootstrap_mean_ci_contains_mean_for_deterministic_seed():
    ci = bootstrap_mean_ci([0.0, 1.0, 2.0, 3.0], seed=7, n_boot=200)
    assert ci[0] <= 1.5 <= ci[1]


def test_rationale_grounding_category_cues_and_contradictions():
    good = rationale_grounding(
        "The feet slip continuously on a low traction surface.", "low_friction"
    )
    assert good.grounded
    assert "slip" in good.matched_cues
    bad = rationale_grounding("The robot carries a heavy external payload.", "effort_decay")
    assert not bad.grounded
    assert bad.contradiction == "heavy external"
    empty = rationale_grounding("", "adhesion")
    assert not empty.grounded


def test_grounding_summary_counts_grounded_correct_separately():
    rows = [
        {
            "truth": "adhesion",
            "thought": "The sticky tether pulls the body back.",
            "attr_ok": True,
        },
        {"truth": "adhesion", "thought": "This is mud and soft terrain.", "attr_ok": True},
        {"truth": "low_friction", "thought": "Slip is sustained.", "attr_ok": False},
    ]
    s = grounding_summary(rows)
    assert s["n"] == 3
    assert s["grounding_rate"] == 0.667
    assert s["n_correct"] == 2
    assert s["grounded_given_correct"] == 0.5
    assert s["grounded_correct_rate"] == 0.333


def test_grounding_summary_accepts_truth_category_alias_for_test_time_rows():
    """Test-time grounding rows use truth_category (matched-corpus schema)."""
    rows = [
        {
            "truth_category": "adhesion",
            "thought": "Sticky adhesive grip tethers the feet.",
            "attr_ok": True,
        },
        {
            "truth_category": "compliant_terrain",
            "thought": "Soft mud sinks under the feet.",
            "attr_ok": True,
        },
        {
            "truth_category": "adhesion",
            "thought": "This looks like soft mud terrain.",
            "attr_ok": False,
        },
    ]
    s = grounding_summary(rows)
    assert s["n"] == 3
    assert s["grounding_rate"] == 0.667
    assert s["grounded_correct_rate"] == 0.667


def test_report_taxonomy_and_ttg_tables_render_from_machine_json():
    from scripts.a7_report import _taxonomy_table, _ttg_table

    tax = {
        "rows": {
            "latent": {
                "status": "available",
                "heatmap": {
                    "T2": {"acc": 0.6, "ci": [0.5, 0.7], "n": 100},
                    "T3": {"acc": 0.9, "ci": [0.8, 1.0], "n": 100},
                    "T4": {"acc": 0.1, "ci": [0.0, 0.2], "n": 50},
                    "T5": {"acc": 0.3, "ci": [0.2, 0.4], "n": 50},
                },
            },
            "reflect_scalar": {"status": "requires_run"},
            "rich_stats": {
                "status": "available",
                "heatmap": {
                    "T2": {"acc": 0.5, "ci": [0.4, 0.6], "n": 100},
                    "T3": {"acc": 0.0, "ci": [0.0, 0.1], "n": 100},
                    "T4": {"acc": 0.0, "ci": [0.0, 0.1], "n": 50},
                    "T5": {"acc": 0.2, "ci": [0.1, 0.3], "n": 50},
                },
            },
        }
    }
    tax_md = _taxonomy_table(tax)
    assert "latent" in tax_md and "0.100" in tax_md
    assert "reflect_scalar" in tax_md and "requires_run" in tax_md

    ttg = {
        "rows": {
            "text_schema:latent": {
                "status": "available",
                "n": 240,
                "attr_acc": 0.8,
                "grounding": {"grounding_rate": 0.7, "grounded_correct_rate": 0.6},
            }
        }
    }
    ttg_md = _ttg_table(ttg)
    assert "text_schema:latent" in ttg_md
    assert "0.6" in ttg_md


def test_a7_eval_exposes_test_time_grounding_and_taxonomy_stages():
    import scripts.a7_eval as m

    # Stage wiring must accept design-required eval entry points.
    assert hasattr(m, "eval_a2_adapter_with_thoughts")
    assert hasattr(m, "eval_a3_adapter_with_thoughts")
    assert hasattr(m, "_test_time_grounding")
    assert hasattr(m, "_text_schema_taxonomy")
    # argparse choices include the new stages (source-level contract).
    src = open(m.__file__).read()
    assert "test-time-grounding" in src
    assert "text-schema-taxonomy" in src
    # pure stage dispatch without GPU: missing caches -> partial/requires_eval status
    out = m.run("configs/eval/a7.yaml", "test-time-grounding", evaluate=False)
    assert out["manifest"]["stage"] == "test-time-grounding"
    assert "test_time_grounding" in out["results"]


def test_dose_sensitivity_table_marks_push_as_not_headline():
    from scripts.a7_report import _dose_sensitivity_text

    sens = {
        "status": "sensitivity_only_not_headline",
        "scope_note": "Unequal-recipe push is not protocol-matched.",
        "headline_policy": "exclude push cells from dose aggregate",
        "protocol_matched_restored": {
            "seed1": {"o4_attr": 0.0, "o2_attr": 0.5},
            "seed2": {"o4_attr": 0.4, "o2_attr": 0.0},
        },
        "push_vs_old": {
            "1": {"push": {"o4": 0.8, "o2": 1.0}},
            "2": {"push": {"o4": 0.8, "o2": 1.0}},
        },
    }
    md = _dose_sensitivity_text(sens)
    assert "sensitivity only" in md
    assert "0.0" in md and "0.8" in md
    assert "exclude push cells" in md


def test_protocol_matched_dose_aggregate_headlines_dose10_not_dose5():
    """Headline dose aggregate must use protocol-matched cells (dose10 best mean)."""
    import json
    from pathlib import Path

    path = Path("outputs/eval/a7/conflict_dose/summary.json")
    if not path.exists():
        pytest.skip("A7 conflict_dose summary not present")
    cd = json.loads(path.read_text())
    agg = cd["aggregate"]
    assert agg["best_mean_dose"] == 10
    assert agg["by_dose"]["10"]["mean_rate"] == 0.733
    # dose5 remains protocol-unstable, not silently best
    assert agg["by_dose"]["5"]["mean_rate"] < agg["by_dose"]["10"]["mean_rate"]
    # protocol cells restored: seed1 dose5 O4 must not be the unequal-recipe 0.8
    rows = json.loads(Path("outputs/eval/a7/conflict_dose/per_item_seed1_dose05.json").read_text())
    o4 = [r for r in rows if r["operator"] == "O4_tether"]
    o4_rate = sum(r["attr_ok"] for r in o4) / len(o4)
    assert o4_rate == 0.0
    # sensitivity artifact exists and is explicitly non-headline
    sens = Path("outputs/eval/a7/conflict_dose/sensitivity_dose05_upsample/summary.json")
    assert sens.exists()
    s = json.loads(sens.read_text())
    assert s["status"] == "sensitivity_only_not_headline"
    # train meta must match protocol recipe for restored cells
    meta = json.loads(
        Path("outputs/eval/a7/adapters/conflict_dose/seed1/dose_05/a7_train_meta.json").read_text()
    )
    assert meta["dataset"].endswith("dose_05")
    assert not meta["dataset"].endswith("upsample8")
    assert meta["metrics"]["epochs"] == 4


def test_dose_curve_summary_orders_and_pairs_shared_ids():
    dose_rows = {
        "10": [
            {"sample_id": "a", "attr_ok": True},
            {"sample_id": "b", "attr_ok": True},
        ],
        "0": [
            {"sample_id": "a", "attr_ok": False},
            {"sample_id": "b", "attr_ok": True},
        ],
        "5": [
            {"sample_id": "a", "attr_ok": True},
            {"sample_id": "b", "attr_ok": True},
        ],
    }
    out = dose_curve_summary(dose_rows)
    assert [r["dose"] for r in out["curve"]] == [0, 5, 10]
    assert out["monotone_non_decreasing"] is True
    assert out["curve"][1]["mcnemar_vs_prev"]["c_a_wrong_b_right"] == 1


def test_primitive_counts_is_stable():
    rows = [{"primitive": "Backstep"}, {"primitive": "Backstep"}, {"primitive": "Switch_Gait"}]
    assert primitive_counts(rows) == {"Backstep": 2, "Switch_Gait": 1}


def test_status_counts_counts_rows_by_status():
    rows = {
        "seed0_dose00": {"status": "available"},
        "seed0_dose05": {"status": "available"},
        "seed1_dose00": {"status": "requires_run"},
        "seed1_dose05": {"status": "blocked"},
    }
    assert status_counts(rows) == {"available": 2, "blocked": 1, "requires_run": 1}


def test_aggregate_dose_curves_reports_mean_best_and_nonmonotone():
    curves = {
        "seed0": {
            "curve": [
                {"dose": 0, "rate": 0.4, "n": 10, "k": 4, "ci": [0.2, 0.6]},
                {"dose": 5, "rate": 0.6, "n": 10, "k": 6, "ci": [0.3, 0.8]},
                {"dose": 10, "rate": 0.5, "n": 10, "k": 5, "ci": [0.2, 0.7]},
            ],
            "monotone_non_decreasing": False,
        },
        "seed1": {
            "curve": [
                {"dose": 0, "rate": 0.5, "n": 10, "k": 5, "ci": [0.2, 0.7]},
                {"dose": 5, "rate": 0.7, "n": 10, "k": 7, "ci": [0.4, 0.9]},
                {"dose": 10, "rate": 0.6, "n": 10, "k": 6, "ci": [0.3, 0.8]},
            ],
            "monotone_non_decreasing": False,
        },
    }
    out = aggregate_dose_curves(curves)
    assert out["n_seeds"] == 2
    assert out["monotone_all_seeds"] is False
    assert out["best_mean_dose"] == 5
    assert out["by_dose"]["5"]["mean_rate"] == 0.65


def test_ers_regret_from_agent_rows_uses_a4_cost_matrix():
    m_cost = {
        "matched_O4_twophase": {"backstep_detour": 1.0, "high_step": 4.0},
        "matched_O2": {"backstep_detour": 1.0, "high_step": 0.0},
    }
    canonical = {"matched_O4_twophase": "backstep_detour", "matched_O2": "high_step"}
    rows = {
        "good": [
            {"scenario": "matched_O4_twophase", "primitive": "backstep_detour"},
            {"scenario": "matched_O2", "primitive": "high_step"},
        ],
        "bad": [
            {"scenario": "matched_O4_twophase", "primitive": "high_step"},
            {"scenario": "matched_O2", "primitive": "backstep_detour"},
        ],
    }
    out = ers_regret_from_agent_rows(rows, m_cost, canonical)
    assert out["good"]["mean_cost"] == 0.5
    assert out["good"]["mean_regret"] == 0.0
    assert out["bad"]["mean_cost"] == 2.5
    assert out["bad"]["mean_regret"] == 2.0


def test_encoder_grid_keys_crosses_declared_variants_windows_and_gates():
    cfg = {
        "variants": {
            "privileged_distillation": {},
            "contrastive_only": {},
            "from_scratch": {},
        },
        "window_lengths": [25, 50, 100],
        "anomaly_gate": [True, False],
    }
    keys = encoder_grid_keys(cfg)
    assert len(keys) == 18
    assert keys[0] == "privileged_distillation__T25__gate_on"
    assert keys[-1] == "from_scratch__T100__gate_off"
    assert encoder_cell_key("contrastive_only", 50, False) in keys


def test_tail_window_rows_supports_downsample_and_fail_closed_for_upsample():
    rows = [[float(i), float(i + 100)] for i in range(25)]
    assert tail_window_rows(rows, 25) == rows
    assert tail_window_rows(rows, 10)[0] == [15.0, 115.0]
    with pytest.raises(ValueError, match="cannot build T=50"):
        tail_window_rows(rows, 50)


def test_aggregate_encoder_grid_reports_best_and_no_placeholders():
    cells = {
        "privileged_distillation__T25__gate_on": {
            "status": "available",
            "variant": "privileged_distillation",
            "window_length": 25,
            "gate": True,
            "theta_mae_mean": 0.3,
            "attr_acc": {"rate": 0.9, "n": 10, "k": 9, "ci": [0.6, 1.0]},
            "residual_error_auroc": 0.8,
        },
        "contrastive_only__T25__gate_on": {
            "status": "available",
            "variant": "contrastive_only",
            "window_length": 25,
            "gate": True,
            "theta_mae_mean": 0.8,
            "attr_acc": {"rate": 0.7, "n": 10, "k": 7, "ci": [0.4, 0.9]},
            "residual_error_auroc": 0.5,
        },
        "from_scratch__T50__gate_off": {"status": "requires_real_artifact"},
    }
    out = aggregate_encoder_grid(cells)
    assert out["status_counts"] == {"available": 2, "requires_real_artifact": 1}
    assert out["best_theta_mae"]["key"] == "privileged_distillation__T25__gate_on"
    assert out["best_attr_acc"]["key"] == "privileged_distillation__T25__gate_on"
    assert out["available_cells"] == 2
    assert out["blocked_cells"] == 1
