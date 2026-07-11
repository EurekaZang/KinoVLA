from kino_vla.eval.a8_metrics import (
    cba,
    delta_conflict,
    delta_fusion,
    macro_f1,
    method_conflict_summary,
    paired_mcnemar,
    wilson,
)


def test_cba_and_gains():
    assert cba(0.8, 0.6) == 0.7
    assert delta_fusion(0.8, 0.5, 0.6) == 0.2
    assert delta_conflict(0.85, 0.7) == 0.15


def test_macro_f1_balanced():
    y = ["a", "b", "a", "b"]
    p = ["a", "b", "a", "a"]
    assert 0.0 < macro_f1(y, p) < 1.0


def test_wilson_bounds():
    lo, hi = wilson(8, 10)
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0


def test_method_conflict_summary_and_mcnemar():
    rows_v = [
        {"sample_id": "1", "stratum": "E2", "correct": True},
        {"sample_id": "2", "stratum": "E3", "correct": False},
    ]
    rows_p = [
        {"sample_id": "1", "stratum": "E2", "correct": False},
        {"sample_id": "2", "stratum": "E3", "correct": True},
    ]
    rows_l = [
        {"sample_id": "1", "stratum": "E2", "correct": True},
        {"sample_id": "2", "stratum": "E3", "correct": True},
    ]
    rows_c = [
        {"sample_id": "1", "stratum": "E2", "correct": True},
        {"sample_id": "2", "stratum": "E3", "correct": True},
    ]
    summary = method_conflict_summary(
        {
            "v_only": rows_v,
            "p_only": rows_p,
            "latent": rows_l,
            "latent_conflict": rows_c,
        }
    )
    assert summary["arms"]["latent"]["cba"] == 1.0
    assert summary["delta_fusion_latent"] == 0.5
    assert summary["delta_conflict"] == 0.0
    m = paired_mcnemar(rows_v, rows_l)
    assert m["n_shared"] == 2
