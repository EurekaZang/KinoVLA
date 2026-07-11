from kino_vla.eval.a8_metrics import cba, delta_fusion
from scripts.a8_eval import _norm_pred, summarize_a8a, summarize_a8b


def test_norm_pred_parses_failure_success():
    assert _norm_pred("Failure: the grasp slipped") == "failure"
    assert _norm_pred("Success") == "success"
    assert _norm_pred('<Action>{"attribution": "failure"}</Action>') == "failure"


def test_summarize_a8a_and_a8b():
    rows = {
        "bdv2__execution": [
            {"truth": "failure", "pred": "failure", "correct": True},
            {"truth": "success", "pred": "failure", "correct": False},
        ]
    }
    s = summarize_a8a(rows)
    assert s["splits"]["bdv2__execution"]["accuracy"]["n"] == 2
    assert "reference" in s

    arms = {
        "v_only": [
            {"sample_id": "1", "stratum": "E2", "correct": True},
            {"sample_id": "2", "stratum": "E3", "correct": False},
        ],
        "p_only": [
            {"sample_id": "1", "stratum": "E2", "correct": False},
            {"sample_id": "2", "stratum": "E3", "correct": True},
        ],
        "latent": [
            {"sample_id": "1", "stratum": "E2", "correct": True},
            {"sample_id": "2", "stratum": "E3", "correct": True},
        ],
        "latent_conflict": [
            {"sample_id": "1", "stratum": "E2", "correct": True},
            {"sample_id": "2", "stratum": "E3", "correct": True},
        ],
    }
    b = summarize_a8b(arms)
    assert b["arms"]["latent"]["cba"] == cba(1.0, 1.0)
    assert b["delta_fusion_latent"] == delta_fusion(1.0, 0.5, 0.5)
