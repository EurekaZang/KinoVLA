from __future__ import annotations

from copy import deepcopy

from kino_vla.eval.realistic_c4 import ESTIMAND, evaluate_direct_c4


def _protocol() -> dict:
    return {
        "estimand": ESTIMAND,
        "allowed_decision_inputs": [
            "model_prediction",
            "model_probabilities",
            "rgb",
            "proprioception",
        ],
        "forbidden_decision_inputs": [
            "truth_attribution",
            "target_operator",
            "scene_cluster",
            "test_cost",
        ],
        "bootstrap": {"draws": 500, "seed": 7},
        "acceptance": {
            "minimum_paired_cases": 3,
            "minimum_scene_clusters": 3,
            "minimum_domains": 3,
            "minimum_release_coverage": 0.1,
            "minimum_released_attribution_precision": 0.95,
        },
    }


def _rows() -> list[dict]:
    rows = []
    for index, (scene, domain) in enumerate(
        (("life_scene", "life"), ("factory_scene", "production"), ("trail_scene", "wild"))
    ):
        decision = {
            "input_fields": ["model_prediction", "model_probabilities", "rgb", "proprioception"],
            "release": True,
            "predicted_attribution": "compliant_terrain",
            "recovery_action": "slow_high_step",
            "safe_action": "backstep_detour",
        }
        common = {
            "case_id": f"case_{index}",
            "scene_cluster": scene,
            "domain": domain,
            "split": "test",
            "operator": "O2_compliance",
            "decision_step": 20,
            "predecision_rows_sha256": f"prefix_{index}",
            "branch_started_strictly_after_decision": True,
            "decision": decision,
            "truth_attribution": "compliant_terrain",
            "success": True,
            "fell": False,
        }
        rows.append(
            common
            | {
                "branch": "selective",
                "executed_action": "slow_high_step",
                "terminal_cost": 1.0,
            }
        )
        rows.append(
            deepcopy(common)
            | {
                "branch": "always_safe",
                "executed_action": "backstep_detour",
                "terminal_cost": 2.0,
            }
        )
    return rows


def test_direct_c4_passes_only_on_direct_matched_policy_comparison() -> None:
    report = evaluate_direct_c4(_rows(), _protocol())
    assert report["passed"] is True
    assert report["estimand"] == ESTIMAND
    assert report["primary_endpoint"]["ci95"][1] < 0.0
    assert report["released_attribution_precision"] == 1.0


def test_direct_c4_rejects_truth_as_deployment_input() -> None:
    rows = _rows()
    for row in rows:
        row["decision"]["input_fields"].append("truth_attribution")
    report = evaluate_direct_c4(rows, _protocol())
    assert report["passed"] is False
    assert (
        report["acceptance"]["all_policy_executions_match_frozen_gate"]
        is False
    )


def test_direct_c4_rejects_unmatched_prefix() -> None:
    rows = _rows()
    rows[0]["predecision_rows_sha256"] = "different"
    report = evaluate_direct_c4(rows, _protocol())
    assert report["passed"] is False
    assert report["acceptance"]["all_prefixes_exactly_matched"] is False


def test_direct_c4_requires_frozen_numerical_prefix_certificate_when_configured() -> None:
    protocol = _protocol()
    protocol["paired_prefix_audit"] = {"required": True, "tolerance": 0.002}
    report = evaluate_direct_c4(_rows(), protocol)
    assert report["passed"] is False
    assert report["acceptance"]["all_prefixes_exactly_matched"] is False


def test_direct_c4_accepts_matching_passed_numerical_prefix_certificate() -> None:
    protocol = _protocol()
    protocol["paired_prefix_audit"] = {"required": True, "tolerance": 0.002}
    rows = _rows()
    audit = {
        "passed": True,
        "tolerance": 0.002,
        "maximum_absolute_difference": 0.0015,
    }
    for row in rows:
        certificate = row["predecision_rows_sha256"]
        row["predecision_pair_certificate_sha256"] = certificate
        row["predecision_tolerance_audit"] = audit
    report = evaluate_direct_c4(rows, protocol)
    assert report["passed"] is True


def test_a4_estimand_is_not_accepted_as_c4() -> None:
    protocol = _protocol()
    protocol["estimand"] = "recovery_action_minus_continue_action"
    try:
        evaluate_direct_c4(_rows(), protocol)
    except ValueError as error:
        assert "protocol estimand" in str(error)
    else:
        raise AssertionError("A4 estimand was incorrectly accepted as C4")
