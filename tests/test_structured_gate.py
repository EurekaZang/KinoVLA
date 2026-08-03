from kino_vla.eval.structured_gate import (
    evaluate_gate,
    fit_structured_gate,
    fit_supported_structured_gate,
    gate_acts,
)

CANONICAL = {
    "adhesion": "Backstep",
    "compliant_terrain": "Switch_Gait",
}


def _row(
    sid: str,
    *,
    label: str,
    category: str,
    delta: float,
    cluster: str,
    split: str = "train",
) -> dict:
    return {
        "sample_id": sid,
        "scenario": "matched_O2",
        "appearance_id": cluster,
        "appearance_split": split,
        "cluster_id": f"matched_O2|{cluster}",
        "agent_label": label,
        "posterior_argmax": category,
        "safe_label": "backstep_detour",
        "continue_label": "continue",
        "cost_agent": 1.0 + delta,
        "cost_safe": 1.0,
        "cost_continue": 2.0,
    }


def test_gate_accepts_canonical_no_harm_stratum():
    rows = [
        _row("a", label="high_step", category="compliant_terrain", delta=-1.0, cluster="mud1"),
        _row("b", label="high_step", category="compliant_terrain", delta=0.0, cluster="mud2"),
    ]
    gate = fit_structured_gate(rows, CANONICAL)
    assert gate["accepted_strata"] == [
        {"agent_label": "high_step", "posterior_argmax": "compliant_terrain"}
    ]
    assert gate_acts(rows[0], gate)


def test_gate_rejects_noncanonical_calibration_shortcut():
    rows = [
        _row("a", label="detour_replan", category="adhesion", delta=-2.0, cluster="tape1"),
        _row("b", label="detour_replan", category="adhesion", delta=-2.0, cluster="tape2"),
    ]
    gate = fit_structured_gate(rows, CANONICAL)
    assert gate["accepted_strata"] == []
    assert not gate["calibration_audit"]["detour_replan|adhesion"]["semantic_consistent"]


def test_gate_rejects_any_calibration_cluster_harm():
    rows = [
        _row("a", label="high_step", category="compliant_terrain", delta=-2.0, cluster="mud1"),
        _row("b", label="high_step", category="compliant_terrain", delta=0.1, cluster="mud2"),
    ]
    gate = fit_structured_gate(rows, CANONICAL)
    assert gate["accepted_strata"] == []


def test_evaluation_uses_only_observable_pair_for_decision():
    calibration = [
        _row("a", label="high_step", category="compliant_terrain", delta=-1.0, cluster="mud1"),
        _row("b", label="high_step", category="compliant_terrain", delta=-1.0, cluster="mud2"),
    ]
    gate = fit_structured_gate(calibration, CANONICAL)
    test = [
        _row(
            "c",
            label="high_step",
            category="compliant_terrain",
            delta=-1.0,
            cluster="mud3",
            split="test",
        ),
        _row(
            "d",
            label="detour_replan",
            category="adhesion",
            delta=3.0,
            cluster="tape3",
            split="test",
        ),
    ]
    result = evaluate_gate(test, gate)
    assert result["coverage"] == 0.5
    assert result["expected_cost"] == 0.5
    assert result["paired_delta"]["safe"] == -0.5


def test_supported_gate_learns_radius_that_rejects_harmful_development_cluster():
    calibration = [
        {
            **_row(
                "a", label="high_step", category="compliant_terrain", delta=-1.0, cluster="mud1"
            ),
            "attribution": "compliant_terrain",
            "material_distance": 0.0,
            "method_split": "train",
            "attr_ok": True,
            "tracking_error_peak": 0.2,
        },
        {
            **_row(
                "b", label="high_step", category="compliant_terrain", delta=-1.0, cluster="mud2"
            ),
            "attribution": "compliant_terrain",
            "material_distance": 0.02,
            "method_split": "train",
            "attr_ok": True,
            "tracking_error_peak": 0.2,
        },
    ]
    development = [
        {
            **_row(
                "c",
                label="high_step",
                category="compliant_terrain",
                delta=-1.0,
                cluster="mud3",
                split="final",
            ),
            "attribution": "compliant_terrain",
            "material_distance": 0.08,
            "method_split": "development",
            "attr_ok": True,
            "tracking_error_peak": 0.2,
        },
        {
            **_row(
                "d",
                label="high_step",
                category="compliant_terrain",
                delta=0.0,
                cluster="red_solid",
                split="final",
            ),
            "attribution": "compliant_terrain",
            "material_distance": 0.15,
            "method_split": "development",
            "attr_ok": False,
            "tracking_error_peak": 0.3,
        },
        {
            **_row(
                "e",
                label="high_step",
                category="compliant_terrain",
                delta=0.0,
                cluster="nominal_mud",
                split="final",
            ),
            "attribution": "compliant_terrain",
            "material_distance": 0.0,
            "method_split": "development",
            "attr_ok": False,
            "tracking_error_peak": 0.1,
        },
    ]
    gate = fit_supported_structured_gate(
        calibration + development,
        CANONICAL,
        min_fit_coverage=0.5,
        allowed_strata=[("high_step", "compliant_terrain")],
        validity_field="attr_ok",
        evidence_field="tracking_error_peak",
    )
    assert gate["support"]["max_value"] == 0.115
    assert gate["evidence"]["min_value"] == 0.15
    assert gate_acts(development[0], gate)
    assert not gate_acts(development[1], gate)
    assert not gate_acts(development[2], gate)


def test_supported_gate_refuses_final_rows_during_fit():
    row = {
        **_row("a", label="high_step", category="compliant_terrain", delta=-1.0, cluster="mud1"),
        "attribution": "compliant_terrain",
        "material_distance": 0.0,
        "method_split": "final",
    }
    try:
        fit_supported_structured_gate([row], CANONICAL)
    except ValueError as exc:
        assert "final/evaluation" in str(exc)
    else:
        raise AssertionError("final rows must not enter gate fitting")
