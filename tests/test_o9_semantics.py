from kino_vla.data.o9_semantics import evaluate_o9_high_centering


def _telemetry(*, base: float, head: float, consecutive: int, duty: float, support: float):
    return {
        "enabled": True,
        "regions": [
            {
                "region": {"cx": 1.0, "cy": 0.0, "hx": 0.18, "hy": 0.05},
                "max_belly_contact_force_n": base,
                "max_consecutive_belly_contact_steps": consecutive,
                "belly_contact_duty_cycle": duty,
                "min_measured_support": support,
                "max_nonfoot_contact_by_body_n": {
                    "base": base,
                    "Head_upper": head,
                    "FL_calf": 0.0,
                },
            }
        ],
    }


def test_o9_semantic_gate_accepts_sustained_belly_beaching():
    result = evaluate_o9_high_centering(
        _telemetry(base=180.0, head=0.0, consecutive=14, duty=0.50, support=0.50),
        robot_position_xy_m=(1.02, 0.0),
    )
    assert result["passed"] is True


def test_o9_semantic_gate_rejects_head_only_collision():
    result = evaluate_o9_high_centering(
        _telemetry(base=0.0, head=700.0, consecutive=0, duty=0.0, support=0.50),
        robot_position_xy_m=(0.60, 0.0),
    )
    assert result["passed"] is False
    checks = result["regions"][0]["checks"]
    assert checks["base_contact_force_sufficient"] is False
    assert checks["not_head_or_limb_only_collision"] is False
    assert checks["ridge_under_chassis_center"] is False
