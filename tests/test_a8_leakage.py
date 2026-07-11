from kino_vla.eval.a8_leakage import EVAL_ONLY_FIELDS, assert_no_leakage, filter_allowed_fields
import pytest


def test_strips_failure_labels_from_model_inputs():
    sample = {
        "images": ["a.png"],
        "task_instruction": "pick cup",
        "robot_state": [0.1, 0.2],
        "failure_mode": "slip",
        "failure_reason": "object dropped",
        "reward": 0,
        "start_caption": "leak",
    }
    out = filter_allowed_fields(sample, mode="model_input")
    assert "images" in out and "task_instruction" in out and "robot_state" in out
    for k in EVAL_ONLY_FIELDS:
        assert k not in out
    assert_no_leakage(out)


def test_assert_no_leakage_raises():
    with pytest.raises(ValueError, match="leaked"):
        assert_no_leakage({"images": [], "failure_mode": "x"})
