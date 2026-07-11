from kino_vla.eval.a8_strata import label_stratum


def test_nominal_success_is_e4():
    assert label_stratum({"reward": 1, "failure_mode": "ground_truth"}, {}) == "E4"


def test_wrong_object_is_e2_when_state_nominal():
    meta = {"reward": 0, "failure_mode": "wrong_object"}
    state = {"gripper_delta": 0.0, "joint_motion_norm": 0.1}
    assert label_stratum(meta, state) == "E2"


def test_no_close_is_e3():
    meta = {"reward": 0, "failure_mode": "no_close"}
    state = {"gripper_cmd_closed": 1.0, "gripper_width": 0.08}
    assert label_stratum(meta, state) == "E3"


def test_agree_fallback_e1():
    assert label_stratum({"reward": 0, "failure_mode": "unknown_mode"}, {}) == "E1"
