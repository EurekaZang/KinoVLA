from scripts.a7_build_datasets import select_conflict_records


def _rec(sid, op, app, split="train", annotated=True):
    return {
        "sample_id": sid,
        "snapshot": {"operator_name": op, "appearance_class": app},
        "appearance_id": app,
        "appearance_split": split,
        "annotation": {"thought": "x"} if annotated else None,
    }


def test_conflict_dose_selects_train_appearances_only():
    records = [
        _rec("o4a", "O4_tether", "yellow_board"),
        _rec("o4b", "O4_tether", "gray_tape", split="test"),
        _rec("o4c", "O4_tether", "amber_adhesive"),
        _rec("o2a", "O2_compliance", "brown_mud"),
        _rec("o2b", "O2_compliance", "wet_sheen_mud", split="test"),
        _rec("o2c", "O2_compliance", "dark_gray_mud", annotated=False),
    ]
    selected, ids = select_conflict_records(
        records,
        train_appearances={
            "O4_tether": ["yellow_board", "amber_adhesive"],
            "O2_compliance": ["brown_mud", "dark_gray_mud"],
        },
        dose=5,
        seed=0,
    )
    sids = {r["sample_id"] for r in selected}
    assert sids == {"o4a", "o4c", "o2a"}
    assert "o4b" not in sids and "o2b" not in sids and "o2c" not in sids
    assert set(ids) == {"O4_tether", "O2_compliance"}


def test_conflict_dose_is_deterministic_and_bounded():
    records = [_rec(f"o4{i}", "O4_tether", "yellow_board") for i in range(10)]
    records += [_rec(f"o2{i}", "O2_compliance", "brown_mud") for i in range(10)]
    kwargs = dict(
        train_appearances={"O4_tether": ["yellow_board"], "O2_compliance": ["brown_mud"]},
        dose=3,
        seed=7,
    )
    a, ids_a = select_conflict_records(records, **kwargs)
    b, ids_b = select_conflict_records(records, **kwargs)
    assert [r["sample_id"] for r in a] == [r["sample_id"] for r in b]
    assert ids_a == ids_b
    assert sum(r["snapshot"]["operator_name"] == "O4_tether" for r in a) == 3
    assert sum(r["snapshot"]["operator_name"] == "O2_compliance" for r in a) == 3
    assert all(r["a7_conflict_dose"] == 3 for r in a)
