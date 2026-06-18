"""VLA dataset builder + seeded operator-stratified split (M7 §11 Stage 1 data path)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from kino_vla.utils.config import load_config
from kino_vla.vla.dataset_build import (
    load_examples,
    split_examples,
    suite_sem_examples,
)

# A minimal kept-sample record, matching DataSample.to_record() (kino_vla/data/schema.py).
_PAIR = {
    "O1_mu_field": ("O1_mu_field|O3_collapse", "low_friction", "A", "Set_Constraint"),
    "O3_collapse": ("O1_mu_field|O3_collapse", "region_collapse", "B", "Update_Topology"),
    "O4_tether": ("O2_compliance|O4_tether", "adhesion", "B", "Backstep"),
    "O2_compliance": ("O2_compliance|O4_tether", "compliant_terrain", "A", "Switch_Gait"),
}
_APPEARANCE = {
    "O1_mu_field": "ice_sheet",
    "O3_collapse": "ice_sheet",
    "O4_tether": "yellow_adhesive",
    "O2_compliance": "brown_mud",
}
_PARAMS = {
    "Set_Constraint": {"max_speed": 0.4, "stiffness": 0.5},
    "Update_Topology": {"region_xy": [2.0, 0.0], "radius_m": 0.6, "status": "untraversable"},
    "Backstep": {"distance_m": 0.5},
    "Switch_Gait": {"mode": "high_step"},
}


@pytest.fixture
def tiny_dataset(tmp_path):
    """Write a hermetic 12-sample dataset (4 operators, 2 ambiguity pairs) + frames sidecar."""
    records = []
    frames: dict[str, np.ndarray] = {}
    for op, (pair, cat, ab, prim) in _PAIR.items():
        for i in range(3):
            sid = f"{op}_{i}"
            records.append(
                {
                    "sample_id": sid,
                    "snapshot": {
                        "operator_name": op,
                        "appearance_class": _APPEARANCE[op],
                        "t": 2.0,
                        "pose_xy": [2.5, float(i)],
                        "heading": 0.0,
                        "prior_outputs": [],
                        "privileged_theta": {"mu": 0.1},
                        "monitor_channel": "slip",
                        "rgb_shape": [5, 8, 8, 3],
                        "depth_shape": [5, 8, 8],
                        "proprio_shape": [25, 11],
                    },
                    "ground_truth": {
                        "category": cat,
                        "ab_class": ab,
                        "theta": {"mu": 0.1},
                        "feasible": [prim],
                        "canonical_primitive": prim,
                        "is_sudden_trap": cat in ("adhesion", "region_collapse"),
                    },
                    "annotation": {
                        "thought": f"reasoning for {op}",
                        "attribution": cat,
                        "attribution_raw": cat,
                        "action": {"primitive": prim, "params": _PARAMS[prim]},
                    },
                    "verdict": {"keep": True, "reason": "keep", "detail": ""},
                    "target_theta": [0.1, 0.0, 1.0, 1.0],
                    "ambiguity_pair": pair,
                }
            )
            frames[f"{sid}__rgb"] = np.zeros((5, 8, 8, 3), dtype=np.float32)
            frames[f"{sid}__depth"] = np.ones((5, 8, 8), dtype=np.float32)
            frames[f"{sid}__proprio"] = np.zeros((25, 11), dtype=np.float32)
    (tmp_path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    np.savez_compressed(tmp_path / "frames.npz", **frames)
    (tmp_path / "dataset_card.json").write_text(json.dumps({"stats": {}}))
    return tmp_path


def test_load_examples_latent(tiny_dataset):
    cfg = load_config("data/hindsight.yaml")
    ex = load_examples(tiny_dataset, cfg, route="latent")
    assert len(ex) == 12
    e = ex[0]
    assert e.rgb.shape == (5, 8, 8, 3)
    assert e.proprio_window.shape == (25, 11)
    assert e.target_text.startswith("<Thought>")
    assert len(e.target_theta) == 4
    assert e.ambiguity_pair is not None


def test_split_is_deterministic_and_disjoint(tiny_dataset):
    cfg = load_config("data/hindsight.yaml")
    ex = load_examples(tiny_dataset, cfg)
    s1 = split_examples(ex, seed=7, val_frac=0.34, test_frac=0.34)
    s2 = split_examples(ex, seed=7, val_frac=0.34, test_frac=0.34)
    ids = lambda lst: [e.sample_id for e in lst]  # noqa: E731
    assert ids(s1.train) == ids(s2.train) and ids(s1.test) == ids(s2.test)  # deterministic
    all_ids = set(ids(s1.train)) | set(ids(s1.val)) | set(ids(s1.test))
    assert len(all_ids) == 12  # no leakage: each sample in exactly one split
    assert len(ids(s1.train)) + len(ids(s1.val)) + len(ids(s1.test)) == 12


def test_split_stratified_every_operator_in_test(tiny_dataset):
    cfg = load_config("data/hindsight.yaml")
    ex = load_examples(tiny_dataset, cfg)
    split = split_examples(ex, seed=1, val_frac=0.34, test_frac=0.34)
    test_ops = {e.operator_name for e in split.test}
    assert test_ops == set(_PAIR)  # each operator (>=3 samples) appears in the held-out test set


def test_suite_sem_subset(tiny_dataset):
    cfg = load_config("data/hindsight.yaml")
    ex = load_examples(tiny_dataset, cfg)
    split = split_examples(ex, seed=1, val_frac=0.34, test_frac=0.34)
    sem = suite_sem_examples(split.test)
    assert sem and all(e.ambiguity_pair is not None for e in sem)
