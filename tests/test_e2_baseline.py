"""Fast CPU tests for B1 (E2) — the proprioception-only baseline policy + A-class scenarios.

No torch model and no Isaac: B1's attribution head is driven by a FAKE monitor (a plain object with
``.predict`` and ``.std.mean``), so we test the wiring — proprio to operator to category to the
canonical recovery, the 11->12 feature bridge, rgb/depth being ignored, and schema-valid output —
independent of the real model (exercised only on the GPU sim gate). The FailureTaxonomy is loaded
from config (CPU). This is scaffolding, NOT a result (no surrogate attribution number is claimed).
"""

from __future__ import annotations

import numpy as np
import pytest

from kino_vla.data.schema import Snapshot
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.eval.proprio_baseline import CLASS_NAMES, ProprioBaselinePolicy
from kino_vla.utils.config import load_config


class _FakeMonitor:
    """Minimal LearnedMonitorModel stand-in: returns a fixed attribution distribution."""

    def __init__(self, favored_op_id: str, n_features: int = 12) -> None:
        self._fav = favored_op_id
        self.std = type("STD", (), {"mean": np.zeros(n_features)})()

    def predict(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.asarray(windows)
        assert w.shape[-1] == self.std.mean.shape[0], "bridge must hand the model its feature count"
        attr = np.full((1, len(CLASS_NAMES)), 0.01)
        attr[0, CLASS_NAMES.index(self._fav)] = 0.99
        return np.array([0.9]), attr


def _snapshot(op_name: str, appearance: str, *, n_feat: int = 11, rgb_val: float = 0.0) -> Snapshot:
    return Snapshot(
        operator_name=op_name,
        appearance_class=appearance,
        t=1.0,
        pose_xy=np.zeros(2),
        heading=0.0,
        rgb=np.full((1, 4, 4, 3), rgb_val),
        depth=np.zeros((1, 4, 4)),
        proprio_window=np.random.default_rng(0).normal(size=(25, n_feat)),
        prior_outputs=[],
        privileged_theta={},
        monitor_channel="slip",
    )


@pytest.fixture
def b1_factory():
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    canonical = pcfg.recovery.canonical.to_dict()
    opcat = pcfg.attribution.operator_category.to_dict()

    def make(favored_op_id: str, n_features: int = 12) -> ProprioBaselinePolicy:
        return ProprioBaselinePolicy(
            _FakeMonitor(favored_op_id, n_features),
            tax,
            canonical=canonical,
            operator_category=opcat,
            expected_features=n_features,
        )

    return make


def test_b1_attributes_o4_to_adhesion_backstep(b1_factory):
    b1 = b1_factory("O4")
    dec = b1.decide(_snapshot("O4_tether", "yellow_adhesive"))
    assert dec.ok and dec.annotation is not None
    assert dec.attribution == "adhesion"
    assert dec.primitive_name == "Backstep"


def test_b1_attributes_o2_to_compliant_switchgait(b1_factory):
    b1 = b1_factory("O2")
    dec = b1.decide(_snapshot("O2_compliance", "brown_mud"))
    assert dec.ok
    assert dec.attribution == "compliant_terrain"
    assert dec.primitive_name == "Switch_Gait"


def test_b1_11_to_12_feature_bridge_runs(b1_factory):
    # snapshot carries the 11-dim M4 window; the monitor wants 12 → bridge appends support mean.
    b1 = b1_factory("O4", n_features=12)
    dec = b1.decide(_snapshot("O4_tether", "yellow_adhesive", n_feat=11))
    assert dec.ok and dec.attribution == "adhesion"


def test_b1_ignores_rgb(b1_factory):
    # B1 is proprioception-only: different RGB must not change the decision.
    b1 = b1_factory("O4")
    d0 = b1.decide(_snapshot("O4_tether", "yellow_adhesive", rgb_val=0.0))
    d1 = b1.decide(_snapshot("O4_tether", "yellow_adhesive", rgb_val=1.0))
    assert d0.attribution == d1.attribution == "adhesion"


def test_b1_rejects_empty_window(b1_factory):
    b1 = b1_factory("O4")
    snap = _snapshot("O4_tether", "yellow_adhesive")
    bad = Snapshot(**{**snap.__dict__, "proprio_window": np.zeros((0, 11))})
    dec = b1.decide(bad)
    assert not dec.ok and "no proprio" in dec.reject_code


def test_b1_satisfies_vla_policy_decide_signature(b1_factory):
    # evaluate_attribution / run_vla_rollout call decide(snapshot) with one positional arg.
    b1 = b1_factory("O2")
    dec = b1.decide(_snapshot("O2_compliance", "brown_mud"))
    assert hasattr(dec, "attribution") and hasattr(dec, "primitive_name")
