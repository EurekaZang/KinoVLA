"""Data-production path tests (spec §10, M6) — the GPU-free critical path the filter golden
tests don't cover: annotate_snapshots (incl. the M1 oracle-error handling), the snapshot
save/load round-trip + collection meta, random_lanes, and merge_datasets (the #32 O5/O10 merge).

These pin the behaviour that actually produced the real dataset, so a regression trips CI even
though the Isaac collection itself is GPU-gated.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from kino_vla.data import (
    DROP_ATTRIBUTION,
    KEEP,
    ORACLE_ERROR,
    CoTAnnotation,
    DataSample,
    FailureTaxonomy,
    Verdict,
    annotate_snapshots,
    load_snapshot_meta,
    load_snapshots,
    merge_datasets,
    save_snapshots,
    write_dataset,
)
from kino_vla.data.isaac_rollout import random_lanes
from kino_vla.data.pipeline import PipelineResult, compute_stats
from kino_vla.data.schema import Snapshot
from kino_vla.tokens.features import N_FEATURES
from kino_vla.utils.config import load_config

# Valid params per primitive (so constructed annotations pass schema validation).
_PARAMS = {
    "Set_Constraint": {"max_speed": 0.4, "stiffness": 0.5},
    "Update_Topology": {"region_xy": [0.0, 0.0], "radius_m": 0.6, "status": "untraversable"},
    "Hold_and_Request": {"reason": "actuator saturated"},
    "Switch_Gait": {"mode": "crawl"},
    "Backstep": {"distance_m": 0.5},
}


@pytest.fixture(scope="module")
def cfg():
    return load_config("data/hindsight.yaml")


@pytest.fixture(scope="module")
def tax(cfg) -> FailureTaxonomy:
    return FailureTaxonomy(cfg)


def _snap(op: str = "O1_mu_field", appearance: str = "ice_sheet") -> Snapshot:
    return Snapshot(
        operator_name=op,
        appearance_class=appearance,
        t=1.0,
        pose_xy=np.zeros(2),
        heading=0.0,
        rgb=np.zeros((5, 4, 4, 3)),
        depth=np.zeros((5, 4, 4)),
        proprio_window=np.zeros((25, N_FEATURES)),
        prior_outputs=[],
        privileged_theta={"mu": 0.1, "payload_kg": 0.0, "effort_scale": 1.0, "support_ratio": 1.0},
        monitor_channel="slip_ratio",
    )


class _StubOracle:
    """Returns a fixed attribution+primitive; counts calls (to verify the cache short-circuits)."""

    def __init__(self, attribution: str, primitive: str) -> None:
        self.attribution = attribution
        self.primitive = primitive
        self.calls = 0

    def annotate(self, snapshot: Snapshot) -> str:  # noqa: ARG002
        self.calls += 1
        return json.dumps(
            {
                "thought": "t",
                "attribution": self.attribution,
                "action": {"primitive": self.primitive, "params": _PARAMS[self.primitive]},
            }
        )


class _RaisingOracle:
    """Models a gateway failure (HTTP 429 / dropped connection)."""

    def annotate(self, snapshot: Snapshot) -> str:  # noqa: ARG002
        raise RuntimeError("Oracle API failed after 6 attempts: HTTP 429")


# ----------------------------------------------------------------- annotate_snapshots (+ M1)
def test_annotate_keeps_grounded_cot(cfg, tax):
    items = [(_snap("O1_mu_field"), {})]
    oracle = _StubOracle("low_friction", "Set_Constraint")
    res = annotate_snapshots(cfg, items, oracle=oracle, taxonomy=tax)
    assert res.stats.kept == 1
    assert res.stats.dropped == 0
    assert res.stats.oracle_error == 0
    assert res.stats.reject_rate == 0.0
    assert res.samples[0].verdict.reason == KEEP


def test_annotate_drops_confabulation(cfg, tax):
    items = [(_snap("O1_mu_field"), {})]  # truth = low_friction
    oracle = _StubOracle("compliant_terrain", "Switch_Gait")  # wrong attribution
    res = annotate_snapshots(cfg, items, oracle=oracle, taxonomy=tax)
    assert res.stats.kept == 0
    assert res.stats.dropped == 1
    assert res.samples[0].verdict.reason == DROP_ATTRIBUTION
    assert res.stats.reject_rate == 1.0


def test_annotate_oracle_failure_is_not_a_filter_rejection(cfg, tax):
    """M1: an API failure is ORACLE_ERROR and is EXCLUDED from the reject-rate (the headline metric
    must reflect the filter's judgments only)."""
    items = [(_snap("O1_mu_field"), {}), (_snap("O3_collapse", "ice_sheet"), {})]
    res = annotate_snapshots(cfg, items, oracle=_RaisingOracle(), taxonomy=tax)
    assert res.stats.kept == 0
    assert res.stats.dropped == 0  # NOT counted as filter drops
    assert res.stats.oracle_error == 2
    assert res.stats.reject_rate == 0.0  # the key M1 assertion (was 1.0 when conflated)
    assert all(s.verdict.reason == ORACLE_ERROR for s in res.samples)
    assert res.stats.by_reason.get(ORACLE_ERROR) == 2  # still visible in by_reason


def test_annotate_mixed_reject_rate_excludes_errors(cfg, tax):
    """1 keep + 1 confab + 1 API error => reject-rate is 1/2 (over the 2 the filter judged)."""

    class _Mixed:
        def __init__(self):
            self.i = 0

        def annotate(self, snapshot):
            self.i += 1
            if self.i == 1:
                return json.dumps(
                    {
                        "thought": "t",
                        "attribution": "low_friction",
                        "action": {
                            "primitive": "Set_Constraint",
                            "params": _PARAMS["Set_Constraint"],
                        },
                    }
                )
            if self.i == 2:
                return json.dumps(
                    {
                        "thought": "t",
                        "attribution": "compliant_terrain",
                        "action": {"primitive": "Switch_Gait", "params": _PARAMS["Switch_Gait"]},
                    }
                )
            raise RuntimeError("Oracle API failed: RemoteDisconnected")

    items = [(_snap("O1_mu_field"), {})] * 3
    res = annotate_snapshots(cfg, items, oracle=_Mixed(), taxonomy=tax)
    assert (res.stats.kept, res.stats.dropped, res.stats.oracle_error) == (1, 1, 1)
    assert res.stats.reject_rate == 0.5


def test_annotate_cache_short_circuits_oracle(cfg, tax):
    items = [(_snap("O1_mu_field"), {}), (_snap("O1_mu_field"), {})]
    cached_ann = CoTAnnotation.from_oracle_text(
        json.dumps(
            {
                "thought": "cached",
                "attribution": "low_friction",
                "action": {"primitive": "Set_Constraint", "params": _PARAMS["Set_Constraint"]},
            }
        ),
        synonyms=tax.synonyms,
        valid_categories=tax.valid_categories,
    )
    cache = {0: (cached_ann, Verdict(True, KEEP))}
    oracle = _StubOracle("low_friction", "Set_Constraint")
    res = annotate_snapshots(cfg, items, oracle=oracle, taxonomy=tax, cache=cache)
    assert oracle.calls == 1  # index 0 reused from cache, only index 1 called
    assert res.stats.kept == 2


def test_annotate_concurrency_matches_serial(cfg, tax):
    items = [(_snap("O1_mu_field"), {}) for _ in range(8)]
    serial = annotate_snapshots(
        cfg, items, oracle=_StubOracle("low_friction", "Set_Constraint"), taxonomy=tax
    )
    parallel = annotate_snapshots(
        cfg,
        items,
        oracle=_StubOracle("low_friction", "Set_Constraint"),
        taxonomy=tax,
        concurrency=4,
    )
    assert serial.stats.kept == parallel.stats.kept == 8
    assert [s.sample_id for s in serial.samples] == [s.sample_id for s in parallel.samples]


def test_annotate_records_no_interception(cfg, tax):
    items = [(_snap("O1_mu_field"), {})]
    res = annotate_snapshots(
        cfg,
        items,
        oracle=_StubOracle("low_friction", "Set_Constraint"),
        taxonomy=tax,
        n_no_interception=3,
    )
    assert res.stats.n_no_interception == 3
    assert res.stats.n_cells == 1 + 3


# ----------------------------------------------------- snapshot store round-trip + meta (m5)
def test_save_load_snapshots_roundtrip_with_meta(tmp_path):
    items = [
        (_snap("O5_payload", "solid_ground"), {"mass_kg": 16.0}),
        (_snap("O10_effort_decay", "solid_ground"), {"floor": 0.13}),
    ]
    path = tmp_path / "snaps.npz"
    save_snapshots(path, items, n_attempted=5)
    loaded = load_snapshots(path)
    assert len(loaded) == 2
    assert loaded[0][0].operator_name == "O5_payload"
    assert loaded[0][1] == {"mass_kg": 16.0}
    assert np.array_equal(loaded[1][0].proprio_window, items[1][0].proprio_window)
    assert load_snapshot_meta(path) == {"n_attempted": 5, "n_saved": 2}


def test_load_snapshot_meta_absent_is_empty(tmp_path):
    items = [(_snap(), {})]
    path = tmp_path / "snaps.npz"
    save_snapshots(path, items)  # no n_attempted ⇒ no sidecar
    assert load_snapshot_meta(path) == {}


# ----------------------------------------------------- random_lanes (scale collection, pure)
def test_random_lanes_cover_observable_ops_incl_o5(cfg):
    lanes = random_lanes(cfg, seed=0, n_lanes=40)
    assert len(lanes) == 40
    configured = {str(s["op"]) for s in cfg.isaac_collect.random.ops}
    assert {lane["op"] for lane in lanes} <= configured
    assert "O5_payload" in configured  # O5 is back in scale collection (#32)
    # θ sampled within the configured range; lanes spaced (no two share a y)
    for lane in lanes:
        if lane["op"] == "O5_payload":
            assert 14.0 <= lane["mass"] <= 18.0
    assert len({lane["y"] for lane in lanes}) == 40


def test_random_lanes_deterministic(cfg):
    a = random_lanes(cfg, seed=7, n_lanes=12)
    b = random_lanes(cfg, seed=7, n_lanes=12)
    assert a == b


# ----------------------------------------------------- merge_datasets (#32 O5/O10 merge)
def _kept_sample(idx: int, op: str, theta: dict, attr: str, primitive: str, tax) -> DataSample:
    gt = tax.ground_truth(op, theta)
    ann = CoTAnnotation.from_oracle_text(
        json.dumps(
            {
                "thought": "t",
                "attribution": attr,
                "action": {"primitive": primitive, "params": _PARAMS[primitive]},
            }
        ),
        synonyms=tax.synonyms,
        valid_categories=tax.valid_categories,
    )
    return DataSample(
        sample_id=f"{idx:04d}_{op}",
        snapshot=_snap(op, "solid_ground"),
        ground_truth=gt,
        annotation=ann,
        verdict=Verdict(True, KEEP),
        target_theta=[0.1, 0.0, 1.0, 1.0],
        ambiguity_pair=None,
    )


def _dropped_sample(idx: int, op: str, theta: dict, reason: str, detail: str, tax) -> DataSample:
    return DataSample(
        sample_id=f"{idx:04d}_{op}",
        snapshot=_snap(op, "solid_ground"),
        ground_truth=tax.ground_truth(op, theta),
        annotation=None,
        verdict=Verdict(False, reason, detail=detail),
        target_theta=[0.1, 0.0, 1.0, 1.0],
        ambiguity_pair=None,
    )


def test_merge_datasets_folds_in_ops_and_separates_oracle_error(cfg, tax, tmp_path):
    base_dir, add_dir = tmp_path / "base", tmp_path / "add"
    # Base: region ops kept, plus a LEGACY O10 api-failure mislabeled schema_invalid (pre-M1).
    base = PipelineResult(
        samples=[
            _kept_sample(0, "O1_mu_field", {}, "low_friction", "Set_Constraint", tax),
            _kept_sample(1, "O3_collapse", {}, "region_collapse", "Update_Topology", tax),
            _dropped_sample(
                2,
                "O10_effort_decay",
                {"floor": 0.2},
                "schema_invalid",
                "oracle call failed: HTTP 503",
                tax,
            ),
        ]
    )
    base.stats = compute_stats(cfg, base.samples, n_cells=3, n_no_interception=0, wall=1.0)
    write_dataset(base, cfg, base_dir, seed=1, oracle_name="scripted", git_commit="base")

    # Add: real O10 + O5 kept, plus one O5 that hit a gateway error.
    add = PipelineResult(
        samples=[
            _kept_sample(
                0, "O10_effort_decay", {"floor": 0.13}, "effort_decay", "Switch_Gait", tax
            ),
            _kept_sample(1, "O5_payload", {"mass_kg": 16.0}, "overload", "Hold_and_Request", tax),
            _dropped_sample(
                2,
                "O5_payload",
                {"mass_kg": 16.0},
                ORACLE_ERROR,
                "oracle call failed: HTTP 429",
                tax,
            ),
        ]
    )
    add.stats = compute_stats(cfg, add.samples, n_cells=3, n_no_interception=0, wall=1.0)
    write_dataset(add, cfg, add_dir, seed=1, oracle_name="api", git_commit="add")

    card = merge_datasets(
        base_dir, add_dir, ["O5_payload", "O10_effort_decay"], cfg=cfg, seed=1, oracle_name="merged"
    )
    s = card["stats"]
    # 2 region kept + O10 + O5 = 4 kept; the legacy base O10 api-fail is replaced (not duplicated).
    assert s["kept"] == 4
    assert s["per_operator_kept"].get("O10_effort_decay") == 1
    assert s["per_operator_kept"].get("O5_payload") == 1
    # the O5 gateway error is ORACLE_ERROR, excluded from the filter reject-rate (M1)
    assert s["oracle_error"] == 1
    assert s["dropped"] == 0
    assert s["reject_rate"] == 0.0
    # frames for every kept sample are present (no missing/duplicate keys)
    frames = np.load(base_dir / "frames.npz")
    for op in ("O1_mu_field", "O3_collapse", "O10_effort_decay", "O5_payload"):
        sid = next(
            r["sample_id"]
            for r in _read(base_dir / "samples.jsonl")
            if r["snapshot"]["operator_name"] == op
        )
        assert f"{sid}__rgb" in frames
    # the card grew the standard release sections (m2)
    md = (base_dir / "dataset_card.md").read_text()
    assert "## Intended use" in md and "## License" in md and "## Limitations" in md


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
