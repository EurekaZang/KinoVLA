"""Hindsight-CoT pipeline + dataset tests (spec §10, M6 exit criteria 2 & 3).

Exercises PHASE 1→2→3→filter end-to-end on the surrogate (torch-free), and the dataset
format + auto-generated card: throughput/reject-rate reporting, per-operator counts, A/B
balance, ambiguity-pair coverage, and a write→read round-trip.
"""

from __future__ import annotations

import numpy as np

from kino_vla.data import (
    DROP_ATTRIBUTION,
    DROP_PRIMITIVE,
    DROP_SAFETY,
    DROP_SCHEMA,
    KEEP,
    FailureTaxonomy,
    ScriptedOracle,
    load_frames,
    read_dataset,
    run_pipeline,
    write_dataset,
)
from kino_vla.utils.config import load_config

_VALID_REASONS = {KEEP, DROP_ATTRIBUTION, DROP_PRIMITIVE, DROP_SAFETY, DROP_SCHEMA}


def _run(n_cells: int = 24, seed: int = 1):
    cfg = load_config("data/hindsight.yaml", {"maze.n_cells": n_cells})
    tax = FailureTaxonomy(cfg)
    oracle = ScriptedOracle(cfg, tax, seed=seed)
    return cfg, run_pipeline(cfg, seed=seed, oracle=oracle, taxonomy=tax)


def test_pipeline_runs_and_intercepts():
    _, result = _run()
    s = result.stats
    assert s.n_interceptions > 0
    # Every cell either intercepts (a sample) or is a recorded miss — nothing lost.
    assert s.n_interceptions + s.n_no_interception == s.n_cells
    assert len(result.samples) == s.n_interceptions


def test_pipeline_stats_are_consistent():
    _, result = _run()
    s = result.stats
    assert s.kept + s.dropped == s.n_interceptions
    assert 0.0 <= s.reject_rate <= 1.0
    assert set(s.by_reason).issubset(_VALID_REASONS)
    assert sum(s.by_reason.values()) == s.n_interceptions
    assert s.kept > 0  # the filter must keep grounded reflections, not drop everything


def test_pipeline_throughput_reported():
    cfg, result = _run()
    s = result.stats
    assert s.wall_time_s > 0.0
    assert s.samples_per_hour >= float(cfg.report.target_samples_per_hour)


def test_pipeline_is_deterministic():
    _, a = _run(n_cells=20, seed=2)
    _, b = _run(n_cells=20, seed=2)
    assert a.stats.kept == b.stats.kept
    assert a.stats.by_reason == b.stats.by_reason
    assert [s.sample_id for s in a.samples] == [s.sample_id for s in b.samples]


def test_ambiguity_coverage_reported():
    _, result = _run(n_cells=40)
    cov = result.stats.ambiguity_coverage
    assert len(cov) == 3  # the three Suite-Sem pairs (spec §8.3)
    for label, c in cov.items():
        assert "both_present" in c
        # each pair's two members appear as keys
        assert sum(1 for k in c if k != "both_present") == 2, label


def test_kept_samples_pass_the_filter():
    _, result = _run()
    for sample in result.samples:
        if sample.verdict.keep:
            assert sample.annotation is not None
            assert sample.annotation.attribution == sample.ground_truth.category
            assert sample.annotation.primitive.name in sample.ground_truth.feasible


def test_dataset_roundtrip(tmp_path):
    cfg, result = _run(n_cells=24, seed=1)
    card = write_dataset(
        result, cfg, tmp_path, seed=1, oracle_name="scripted", git_commit="testsha"
    )
    records, card_read = read_dataset(tmp_path)
    assert len(records) == result.stats.kept
    assert card_read["provenance"]["git_commit"] == "testsha"
    assert card["stats"]["kept"] == result.stats.kept
    assert (tmp_path / "dataset_card.md").exists()
    assert (tmp_path / "sample_annotations.md").exists()
    assert (tmp_path / "dropped.jsonl").exists()
    # the heavy modalities round-trip with the documented shapes
    kept = [s for s in result.samples if s.verdict.keep]
    frames = load_frames(tmp_path, kept[0].sample_id)
    assert frames["rgb"].shape[0] == int(cfg.snapshot.n_frames)
    assert frames["proprio"].shape == tuple(kept[0].snapshot.proprio_window.shape)


def test_manifest_records_have_target_theta():
    cfg, result = _run()
    kept = [s for s in result.samples if s.verdict.keep]
    rec = kept[0].to_record()
    assert len(rec["target_theta"]) == 4  # mu, payload_kg, effort_scale, support_ratio
    assert np.isfinite(rec["target_theta"]).all()
