from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kino_vla.data.realistic_snapshots import _runtime_is_accepted, build_event_aligned_snapshots


def test_o4_backend_force_readback_recovers_only_known_validator_false_negative() -> None:
    schedule = {"condition": "anomaly", "target_operator": "O4_tether"}
    manifest = {
        "runtime_validation": {"passed": False, "issues": ["operator_local_qa_failed"]},
        "operator_readback": {
            "qa_passed": False,
            "telemetry": {
                "total_attachment_cycles": 2,
                "total_applied_force_n": 10.0,
                "total_tangential_work_j": 0.5,
            },
        },
    }
    assert _runtime_is_accepted(schedule, manifest, allowed_suffixes=[])
    manifest["operator_readback"]["telemetry"]["total_attachment_cycles"] = 0
    assert not _runtime_is_accepted(schedule, manifest, allowed_suffixes=[])


def test_o4_completed_force_history_is_valid_when_final_instantaneous_force_is_zero() -> None:
    schedule = {"condition": "anomaly", "target_operator": "O4_tether"}
    manifest = {
        "runtime_validation": {"passed": False, "issues": ["operator_local_qa_failed"]},
        "operator_readback": {
            "qa_passed": False,
            "telemetry": {
                "total_attachment_cycles": 9,
                "total_applied_force_n": 0.0,
                "total_tangential_work_j": 13.1,
                "feet": [{"max_force_n": 54.2}],
            },
        },
    }
    assert _runtime_is_accepted(schedule, manifest, allowed_suffixes=[])
    manifest["operator_readback"]["telemetry"]["feet"][0]["max_force_n"] = 0.0
    assert not _runtime_is_accepted(schedule, manifest, allowed_suffixes=[])
from kino_vla.data.runtime_manifest import schedule_record_sha256, sha256_file


def _write_episode(root: Path, schedule: dict, *, anomaly: bool) -> None:
    episode = root / schedule["required_outputs"]["episode_manifest"]
    directory = episode.parent
    (directory / "rgb").mkdir(parents=True)
    (directory / "rgb_views" / "swap_01").mkdir(parents=True)
    timestamps = np.arange(0.02, 1.22, 0.1)
    views = {}
    for view_index, (view_id, relative_dir) in enumerate(
        (("primary", "rgb"), ("swap_01", "rgb_views/swap_01"))
    ):
        entries = []
        for index, timestamp in enumerate(timestamps):
            relative = f"{relative_dir}/{index:06d}.png"
            path = directory / relative
            Image.fromarray(
                np.full((8, 12, 3), 30 + view_index * 80 + index, dtype=np.uint8)
            ).save(path)
            entries.append(
                {"path": relative, "timestamp_s": float(timestamp), "sha256": sha256_file(path)}
            )
        views[view_id] = entries
    proprio_timestamps = np.arange(0.02, 1.22, 0.02)
    np.savez_compressed(
        directory / "proprio.npz",
        features=np.arange(len(proprio_timestamps) * 3, dtype=np.float32).reshape(-1, 3),
        timestamp_s=proprio_timestamps,
        feature_names=np.asarray(["a", "b", "c"]),
    )
    rows = []
    for timestamp in proprio_timestamps:
        event = "attached" if anomaly and abs(timestamp - 0.42) < 1.0e-8 else "none"
        rows.append(
            {
                "timestamp_s": float(timestamp),
                "adhesion": {"feet": [{"event": event}]},
            }
        )
    telemetry = directory / "privileged.jsonl"
    telemetry.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    appearance_views = {
        view_id: {"appearance_id": f"app_{view_id}", "material_family": f"mat_{view_id}"}
        for view_id in views
    }
    manifest = {
        "evaluation_eligible": True,
        "runtime_validation": {"passed": True},
        "schedule_record_sha256": schedule_record_sha256(schedule),
        "artifacts": {
            "primary_rgb_view_id": "primary",
            "rgb_views": views,
            "proprio": {
                "path": "proprio.npz",
                "sha256": sha256_file(directory / "proprio.npz"),
                "features_key": "features",
                "timestamps_key": "timestamp_s",
            },
            "telemetry": {"path": "privileged.jsonl", "sha256": sha256_file(telemetry)},
        },
        "appearance_readback": {"views": appearance_views},
    }
    episode.write_text(json.dumps(manifest), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    schedule_path = tmp_path / "schedule.jsonl"
    corpus = tmp_path / "corpus"
    rows = []
    for condition, suffix in (("anomaly", "anomaly"), ("nominal_counterfactual", "nominal")):
        row = {
            "episode_id": f"cf_test_{suffix}",
            "counterfactual_group_id": "cf_test",
            "condition": condition,
            "target_operator": "O4_tether",
            "attribution_category": "adhesion" if condition == "anomaly" else "nominal",
            "domain": "life",
            "scene_family": "scene_1",
            "camera_profile": "go2_front_calib_c",
            "severity_id": "moderate",
            "required_outputs": {"episode_manifest": f"{suffix}/manifest.json"},
        }
        rows.append(row)
        _write_episode(corpus, row, anomaly=condition == "anomaly")
    schedule_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    protocol = {
        "protocol_id": "test",
        "development_only": True,
        "source_schedule": str(schedule_path),
        "source_corpus_root": str(corpus),
        "selection": {
            "operators_with_admitted_event_adapter": ["O4_tether"],
            "require_evaluation_eligible": True,
        },
        "temporal_alignment": {
            "decision_delay_s": 0.4,
            "rgb_offsets_from_decision_s": [-0.4, -0.3, -0.2, -0.1, 0.0],
            "max_rgb_target_skew_s": 0.041,
            "proprio_window_s": 0.5,
            "proprio_samples": 25,
            "max_proprio_end_skew_s": 0.021,
        },
        "publication_guard": {"may_satisfy_realistic_a0_a7": False},
    }
    return protocol, tmp_path


def test_event_aligned_snapshot_reuses_time_and_proprio_across_counterfactuals(tmp_path: Path):
    protocol, root = _fixture(tmp_path)
    records, arrays, audit = build_event_aligned_snapshots(protocol, repo_root=root)
    assert audit["passed"] is True
    assert audit["counts"] == {
        "snapshot_records": 4,
        "physical_episodes": 2,
        "independent_counterfactual_pairs": 1,
        "appearance_intervention_sequences": 4,
        "operators": 1,
        "domains": 1,
        "scene_families": 1,
        "temporally_infeasible_counterfactual_pairs": 0,
    }
    by_view = {row["sample_id"]: row for row in records}
    anomaly = by_view["cf_test_anomaly__primary"]
    nominal = by_view["cf_test_nominal__primary"]
    assert anomaly["event_time_s_from_anomaly_privileged_telemetry"] == pytest.approx(0.42)
    assert anomaly["rgb_timestamp_s"] == nominal["rgb_timestamp_s"]
    assert anomaly["proprio_timestamp_s"] == nominal["proprio_timestamp_s"]
    assert np.array_equal(
        arrays["cf_test_anomaly__primary__proprio"],
        arrays["cf_test_anomaly__swap_01__proprio"],
    )
    assert all(row["appearance_views_are_independent_samples"] is False for row in records)


def test_unknown_operator_has_no_heuristic_event_fallback(tmp_path: Path):
    protocol, root = _fixture(tmp_path)
    protocol["selection"]["operators_with_admitted_event_adapter"] = ["O5_payload"]
    records, arrays, audit = build_event_aligned_snapshots(protocol, repo_root=root)
    assert records == []
    assert arrays == {}
    assert audit["passed"] is False


def test_counterfactual_pair_can_be_selected_from_repair_overlay(tmp_path: Path):
    protocol, root = _fixture(tmp_path)
    base = Path(protocol["source_corpus_root"])
    repair = root / "repair_corpus"
    base.rename(repair)
    protocol["source_corpus_overlays"] = [
        {
            "corpus_root": str(repair),
            "required_collection_protocol_id": "",
            "counterfactual_group_ids": ["cf_test"],
        }
    ]
    records, _, audit = build_event_aligned_snapshots(protocol, repo_root=root)
    assert audit["passed"] is True
    assert len(records) == 4


def test_temporally_infeasible_condition_excludes_the_complete_pair_with_audit(
    tmp_path: Path,
):
    protocol, root = _fixture(tmp_path)
    nominal_manifest_path = (
        Path(protocol["source_corpus_root"]) / "nominal/manifest.json"
    )
    nominal_manifest = json.loads(nominal_manifest_path.read_text(encoding="utf-8"))
    for view_id in nominal_manifest["artifacts"]["rgb_views"]:
        nominal_manifest["artifacts"]["rgb_views"][view_id] = nominal_manifest[
            "artifacts"
        ]["rgb_views"][view_id][:8]
    nominal_manifest_path.write_text(
        json.dumps(nominal_manifest),
        encoding="utf-8",
    )
    protocol["selection"][
        "on_temporal_alignment_failure"
    ] = "exclude_complete_pair_and_audit"

    records, arrays, audit = build_event_aligned_snapshots(
        protocol,
        repo_root=root,
    )

    assert records == []
    assert arrays == {}
    assert audit["counts"]["temporally_infeasible_counterfactual_pairs"] == 1
    assert audit["temporal_alignment_exclusions"][0][
        "counterfactual_group_id"
    ] == "cf_test"
    assert audit["temporal_alignment_exclusions"][0][
        "failing_condition"
    ] == "nominal_counterfactual"
