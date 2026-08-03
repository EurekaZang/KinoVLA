from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import build_kinofail_realistic_snapshot_dev as snapshot_entry
from scripts.kinofail_reconfirmation_attrition_f13 import build_ledger


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    schedule = tmp_path / "c2_t3" / "schedule.jsonl"
    schedule.parent.mkdir(parents=True)
    rows = []
    for pair_id in ("cf_good", "cf_interrupted"):
        for condition in ("anomaly", "nominal_counterfactual"):
            rows.append(
                {
                    "scene_family": "scene_00",
                    "counterfactual_group_id": pair_id,
                    "condition": condition,
                }
            )
    schedule.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    audits = tmp_path / "launcher_audits"
    _write_json(
        audits / "c2_t3_f12v9_partition_0_of_1.json",
        {
            "scene_id": "scene_00",
            "battery": "c2_t3",
            "attempts": [
                {
                    "counterfactual_group_id": "cf_good",
                    "state": "terminal",
                    "returncode": 0,
                    "summary_exists": True,
                    # A narrow physical certificate may accept this later.
                    "passed": False,
                },
                {
                    "counterfactual_group_id": "cf_interrupted",
                    "state": "started",
                    # Complete files do not override launcher attrition.
                    "summary_exists": True,
                },
            ],
        },
    )
    return schedule, audits


def test_ledger_uses_launcher_terminality_not_physical_outcome(
    tmp_path: Path,
) -> None:
    schedule, audits = _fixture(tmp_path)
    ledger = build_ledger(
        schedule_path=schedule,
        launcher_audit_dir=audits,
        battery="c2_t3",
        output_path=tmp_path / "ledger.json",
    )
    assert ledger["eligible_pair_ids"] == ["cf_good"]
    assert [
        row["counterfactual_group_id"] for row in ledger["attrition"]
    ] == ["cf_interrupted"]
    assert ledger["model_feature_label_outcome_or_score_read"] is False


def test_duplicate_attempt_fails_closed(tmp_path: Path) -> None:
    schedule, audits = _fixture(tmp_path)
    duplicate = json.loads(
        (audits / "c2_t3_f12v9_partition_0_of_1.json").read_text()
    )
    duplicate["attempts"] = [duplicate["attempts"][0]]
    _write_json(
        audits / "c2_t3_f11v8_partition_0_of_1.json",
        duplicate,
    )
    with pytest.raises(RuntimeError, match="duplicate launcher attempt"):
        build_ledger(
            schedule_path=schedule,
            launcher_audit_dir=audits,
            battery="c2_t3",
            output_path=tmp_path / "ledger.json",
        )


def test_snapshot_filter_removes_predeclared_attrition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    _write_json(ledger_path, {"placeholder": True})
    ledger = {
        "eligible_pair_ids": ["cf_good"],
        "attrition": [
            {
                "counterfactual_group_id": "cf_interrupted",
                "reason": "interrupted_started_without_retry",
            }
        ],
        "counts": {"planned_pairs": 2},
        "operational_amendment": "F13",
        "operational_amendment_sha256": "f13",
    }
    monkeypatch.setattr(
        snapshot_entry,
        "build_for_protocol",
        lambda **_: (ledger, ledger_path),
    )
    records = [
        {
            "sample_id": "cf_good_anomaly__primary",
            "counterfactual_group_id": "cf_good",
            "physical_episode_id": "cf_good_anomaly",
            "target_operator": "O4_tether",
            "domain": "life",
            "scene_family": "scene_00",
            "appearance_views_are_independent_samples": False,
        },
        {
            "sample_id": "cf_good_nominal__primary",
            "counterfactual_group_id": "cf_good",
            "physical_episode_id": "cf_good_nominal",
            "target_operator": "O4_tether",
            "domain": "life",
            "scene_family": "scene_00",
            "appearance_views_are_independent_samples": False,
        },
        {
            "sample_id": "cf_interrupted_anomaly__primary",
            "counterfactual_group_id": "cf_interrupted",
            "physical_episode_id": "cf_interrupted_anomaly",
            "target_operator": "O8_invisible_collider",
            "domain": "life",
            "scene_family": "scene_00",
            "appearance_views_are_independent_samples": False,
        },
        {
            "sample_id": "cf_interrupted_nominal__primary",
            "counterfactual_group_id": "cf_interrupted",
            "physical_episode_id": "cf_interrupted_nominal",
            "target_operator": "O8_invisible_collider",
            "domain": "life",
            "scene_family": "scene_00",
            "appearance_views_are_independent_samples": False,
        },
    ]
    arrays = {
        "cf_good_anomaly__primary__rgb": np.zeros((1,)),
        "cf_good_nominal__primary__rgb": np.zeros((1,)),
        "cf_interrupted_anomaly__primary__rgb": np.zeros((1,)),
        "cf_interrupted_nominal__primary__rgb": np.zeros((1,)),
    }
    audit = {
        "passed": True,
        "checks": {
            "all_records_from_complete_pairs": True,
            "all_selected_or_excluded_pairs_accounted_for": True,
            "has_at_least_one_pair": True,
        },
        "counts": {
            "snapshot_records": 4,
            "physical_episodes": 4,
            "independent_counterfactual_pairs": 2,
            "appearance_intervention_sequences": 4,
            "operators": 2,
            "domains": 1,
            "scene_families": 1,
            "temporally_infeasible_counterfactual_pairs": 0,
        },
        "skipped_incomplete_pairs": 0,
        "pair_audits": [
            {
                "counterfactual_group_id": "cf_good",
                "appearance_views_per_physical_episode": 1,
            },
            {
                "counterfactual_group_id": "cf_interrupted",
                "appearance_views_per_physical_episode": 1,
            },
        ],
        "temporal_alignment_exclusions": [],
    }
    filtered_records, filtered_arrays, corrected = (
        snapshot_entry._apply_launcher_eligibility(
            records,
            arrays,
            audit,
            protocol={"protocol_id": "reconfirmation-v2-t3-test"},
            repo_root=tmp_path,
        )
    )
    assert {
        row["counterfactual_group_id"] for row in filtered_records
    } == {"cf_good"}
    assert list(filtered_arrays) == [
        "cf_good_anomaly__primary__rgb",
        "cf_good_nominal__primary__rgb",
    ]
    assert corrected["skipped_incomplete_pairs"] == 1
    assert corrected["counts"]["independent_counterfactual_pairs"] == 1
    assert corrected["passed"] is True
