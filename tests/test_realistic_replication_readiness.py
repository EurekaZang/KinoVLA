from __future__ import annotations

import json
from pathlib import Path

from kino_vla.eval.realistic_replication_readiness import (
    audit_realistic_replication_readiness,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _contract() -> dict:
    return {
        "contract_id": "test",
        "status": "frozen",
        "benchmark": {
            "physical_episodes": 396,
            "counterfactual_groups": 198,
            "domains": 3,
            "scene_families": 9,
            "operators": 11,
            "camera_profiles": 3,
        },
        "evidence_roots": {
            "scene_registry": "registry.json",
            "runtime_audit": "runtime.json",
            "runtime_coverage": "coverage.json",
            "snapshot_development_audit": "snapshot_audit.json",
            "training_manifest": "training_manifest.json",
            "inference_manifest": "inference_manifest.json",
            "predictions": "predictions.jsonl",
        },
        "experiments": {
            key: {"output": f"{key.lower()}.json"}
            for key in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")
        },
    }


def test_partial_o4_data_cannot_satisfy_realistic_a0(tmp_path: Path) -> None:
    _write(tmp_path / "registry.json", {"passed": True, "publication_ready": False})
    _write(
        tmp_path / "runtime.json",
        {
            "passed": True,
            "publication_freeze_ready": False,
            "evaluation_eligible_records": 4,
            "complete_counterfactual_pairs": 2,
        },
    )
    _write(
        tmp_path / "coverage.json",
        {
            "publication_ready": False,
            "observed": {
                "scene_families": 1,
                "domains": 1,
                "operators": 1,
                "camera_profiles": 1,
            },
            "measured_texture_swap": {"all_pass": True},
        },
    )
    _write(
        tmp_path / "snapshot_audit.json",
        {
            "passed": True,
            "counts": {"independent_counterfactual_pairs": 2, "snapshot_records": 12},
            "event_adapter_coverage": {"implemented": ["O4_tether"]},
            "publication_guard": {"may_satisfy_realistic_a0_a7": False},
        },
    )
    result = audit_realistic_replication_readiness(_contract(), root=tmp_path)
    assert not result["experiments"]["A0"]["ready"]
    assert result["n_ready"] == 0
    assert not result["realistic_a0_a7_complete"]
    milestone = result["development_milestones"]["event_aligned_snapshot_pipeline"]
    assert milestone["extraction_passed"] is True
    assert milestone["counts_as_experiment_ready"] is False


def test_all_realistic_outputs_are_required(tmp_path: Path) -> None:
    _write(
        tmp_path / "registry.json",
        {
            "passed": True,
            "publication_ready": True,
            "coverage": {
                "domain_scene_counts": {"life": 3, "production": 3, "wild": 3},
                "all_11_operators_scene_admitted": True,
            },
            "schedule_binding": {"fully_bound": True},
        },
    )
    _write(
        tmp_path / "runtime.json",
        {
            "passed": True,
            "publication_freeze_ready": True,
            "evaluation_eligible_records": 396,
            "complete_counterfactual_pairs": 198,
        },
    )
    _write(
        tmp_path / "coverage.json",
        {
            "publication_ready": True,
            "observed": {
                "scene_families": 9,
                "domains": 3,
                "operators": 11,
                "camera_profiles": 3,
            },
            "measured_texture_swap": {"all_pass": True},
        },
    )
    preliminary = audit_realistic_replication_readiness(_contract(), root=tmp_path)
    bundle_sha = preliminary["completion_invariant"]["a0_evidence_bundle_sha256"]
    _write(
        tmp_path / "a0.json",
        {
            "status": "confirmatory_complete",
            "dataset_scope": "kinofail_realistic",
            "a0_evidence_bundle_sha256": bundle_sha,
        },
    )
    _write(
        tmp_path / "training_manifest.json",
        {
            "status": "confirmatory_complete",
            "dataset_scope": "kinofail_realistic",
            "a0_evidence_bundle_sha256": bundle_sha,
            "model_runs": [{"seed": 1}],
        },
    )
    training_sha = __import__("hashlib").sha256(
        (tmp_path / "training_manifest.json").read_bytes()
    ).hexdigest()
    (tmp_path / "predictions.jsonl").write_text('{"prediction": "O1"}\n')
    predictions_sha = __import__("hashlib").sha256(
        (tmp_path / "predictions.jsonl").read_bytes()
    ).hexdigest()
    _write(
        tmp_path / "inference_manifest.json",
        {
            "status": "confirmatory_complete",
            "dataset_scope": "kinofail_realistic",
            "a0_evidence_bundle_sha256": bundle_sha,
            "training_manifest_sha256": training_sha,
            "predictions_sha256": predictions_sha,
        },
    )
    inference_sha = __import__("hashlib").sha256(
        (tmp_path / "inference_manifest.json").read_bytes()
    ).hexdigest()
    for key in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        value = {
            "status": "confirmatory_complete",
            "dataset_scope": "kinofail_realistic",
            "a0_evidence_bundle_sha256": bundle_sha,
        }
        if key in {"A2", "A3", "A5", "A7"}:
            value.update(
                {
                    "training_manifest_sha256": training_sha,
                    "inference_manifest_sha256": inference_sha,
                    "predictions_sha256": predictions_sha,
                }
            )
        _write(tmp_path / f"{key.lower()}.json", value)
    result = audit_realistic_replication_readiness(_contract(), root=tmp_path)
    assert result["experiments"]["A0"]["ready"]
    assert result["n_ready"] == 8
    assert result["realistic_a0_a7_complete"]
    assert result["completion_invariant"]["satisfied"]


def test_controlled_style_status_files_cannot_bypass_realistic_hash_chain(
    tmp_path: Path,
) -> None:
    for key in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        _write(tmp_path / f"{key.lower()}.json", {"status": "confirmatory_complete"})
    (tmp_path / "predictions.jsonl").write_text('{"legacy": true}\n')
    result = audit_realistic_replication_readiness(_contract(), root=tmp_path)
    assert result["n_ready"] == 0
    assert not result["completion_invariant"]["satisfied"]
    assert not result["publication_verdict"]["kinofail_realistic_benchmark_ready"]
