from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.score_kinofail_unified_moe_v3_confirmatory import (
    BOOTSTRAP_REPLICATES,
    ONTOLOGY,
    main,
    score_confirmatory,
)

CHECKPOINTS = [f"seed{index}" for index in range(5)]
METHODS = [
    "learned_router",
    "late_average",
    "fixed_vision",
    "fixed_proprio",
    "fixed_joint",
]
CONFLICT_CLASSES = [
    "adhesion",
    "compliant_terrain",
    "low_friction",
    "invisible_obstacle",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _probability(prediction: str) -> dict[str, float]:
    residual = 0.1 / (len(ONTOLOGY) - 1)
    return {label: (0.9 if label == prediction else residual) for label in ONTOLOGY}


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    truth_rows = []
    prediction_rows = []

    for replicate in range(2):
        for index, truth in enumerate(ONTOLOGY):
            sample_id = f"scale_{replicate}_{index}"
            group_id = f"scale_group_{replicate}_{index}"
            scene = f"scene_{(index + replicate) % 4}"
            material = f"material_{(index + replicate) % 3}"
            truth_rows.append(
                {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "dataset": "scale",
                    "scene": scene,
                    "material": material,
                    "cluster_material": material,
                    "truth": truth,
                    "valid": True,
                }
            )
            wrong = ONTOLOGY[(index + 1) % len(ONTOLOGY)]
            for checkpoint in CHECKPOINTS:
                for method in METHODS:
                    prediction = truth
                    if method == "late_average" and index == 0:
                        prediction = wrong
                    elif method == "fixed_vision" and index % 2 == 0:
                        prediction = wrong
                    elif method == "fixed_proprio" and index % 2 == 1:
                        prediction = wrong
                    route = "consensus"
                    row = {
                        "sample_id": sample_id,
                        "method": method,
                        "checkpoint_id": checkpoint,
                        "prediction": prediction,
                        "probabilities": _probability(prediction),
                    }
                    if method == "learned_router":
                        row["evidence_route"] = route
                    prediction_rows.append(row)

    cells = [
        ("T2_vision_decisive", "vision"),
        ("T2_vision_decisive", "vision"),
        ("T3_proprio_decisive", "proprio"),
        ("T3_proprio_decisive", "proprio"),
    ]
    for index, (cell, expected_route) in enumerate(cells):
        truth = CONFLICT_CLASSES[index]
        wrong = ONTOLOGY[(ONTOLOGY.index(truth) + 1) % len(ONTOLOGY)]
        sample_id = f"conflict_{index}"
        group_id = f"conflict_group_{index}"
        scene = f"scene_{index}"
        material = f"material_{index % 3}"
        truth_rows.append(
            {
                "sample_id": sample_id,
                "group_id": group_id,
                "dataset": "conflict",
                "scene": scene,
                "material": material,
                "cluster_material": material,
                "truth": truth,
                "valid": True,
                "cell": cell,
            }
        )
        for checkpoint in CHECKPOINTS:
            for method in METHODS:
                prediction = truth
                if method == "late_average" and index == 0:
                    prediction = wrong
                if method == "fixed_vision":
                    prediction = truth if cell.startswith("T2") else wrong
                if method == "fixed_proprio":
                    prediction = wrong if cell.startswith("T2") else truth
                row = {
                    "sample_id": sample_id,
                    "method": method,
                    "checkpoint_id": checkpoint,
                    "prediction": prediction,
                    "probabilities": _probability(prediction),
                }
                if method == "learned_router":
                    row["evidence_route"] = expected_route
                prediction_rows.append(row)

    truth_path = tmp_path / "truth.jsonl"
    prediction_path = tmp_path / "blind_predictions.jsonl"
    protocol_path = tmp_path / "protocol.json"
    _write_jsonl(truth_path, truth_rows)
    _write_jsonl(prediction_path, prediction_rows)
    protocol = {
        "schema_version": "test.confirmatory.v1",
        "protocol_id": "synthetic-test",
        "ontology": list(ONTOLOGY),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": 20260725,
        },
        "checkpoint_ids": CHECKPOINTS,
        "methods": METHODS,
        "secondary_baselines": [
            "fixed_vision",
            "fixed_proprio",
            "fixed_joint",
        ],
        "expected_classes": {
            "scale": list(ONTOLOGY),
            "conflict": CONFLICT_CLASSES,
        },
        "planned_groups": {
            "scale": 2 * len(ONTOLOGY),
            "conflict": len(CONFLICT_CLASSES),
        },
        "decision_critical_sample_ids": [
            f"conflict_{index}" for index in range(len(CONFLICT_CLASSES))
        ],
        "input_sha256": {
            "blind_predictions": _sha256(prediction_path),
            "truth_key": _sha256(truth_path),
        },
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return protocol_path, prediction_path, truth_path


def test_confirmatory_scorer_passes_and_keeps_checkpoints_nested(
    tmp_path: Path,
) -> None:
    protocol, predictions, truth = _fixture(tmp_path)
    output = tmp_path / "report.json"
    report = score_confirmatory(protocol, predictions, truth, output)

    assert report["status"] == "confirmatory_passed_all_preregistered_gates"
    assert report["passed_all_preregistered_gates"] is True
    assert report["counts"]["fixed_checkpoints"] == 5
    assert report["counts"]["valid_groups"] == {
        "scale": 2 * len(ONTOLOGY),
        "conflict": len(CONFLICT_CLASSES),
    }
    assert report["statistical_contract"]["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
    assert (
        report["primary_cross_battery_gates"]["gate_2_worst_battery_superiority"]["sequential_pass"]
        is True
    )
    assert all(value["passed"] for value in report["route_fidelity_gates"].values())


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    protocol, predictions, truth = _fixture(tmp_path)
    output = tmp_path / "report.json"
    output.write_text('{"sentinel": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        score_confirmatory(protocol, predictions, truth, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"sentinel": True}
    assert (
        main(
            [
                "--protocol",
                str(protocol),
                "--blind-predictions",
                str(predictions),
                "--truth-key",
                str(truth),
                "--out",
                str(output),
            ]
        )
        == 3
    )


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    protocol, predictions, truth = _fixture(tmp_path)
    with predictions.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = tmp_path / "report.json"

    report = score_confirmatory(protocol, predictions, truth, output)
    assert report["confirmatory"] is False
    assert report["status"] == "fail_closed_integrity_or_structure_violation"
    assert "SHA-256" in report["failure_reason"]


def test_valid_negative_result_is_reported_without_reopening_protocol(
    tmp_path: Path,
) -> None:
    protocol, predictions, truth = _fixture(tmp_path)
    rows = [
        json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line
    ]
    learned = {
        (row["sample_id"], row["checkpoint_id"]): row
        for row in rows
        if row["method"] == "learned_router"
    }
    for row in rows:
        if row["method"] != "late_average":
            continue
        source = learned[(row["sample_id"], row["checkpoint_id"])]
        row["prediction"] = source["prediction"]
        row["probabilities"] = source["probabilities"]
    _write_jsonl(predictions, rows)
    config = json.loads(protocol.read_text(encoding="utf-8"))
    config["input_sha256"]["blind_predictions"] = _sha256(predictions)
    protocol.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "negative_report.json"

    report = score_confirmatory(protocol, predictions, truth, output)
    assert report["confirmatory"] is True
    assert report["status"] == "confirmatory_completed_statistical_gate_failed"
    assert report["passed_all_preregistered_gates"] is False
    assert (
        report["primary_cross_battery_gates"]["gate_2_worst_battery_superiority"]["raw_pass"]
        is False
    )
    assert (
        report["primary_cross_battery_gates"]["gate_3_equal_battery_macro_superiority"]["raw_pass"]
        is False
    )


def test_missing_frozen_class_fails_closed(tmp_path: Path) -> None:
    protocol, predictions, truth = _fixture(tmp_path)
    truth_rows = [
        json.loads(line) for line in truth.read_text(encoding="utf-8").splitlines() if line
    ]
    truth_rows = [
        row
        for row in truth_rows
        if not (row["dataset"] == "scale" and row["truth"] == ONTOLOGY[-1])
    ]
    _write_jsonl(truth, truth_rows)
    config = json.loads(protocol.read_text(encoding="utf-8"))
    config["planned_groups"]["scale"] -= 2
    config["input_sha256"]["truth_key"] = _sha256(truth)
    protocol.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    report = score_confirmatory(protocol, predictions, truth, output)
    assert report["confirmatory"] is False
    assert "class support" in report["failure_reason"]


def test_attrition_above_five_percent_is_descriptive_only(
    tmp_path: Path,
) -> None:
    protocol, predictions, truth = _fixture(tmp_path)
    truth_rows = [
        json.loads(line) for line in truth.read_text(encoding="utf-8").splitlines() if line
    ]
    invalid_samples = {
        next(
            row["sample_id"]
            for row in truth_rows
            if row["dataset"] == "scale" and row["truth"] == label
        )
        for label in ONTOLOGY[:2]
    }
    for row in truth_rows:
        if row["sample_id"] in invalid_samples:
            row["valid"] = False
    _write_jsonl(truth, truth_rows)

    prediction_rows = [
        json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line
    ]
    prediction_rows = [row for row in prediction_rows if row["sample_id"] not in invalid_samples]
    _write_jsonl(predictions, prediction_rows)
    config = json.loads(protocol.read_text(encoding="utf-8"))
    config["input_sha256"] = {
        "blind_predictions": _sha256(predictions),
        "truth_key": _sha256(truth),
    }
    protocol.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    report = score_confirmatory(protocol, predictions, truth, output)
    assert report["confirmatory"] is False
    assert report["status"] == "fail_closed_attrition_exceeded"
    assert report["attrition"]["scale"]["rate"] > 0.05
