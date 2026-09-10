"""Regression guards for the paper's registered evaluation paths."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publication_evaluation", ROOT / "scripts/evaluate_publication.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_registered_sources_exist():
    registry = evaluation.load_registry()
    assert set(registry["methods"]) == {
        "vision",
        "proprioception",
        "early_fusion",
        "late_fusion",
        "concat_mlp",
        "tfn",
        "lmf",
        "embracenet",
        "moddrop",
        "gmu",
    }
    assert len(registry["protected_dependencies"]) == len(set(registry["protected_dependencies"]))
    for source in registry["protected_entrypoints"] + registry["protected_dependencies"]:
        assert (ROOT / source).is_file(), source


@pytest.mark.parametrize("method", list(evaluation.load_registry()["methods"]))
def test_command_uses_registered_checkpoint_and_original_evaluator(method, tmp_path):
    registry = evaluation.load_registry()
    command = evaluation.score_command(registry, method, tmp_path / "out", tmp_path / "features")
    assert command[1] == str(ROOT / "scripts/evaluate_kinofail_single_gmu_v1.py")
    assert command[2] == "score"
    assert command[command.index("--freeze") + 1] == str(
        ROOT / registry["methods"][method]["freeze_dir"]
    )
    assert command[command.index("--all191") + 1] == str(tmp_path / "features")
    assert not {"--epochs", "--seed", "--learning-rate", "train"} & set(command)


def test_registered_model_structure():
    registry = evaluation.load_registry()
    for method, spec in registry["methods"].items():
        assert spec["config"]["method"] == method
        assert spec["config"]["hidden"] == 128
        assert spec["config"]["visual_mode"] == "dino"
        assert spec["training_seeds"] == [2026082411, 2026082412, 2026082413]
    assert registry["action"]["expected_cases"] == 277


def test_prediction_comparison_accepts_roundoff_only(tmp_path):
    reference = tmp_path / "reference.jsonl"
    actual = tmp_path / "actual.jsonl"
    row = {"physical_unit_id": "unit_0", "pred": "adhesion", "confidence": 0.9}
    reference.write_text(json.dumps(row) + "\n")
    actual.write_text(json.dumps({**row, "confidence": 0.9000001}) + "\n")
    assert evaluation.compare_predictions(actual, reference) == 1
    actual.write_text(json.dumps({**row, "pred": "low_friction"}) + "\n")
    with pytest.raises(RuntimeError, match="Prediction changed"):
        evaluation.compare_predictions(actual, reference)


def test_prediction_comparison_rejects_dropped_units(tmp_path):
    reference = tmp_path / "reference.jsonl"
    actual = tmp_path / "actual.jsonl"
    reference.write_text('{"id": 1}\n')
    actual.write_text("")
    with pytest.raises(RuntimeError, match="count changed"):
        evaluation.compare_predictions(actual, reference)


def test_configuration_drift_is_rejected_before_scoring(tmp_path):
    registry = copy.deepcopy(evaluation.load_registry())
    method = "gmu"
    freeze = tmp_path / "model"
    freeze.mkdir()
    registry["methods"][method]["freeze_dir"] = str(freeze)
    spec = registry["methods"][method]
    manifest = {
        spec["freeze_configuration_key"]: {"changed": True},
        "training": [{"seed": seed} for seed in spec["training_seeds"]],
    }
    (freeze / "freeze_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="configuration changed"):
        evaluation.check_paths(registry, [method], tmp_path / "features")
