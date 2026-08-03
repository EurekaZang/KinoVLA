from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from kino_vla.eval.realistic_multimodal import ObservableClassifier
from kino_vla.eval.unified_moe import UnifiedEvidenceMoE
from scripts.predict_kinofail_unified_moe_v3_blind import (
    _ensure_new_output,
    _install_training_guards,
    load_blind_features,
)


def test_blind_bundle_accepts_only_three_observable_arrays(
    tmp_path,
) -> None:
    path = tmp_path / "blind.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["sample_0", "sample_1"]),
        visual=np.zeros((2, 1536), dtype=np.float32),
        proprio=np.zeros((2, 80), dtype=np.float32),
    )
    sample_ids, visual, proprio, keys = load_blind_features(path)
    assert sample_ids.tolist() == ["sample_0", "sample_1"]
    assert visual.shape == (2, 1536)
    assert proprio.shape == (2, 80)
    assert set(keys) == {"sample_ids", "visual", "proprio"}


@pytest.mark.parametrize(
    "forbidden_key",
    ["truth", "scene_id", "material_family", "target_operator", "outcome"],
)
def test_blind_bundle_rejects_privileged_or_label_metadata(
    tmp_path,
    forbidden_key,
) -> None:
    path = tmp_path / f"{forbidden_key}.npz"
    values = {
        "sample_ids": np.asarray(["sample_0"]),
        "visual": np.zeros((1, 1536), dtype=np.float32),
        "proprio": np.zeros((1, 80), dtype=np.float32),
        forbidden_key: np.asarray(["forbidden"]),
    }
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="forbidden metadata"):
        load_blind_features(path)


def test_blind_predictor_output_is_write_once(tmp_path) -> None:
    output = tmp_path / "prediction"
    _ensure_new_output(output)
    output.mkdir()
    with pytest.raises(FileExistsError):
        _ensure_new_output(output)


def test_training_guards_block_both_training_entry_points(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        UnifiedEvidenceMoE,
        "fit",
        UnifiedEvidenceMoE.__dict__["fit"],
    )
    monkeypatch.setattr(
        ObservableClassifier,
        "fit",
        ObservableClassifier.__dict__["fit"],
    )
    counter = _install_training_guards()
    with pytest.raises(RuntimeError, match="forbidden"):
        UnifiedEvidenceMoE.fit(
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.asarray(["x"]),
            np.asarray(["e"]),
            np.asarray(["g"]),
            seed=0,
        )
    with pytest.raises(RuntimeError, match="forbidden"):
        ObservableClassifier.fit(
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.asarray(["x"]),
            np.asarray(["e"]),
            feature_key="vision",
            seed=0,
        )
    assert counter["fit_attempts"] == 2


def test_blind_predictor_source_never_calls_fit() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/predict_kinofail_unified_moe_v3_blind.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
    ]
    assert fit_calls == []
