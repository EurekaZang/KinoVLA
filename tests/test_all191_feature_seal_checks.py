from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARER = ROOT / "scripts/prepare_kino_v4_all191_features_v1.py"


def test_feature_seal_uses_positive_model_loading_gate() -> None:
    tree = ast.parse(PREPARER.read_text(encoding="utf-8"))
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "no_classifier_or_prediction_loaded" in strings
    assert "classifier_or_prediction_loaded" not in strings
