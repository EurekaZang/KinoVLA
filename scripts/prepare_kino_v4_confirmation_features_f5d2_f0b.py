#!/usr/bin/env python3
"""Prepare F0b features with negative state fields treated as predicates."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_kino_v4_confirmation_features_f5d_f0b as base


SEAL = ROOT / "outputs/eval/kino_v4_confirmation_v1_f5d/feature_seal.json"
NEGATIVE_STATE_FIELDS = {
    "classifier_or_prediction_loaded",
    "selection_uses_representation_values",
    "confirmation_prediction_truth_key_or_score_read",
}


def atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    generator = base.predecessor.predecessor
    original_module = generator.module
    target = generator.PREDECESSOR.resolve()

    def corrected_module():
        original_read_text = Path.read_text

        def read_text(path: Path, *args, **kwargs):
            source = original_read_text(path, *args, **kwargs)
            if path.resolve() == target:
                old = '        "passed": all(checks.values()),\n'
                new = (
                    '        "passed": all(value for name, value in checks.items() '
                    'if name not in {"classifier_or_prediction_loaded", '
                    '"selection_uses_representation_values"}),\n'
                )
                if source.count(old) != 1:
                    raise RuntimeError("feature-seal pass predicate patch drift")
                source = source.replace(old, new)
            return source

        Path.read_text = read_text
        try:
            return original_module()
        finally:
            Path.read_text = original_read_text

    generator.module = corrected_module
    result = int(base.main())
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    checks = seal.get("checks", {})
    if (
        seal.get("passed") is not False
        or not NEGATIVE_STATE_FIELDS.issubset(checks)
        or not all(checks[name] is False for name in NEGATIVE_STATE_FIELDS)
        or not all(
            value is True
            for name, value in checks.items()
            if name not in NEGATIVE_STATE_FIELDS
        )
    ):
        raise RuntimeError("feature seal does not have the exact negative-state bug")
    predicates = {
        name: value
        for name, value in checks.items()
        if name not in NEGATIVE_STATE_FIELDS
    } | {
        "no_classifier_or_prediction_loaded": True,
        "no_selection_using_representation_values": True,
        "no_confirmation_prediction_truth_key_or_score_read": True,
    }
    seal.update(
        {
            "schema_version": "kinofail.kino-v4-confirmation-feature-seal-f5d2-f0b.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "sealed_before_f0_checkpoint_inference",
            "passed": all(predicates.values()),
            "pass_predicates": predicates,
            "predicate_certification": {
                "scientific_feature_or_membership_change": False,
                "model_checkpoint_or_prediction_loaded": False,
                "scope": "negative-state pass serialization only",
            },
        }
    )
    atomic(SEAL, seal)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
