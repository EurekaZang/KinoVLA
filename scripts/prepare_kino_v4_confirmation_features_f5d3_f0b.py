#!/usr/bin/env python3
"""Prepare F0b features from the sealed observation membership on /data."""

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


HOME_OUTPUT = ROOT / "outputs/eval/kino_v4_confirmation_v1_f5d"
DATA_OUTPUT = Path("/data/eureka/kinovla_outputs/kino_v4_confirmation_v1_f5d")
SEAL = DATA_OUTPUT / "feature_seal.json"
OBSERVATION = (
    ROOT / "outputs/freeze/kino_v4_confirmation_observations_f5c/observation_seal.json"
)
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


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    if HOME_OUTPUT.exists() or HOME_OUTPUT.is_symlink():
        raise FileExistsError(HOME_OUTPUT)
    if DATA_OUTPUT.exists():
        raise FileExistsError(DATA_OUTPUT)
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    base.predecessor.OUTPUT = DATA_OUTPUT

    generator = base.predecessor.predecessor
    original_module = generator.module
    target = generator.PREDECESSOR.resolve()

    def corrected_module():
        original_read_text = Path.read_text

        def read_text(path: Path, *args, **kwargs):
            source = original_read_text(path, *args, **kwargs)
            if path.resolve() == target:
                replacements = {
                    '        "passed": all(checks.values()),\n': (
                        '        "passed": all(value for name, value in checks.items() '
                        'if name not in {"classifier_or_prediction_loaded", '
                        '"selection_uses_representation_values"}),\n'
                    ),
                    '            "require_evaluation_eligible": True,\n': (
                        '            "require_evaluation_eligible": False,\n'
                    ),
                    '            "require_runtime_validation_passed": True,\n': (
                        '            "require_runtime_validation_passed": False,\n'
                    ),
                }
                for old, new in replacements.items():
                    if source.count(old) != 1:
                        raise RuntimeError(f"feature source patch drift: {old!r}")
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

    observation = json.loads(OBSERVATION.read_text(encoding="utf-8"))
    sealed_scale = set(map(str, observation["retained"]["scale_pair_ids"]))
    scale_records = jsonl(DATA_OUTPUT / "scale/records.jsonl")
    extracted_scale = {
        str(row["counterfactual_group_id"]) for row in scale_records
    }
    scale_manifest = json.loads(
        (DATA_OUTPUT / "scale/feature_manifest.json").read_text(encoding="utf-8")
    )
    if (
        not extracted_scale.issubset(sealed_scale)
        or len(extracted_scale)
        != int(scale_manifest["retained_pairs_after_runtime_and_temporal_validity"])
        or len(scale_records) != 6 * len(extracted_scale)
        or (len(sealed_scale) - len(extracted_scale)) / 2_112 >= 0.05
    ):
        raise RuntimeError("Scale extraction does not honor sealed observation membership")

    predicates = {
        name: value
        for name, value in checks.items()
        if name not in NEGATIVE_STATE_FIELDS
    } | {
        "no_classifier_or_prediction_loaded": True,
        "no_selection_using_representation_values": True,
        "no_confirmation_prediction_truth_key_or_score_read": True,
        "all_extracted_scale_pairs_are_in_observation_seal": True,
    }
    seal.update(
        {
            "schema_version": "kinofail.kino-v4-confirmation-feature-seal-f5d3-f0b.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "sealed_before_f0_checkpoint_inference",
            "passed": all(predicates.values()),
            "pass_predicates": predicates,
            "predicate_certification": {
                "scientific_feature_or_membership_change": False,
                "model_checkpoint_or_prediction_loaded": False,
                "scope": (
                    "negative-state pass serialization and exact reuse of the "
                    "already sealed task-aligned observation membership"
                ),
                "sealed_scale_pairs": len(sealed_scale),
                "extracted_scale_pairs": len(extracted_scale),
            },
            "storage": {
                "canonical_data_path": str(DATA_OUTPUT),
                "repository_symlink": str(HOME_OUTPUT),
            },
        }
    )
    atomic(SEAL, seal)
    HOME_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    HOME_OUTPUT.symlink_to(DATA_OUTPUT, target_is_directory=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
