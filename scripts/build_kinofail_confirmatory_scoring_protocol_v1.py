#!/usr/bin/env python3
"""Bind blind predictions and a separate truth key for one-shot scoring.

This program contains no model fitting, metric computation, or result-based
selection.  It derives the preregistered route-fidelity registry from every
valid conflict sample, verifies fixed group/class support, and writes a
write-once protocol containing the two input hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = (
    "adhesion",
    "compliant_terrain",
    "effort_decay",
    "external_push",
    "high_centering",
    "invisible_obstacle",
    "low_friction",
    "nominal",
    "obs_bias",
    "overload",
    "region_collapse",
)
CONFLICT_CLASSES = (
    "adhesion",
    "compliant_terrain",
    "invisible_obstacle",
    "low_friction",
)
METHODS = (
    "learned_router",
    "fixed_vision",
    "fixed_proprio",
    "fixed_joint",
    "late_average",
    "legacy_class_rule",
    "random_route",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(path)
    return rows


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-predictions", type=Path, required=True)
    parser.add_argument("--truth-key", type=Path, required=True)
    parser.add_argument("--f0-manifest", type=Path, required=True)
    parser.add_argument("--f1-manifest", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite scoring protocol: {out}")
    predictions_path = args.blind_predictions.resolve()
    truth_path = args.truth_key.resolve()
    frozen_paths = [
        args.f0_manifest.resolve(),
        args.f1_manifest.resolve(),
        args.feature_manifest.resolve(),
        Path(__file__).resolve(),
        ROOT / "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
    ]
    if any(not path.is_file() for path in [predictions_path, truth_path, *frozen_paths]):
        missing = [
            str(path)
            for path in [predictions_path, truth_path, *frozen_paths]
            if not path.is_file()
        ]
        raise FileNotFoundError(", ".join(missing))

    truth_rows = _jsonl(truth_path)
    prediction_rows = _jsonl(predictions_path)
    truth_ids = [str(row.get("sample_id", "")) for row in truth_rows]
    if not truth_ids or "" in truth_ids or len(truth_ids) != len(set(truth_ids)):
        raise RuntimeError("truth key must contain unique nonempty sample IDs")
    if any(
        key in row
        for row in prediction_rows
        for key in ("truth", "label", "attribution_category", "expected_route")
    ):
        raise RuntimeError("blind prediction file contains a forbidden truth field")

    group_dataset: dict[str, str] = {}
    for row in truth_rows:
        dataset = str(row.get("dataset"))
        group_id = str(row.get("group_id", ""))
        if dataset not in {"scale", "conflict"} or not group_id:
            raise RuntimeError("truth key contains an invalid dataset/group")
        previous = group_dataset.setdefault(group_id, dataset)
        if previous != dataset:
            raise RuntimeError(f"group crosses batteries: {group_id}")
    counts = Counter(group_dataset.values())
    if counts != Counter({"scale": 10_560, "conflict": 3_000}):
        raise RuntimeError(f"unexpected planned group counts: {dict(counts)}")

    class_support = {
        dataset: {
            str(row["truth"])
            for row in truth_rows
            if row["dataset"] == dataset and bool(row.get("valid", True))
        }
        for dataset in ("scale", "conflict")
    }
    if class_support["scale"] != set(ONTOLOGY):
        raise RuntimeError("valid scale truth does not span the frozen ontology")
    if class_support["conflict"] != set(CONFLICT_CLASSES):
        raise RuntimeError("valid conflict truth does not span the frozen class support")
    critical_ids = sorted(
        str(row["sample_id"])
        for row in truth_rows
        if row["dataset"] == "conflict"
        and bool(row.get("valid", True))
        and str(row.get("cell", ""))
        in {"T2_vision_decisive", "T3_proprio_decisive"}
    )
    if not critical_ids or len(critical_ids) != len(set(critical_ids)):
        raise RuntimeError("decision-critical registry is empty or duplicated")

    protocol = {
        "schema_version": "kinofail.unified-confirmatory-scoring-protocol.v1",
        "protocol_id": "kinofail-unified-moe-v3-confirmatory-score-once-20260725",
        "status": "sealed_after_blind_prediction_before_scoring",
        "ontology": list(ONTOLOGY),
        "checkpoint_ids": [f"seed{index}" for index in range(5)],
        "methods": list(METHODS),
        "secondary_baselines": [
            "fixed_vision",
            "fixed_proprio",
            "fixed_joint",
            "legacy_class_rule",
            "random_route",
        ],
        "planned_groups": {"scale": 10_560, "conflict": 3_000},
        "expected_classes": {
            "scale": list(ONTOLOGY),
            "conflict": list(CONFLICT_CLASSES),
        },
        "decision_critical_sample_ids": critical_ids,
        "decision_critical_selection": (
            "all valid preregistered T2/T3 samples; no model output or correctness "
            "is used by this builder"
        ),
        "bootstrap": {"replicates": 20_000, "seed": 2_027_012_751},
        "input_sha256": {
            "blind_predictions": _sha256(predictions_path),
            "truth_key": _sha256(truth_path),
        },
        "frozen_files_sha256": {
            _display(path): _sha256(path) for path in frozen_paths
        },
        "input_paths": {
            "blind_predictions": _display(predictions_path),
            "truth_key": _display(truth_path),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "protocol": str(out),
                "sha256": _sha256(out),
                "decision_critical_samples": len(critical_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
