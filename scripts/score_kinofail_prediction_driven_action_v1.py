#!/usr/bin/env python3
"""Score frozen attribution models on the primary view of action-study cases.

This program performs inference only.  It reuses the exact frozen checkpoints
and feature pipeline used by the paper's Scale benchmark, restricts evaluation
to the primary appearance view, and never opens action outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.known_multimodal_fusions import (  # noqa: E402
    FusionDimensions,
    build_fusion_model,
)
from scripts.evaluate_kinofail_known_fusion_baselines_v1 import (  # noqa: E402
    METHODS,
    MODEL_SOURCE,
    SCRIPT as FROZEN_SCORER,
    _load_confirmation,
    _normalize,
    _predict,
)


REFERENCE_METHODS = (
    "vision",
    "proprioception",
    "joint_early_fusion",
    "late_fusion",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def action_groups(case_csv: Path) -> set[str]:
    with case_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = {str(row["case_id"]).rsplit("__", 1)[-1] for row in rows}
    if len(rows) != 277 or len(groups) != 277 or not all(
        value.startswith("cf_") for value in groups
    ):
        raise RuntimeError("action-study case identifiers do not define 277 unique groups")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--action-case-csv", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    predictions_path = output / "primary_view_predictions.jsonl"
    provenance_path = output / "prediction_provenance.json"
    if predictions_path.exists() or provenance_path.exists():
        raise FileExistsError("prediction-driven action inference is score-once")

    freeze_manifest_path = args.freeze_manifest.resolve()
    freeze_manifest = json.loads(freeze_manifest_path.read_text())
    checkpoint_path = ROOT / freeze_manifest["artifacts"]["checkpoint"]
    if freeze_manifest.get("status") != "frozen_before_known_fusion_confirmation_inference":
        raise RuntimeError("unexpected checkpoint status")
    expected = freeze_manifest["source_sha256"]
    if (
        sha256(checkpoint_path) != freeze_manifest["artifacts"]["checkpoint_sha256"]
        or sha256(FROZEN_SCORER) != expected["script"]
        or sha256(MODEL_SOURCE) != expected["model_source"]
    ):
        raise RuntimeError("frozen attribution checkpoint seal failed")

    groups = action_groups(args.action_case_csv.resolve())
    confirmation = _load_confirmation()["scale"]
    indices = np.asarray(
        [
            index
            for index, row in enumerate(confirmation["rows"])
            if str(row["counterfactual_group_id"]) in groups
            and str(row["condition"]) == "anomaly"
            and str(row["appearance_intervention_id"]) == "primary"
        ],
        dtype=np.int64,
    )
    if len(indices) != 277:
        raise RuntimeError(f"expected 277 primary-view action cases, found {len(indices)}")

    reference = {
        str(row["sample_id"]): row
        for row in jsonl(args.reference_predictions.resolve())
        if str(row.get("battery")) == "scale"
    }
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = bundle["datasets"]["scale"]
    classes = [str(value) for value in saved["classes"]]
    dims = FusionDimensions(**saved["dimensions"])
    normalizer = saved["normalizer"]
    visual = torch.from_numpy(
        _normalize(
            confirmation["visual_fusion"][indices],
            normalizer["visual_mean"],
            normalizer["visual_scale"],
        )
    )
    proprio = torch.from_numpy(
        _normalize(
            confirmation["proprio_fusion"][indices],
            normalizer["proprio_mean"],
            normalizer["proprio_scale"],
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_probabilities: dict[str, np.ndarray] = {}
    for method in METHODS:
        seed_values: list[np.ndarray] = []
        for state in saved["states"][method]:
            model = build_fusion_model(method, dims)
            model.load_state_dict(state)
            model.to(device)
            seed_values.append(_predict(model, visual, proprio, device=device))
        neural_probabilities[method] = np.mean(seed_values, axis=0)

    rows: list[dict[str, Any]] = []
    for local_index, source_index in enumerate(indices.tolist()):
        source = confirmation["rows"][source_index]
        sample_id = str(source["sample_id"])
        reference_row = reference.get(sample_id)
        if reference_row is None:
            raise RuntimeError(f"missing frozen reference prediction: {sample_id}")
        predictions = {
            method: str(reference_row["predictions"][method])
            for method in REFERENCE_METHODS
        }
        confidence: dict[str, float | None] = {method: None for method in REFERENCE_METHODS}
        class_probability: dict[str, dict[str, float]] = {}
        for method, values in neural_probabilities.items():
            probability = values[local_index]
            predictions[method] = classes[int(probability.argmax())]
            confidence[method] = float(probability.max())
            class_probability[method] = {
                label: float(value)
                for label, value in zip(classes, probability.tolist(), strict=True)
            }
        rows.append(
            {
                "counterfactual_group_id": str(source["counterfactual_group_id"]),
                "sample_id": sample_id,
                "scene": str(source["scene_cluster"]),
                "material": str(source["cluster_material"]),
                "truth": str(source["attribution_category"]),
                "predictions": predictions,
                "confidence": confidence,
                "neural_class_probability": class_probability,
            }
        )
    if {row["counterfactual_group_id"] for row in rows} != groups:
        raise RuntimeError("action-study prediction join is not one-to-one")

    output.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "kinofail.prediction-driven-action-predictions.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_models_scored_once_on_primary_views",
        "cases": len(rows),
        "appearance_view": "primary only; appearance swaps are not policy inputs",
        "action_outcomes_opened": False,
        "methods": sorted(rows[0]["predictions"]),
        "source_sha256": {
            "freeze_manifest": sha256(freeze_manifest_path),
            "checkpoint": sha256(checkpoint_path),
            "frozen_scorer": sha256(FROZEN_SCORER),
            "model_source": sha256(MODEL_SOURCE),
            "action_case_csv": sha256(args.action_case_csv.resolve()),
            "reference_predictions": sha256(args.reference_predictions.resolve()),
            "this_script": sha256(Path(__file__).resolve()),
        },
        "artifact": {
            "path": str(predictions_path.relative_to(ROOT)),
            "sha256": sha256(predictions_path),
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": len(rows), "device": str(device)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
