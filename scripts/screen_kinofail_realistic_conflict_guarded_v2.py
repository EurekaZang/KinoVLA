#!/usr/bin/env python3
"""One-shot development screen for the conflict-guarded bidirectional model.

The screen uses the already-observed scale-v8 held-out folds, so it can reject
an architecture but cannot promote it to confirmatory evidence.  Thresholds
inside the model are chosen only from grouped out-of-fold training predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_multimodal import (  # noqa: E402
    ConflictGuardedBidirectionalClassifier,
    ObservableClassifier,
    predictions_from_probabilities,
)
from scripts.run_kinofail_realistic_multimodal_v1 import (  # noqa: E402
    _masks,
    _split_labels,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _paired_cluster_bootstrap(
    truth: np.ndarray,
    ours: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for target, left, right, group in zip(
        truth, ours, baseline, groups, strict=True
    ):
        by_group[str(group)].append(
            float(left == target) - float(right == target)
        )
    values = np.asarray(
        [np.mean(by_group[group]) for group in sorted(by_group)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [
            np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(draws)
        ]
    )
    return {
        "unit": "paired counterfactual physics group",
        "group_count": len(values),
        "estimate": float(values.mean()),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "draws": draws,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eval/kinofail_realistic_conflict_guarded_v2_development.json",
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = _json(config_path)
    source_path = Path(__file__).resolve()
    model_source_path = ROOT / "kino_vla/eval/realistic_multimodal.py"
    if config["screen_script_sha256"] != _sha(source_path):
        raise RuntimeError("screen script changed after development freeze")
    if config["model_source_sha256"] != _sha(model_source_path):
        raise RuntimeError("model source changed after development freeze")

    snapshot_dir = (ROOT / config["snapshot_dir"]).resolve()
    feature_dir = (ROOT / config["feature_dir"]).resolve()
    records_path = snapshot_dir / "snapshot_records.jsonl"
    features_path = feature_dir / "features.npz"
    feature_manifest_path = feature_dir / "feature_manifest.json"
    registry_path = (ROOT / config["scene_registry"]).resolve()
    records = _jsonl(records_path)
    archive = np.load(features_path, allow_pickle=False)
    if archive["sample_ids"].astype(str).tolist() != [
        str(row["sample_id"]) for row in records
    ]:
        raise RuntimeError("feature rows and metadata are not aligned")
    visual = archive["visual"]
    proprio = archive["proprio"]
    labels = np.asarray(
        [str(row["attribution_category"]) for row in records]
    )
    episodes = np.asarray([str(row["physical_episode_id"]) for row in records])
    groups = np.asarray([str(row["counterfactual_group_id"]) for row in records])
    registry = _json(registry_path)
    scene_split, material_split = _split_labels(records, registry)

    output_root = (ROOT / config["output_root"]).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"one-shot development output already exists: {output_root}"
        )
    output_root.mkdir(parents=True)
    axes: dict[str, Any] = {}
    for axis in config["heldout_axes"]:
        train, main_test, _ = _masks(
            axis, records, scene_split, material_split
        )
        indices = np.flatnonzero(main_test)
        seed = int(config["seed"])
        baseline = ObservableClassifier.fit(
            visual[train],
            proprio[train],
            labels[train],
            episodes[train],
            feature_key="proprio",
            seed=seed,
        )
        ours = ConflictGuardedBidirectionalClassifier.fit(
            visual[train],
            proprio[train],
            labels[train],
            episodes[train],
            groups[train],
            seed=seed,
            minimum_override_precision=float(
                config["router"]["minimum_override_precision"]
            ),
            minimum_override_groups=int(
                config["router"]["minimum_override_groups"]
            ),
        )
        base_probability = baseline.predict_proba(
            visual[main_test], proprio[main_test]
        )
        ours_probability = ours.predict_proba(
            visual[main_test], proprio[main_test]
        )
        base_prediction = predictions_from_probabilities(
            base_probability, baseline.classes
        )
        ours_prediction = predictions_from_probabilities(
            ours_probability, ours.classes
        )
        truth = labels[main_test]
        delta = _paired_cluster_bootstrap(
            truth,
            ours_prediction,
            base_prediction,
            groups[main_test],
            draws=int(config["bootstrap"]["draws"]),
            seed=int(config["bootstrap"]["seed"]) + len(axes),
        )
        axis_dir = output_root / axis
        axis_dir.mkdir()
        with (axis_dir / "proprio_only.pkl").open("wb") as handle:
            pickle.dump(baseline, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (axis_dir / "conflict_guarded_v2.pkl").open("wb") as handle:
            pickle.dump(ours, handle, protocol=pickle.HIGHEST_PROTOCOL)
        rows = []
        for local, index in enumerate(indices):
            rows.append(
                {
                    "sample_id": records[int(index)]["sample_id"],
                    "counterfactual_group_id": groups[int(index)],
                    "truth": truth[local],
                    "proprio_only": str(base_prediction[local]),
                    "conflict_guarded_v2": str(ours_prediction[local]),
                }
            )
        prediction_path = axis_dir / "predictions.jsonl"
        prediction_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        axes[axis] = {
            "test_sample_count": len(indices),
            "test_group_count": len(set(groups[main_test])),
            "proprio_only_balanced_accuracy": float(
                balanced_accuracy_score(truth, base_prediction)
            ),
            "conflict_guarded_v2_balanced_accuracy": float(
                balanced_accuracy_score(truth, ours_prediction)
            ),
            "ours_minus_proprio": delta,
            "fit_audit": ours.fit_audit,
            "artifacts": {
                "proprio_checkpoint_sha256": _sha(
                    axis_dir / "proprio_only.pkl"
                ),
                "ours_checkpoint_sha256": _sha(
                    axis_dir / "conflict_guarded_v2.pkl"
                ),
                "predictions_sha256": _sha(prediction_path),
            },
        }

    report = {
        "schema_version": "kinofail.realistic-conflict-guarded-v2-screen.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_complete",
        "passed": all(
            row["ours_minus_proprio"]["estimate"] >= 0.0
            for row in axes.values()
        ),
        "counts_as_a0_a7_evidence": False,
        "existing_test_was_already_observed": True,
        "architecture_may_be_rejected_but_not_confirmed_here": True,
        "axes": axes,
        "input_artifacts": {
            "config_sha256": _sha(config_path),
            "snapshot_records_sha256": _sha(records_path),
            "features_sha256": _sha(features_path),
            "feature_manifest_sha256": _sha(feature_manifest_path),
            "scene_registry_sha256": _sha(registry_path),
            "screen_script_sha256": _sha(source_path),
            "model_source_sha256": _sha(model_source_path),
        },
        "claim_boundary": (
            "The architecture and router thresholds never use held-out outcomes, "
            "but this corpus was already inspected under earlier methods. Positive "
            "results require an unchanged-model untouched replication extension."
        ),
    }
    report_path = output_root / "screen_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({axis: {
        "proprio": row["proprio_only_balanced_accuracy"],
        "ours": row["conflict_guarded_v2_balanced_accuracy"],
        "delta": row["ours_minus_proprio"],
    } for axis, row in axes.items()}, indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
