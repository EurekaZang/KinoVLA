#!/usr/bin/env python3
"""Decision-only screen of the frozen structured-v2 gate on realistic scale-v8.

This is deliberately development evidence, not the C4 endpoint: it reconstructs
the old observable tracking-error channel from the issued command and raw Go2
odometry, applies the byte-frozen old gate, and reports which test snapshots
would be released.  No action consequence is inferred from this screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.material_support import (
    dominant_chromatic_rgb,
    material_support_distance,
)
from kino_vla.eval.structured_gate import gate_acts


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _nearest_command(
    telemetry: list[dict[str, Any]],
    timestamp_s: float,
) -> np.ndarray:
    row = min(
        telemetry,
        key=lambda value: abs(float(value["timestamp_s"]) - timestamp_s),
    )
    if abs(float(row["timestamp_s"]) - timestamp_s) > 0.011:
        raise ValueError("issued-command timestamp is not aligned to proprioception")
    command = np.asarray(row["command"], dtype=np.float64)
    if command.shape != (3,):
        raise ValueError("issued command is not a three-vector")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        type=Path,
        default=ROOT / "configs/eval/c4_structured_gate_v2.yaml",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/corpus_scale_v8_replication_v1",
    )
    parser.add_argument(
        "--overlay-corpus-root",
        type=Path,
        action="append",
        default=[
            ROOT
            / "outputs/kinofail_realistic/corpus_scale_v8_temporal_repair_v4"
        ],
        help=(
            "Additional snapshot source root. The source manifest hash selects the exact "
            "base or repaired episode; order does not affect the decision."
        ),
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/snapshots",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/predictions.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/eval/realistic_a0_a7_v6/a5_c4_transfer_screen.json",
    )
    args = parser.parse_args()

    gate_path = args.gate.resolve()
    schedule_path = args.schedule.resolve()
    corpus_root = args.corpus_root.resolve()
    corpus_roots = [corpus_root, *(path.resolve() for path in args.overlay_corpus_root)]
    snapshot_dir = args.snapshot_dir.resolve()
    predictions_path = args.predictions.resolve()
    frozen = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    gate = frozen["gate"]
    material_model = frozen["material_model"]

    schedule = {
        str(row["episode_id"]): row for row in _jsonl(schedule_path)
    }
    records = {
        str(row["sample_id"]): row
        for row in _jsonl(snapshot_dir / "snapshot_records.jsonl")
        if row["appearance_intervention_id"] == "primary"
    }
    grouped_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(predictions_path):
        if (
            row["method"] == "structured_bidirectional"
            and row["axis"] == "scene_and_material"
            and row["condition"] == "anomaly"
            and row["headline_primary_view"] is True
        ):
            grouped_predictions[str(row["sample_id"])].append(row)

    telemetry_cache: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    with np.load(snapshot_dir / "snapshots.npz", allow_pickle=False) as archive:
        for sample_id, seed_rows in sorted(grouped_predictions.items()):
            if len(seed_rows) != 5:
                raise ValueError(f"{sample_id} does not have five seed predictions")
            record = records[sample_id]
            episode_id = str(record["physical_episode_id"])
            scheduled = schedule[episode_id]
            if scheduled["split"] != "test":
                raise ValueError("scene-and-material heldout prediction is not test split")

            classes = [str(value) for value in seed_rows[0]["classes"]]
            probabilities = np.mean(
                np.asarray(
                    [row["probabilities"] for row in seed_rows],
                    dtype=np.float64,
                ),
                axis=0,
            )
            predicted = classes[int(np.argmax(probabilities))]
            # Exact gate short-circuit: only this frozen stratum can ever reach
            # the support/evidence tests. Skipping RGB/telemetry work for every
            # other attribution does not alter one decision bit.
            if predicted == "compliant_terrain":
                rgb = np.asarray(
                    archive[f"{sample_id}__rgb"],
                    dtype=np.float64,
                )
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                material_rgb_value = dominant_chromatic_rgb(rgb)
                material_distance_value = material_support_distance(
                    material_rgb_value,
                    material_model,
                )
                proprio = np.asarray(
                    archive[f"{sample_id}__proprio"],
                    dtype=np.float64,
                )
                timestamps = np.asarray(
                    archive[f"{sample_id}__proprio_timestamp_s"],
                    dtype=np.float64,
                )
                if episode_id not in telemetry_cache:
                    relative_manifest = Path(
                        str(scheduled["required_outputs"]["episode_manifest"])
                    )
                    candidates = [
                        root / relative_manifest for root in corpus_roots
                        if (root / relative_manifest).is_file()
                        and _sha(root / relative_manifest)
                        == str(record["source_manifest_sha256"])
                    ]
                    if len(candidates) != 1:
                        raise ValueError(
                            f"{sample_id} does not resolve to one hash-matched "
                            f"source manifest: {candidates}"
                        )
                    telemetry_path = candidates[0].parent / "privileged.jsonl"
                    telemetry_cache[episode_id] = _jsonl(telemetry_path)
                telemetry = telemetry_cache[episode_id]
                tracking_errors = []
                for index, timestamp in enumerate(timestamps):
                    command = _nearest_command(telemetry, float(timestamp))
                    measured_velocity = proprio[index, 12:14]
                    tracking_errors.append(
                        float(
                            np.linalg.norm(command[:2] - measured_velocity)
                        )
                    )
                tracking_error_peak_value = max(tracking_errors)
                material_rgb = material_rgb_value.tolist()
                material_distance = material_distance_value
                tracking_error_peak = tracking_error_peak_value
            else:
                material_rgb = None
                material_distance = None
                tracking_error_peak = None
            point = {
                "agent_label": (
                    "high_step" if predicted == "compliant_terrain" else "continue"
                ),
                "attribution": predicted,
                "material_distance": material_distance,
                "tracking_error_peak": tracking_error_peak,
            }
            release = gate_acts(point, gate)
            decisions.append(
                {
                    "sample_id": sample_id,
                    "counterfactual_group_id": record[
                        "counterfactual_group_id"
                    ],
                    "scene_cluster": record["scene_family"],
                    "domain": record["domain"],
                    "truth_attribution": record["attribution_category"],
                    "target_operator": record["target_operator"],
                    "predicted_attribution": predicted,
                    "ensemble_confidence": float(np.max(probabilities)),
                    "material_rgb": material_rgb,
                    "material_distance": material_distance,
                    "tracking_error_peak": tracking_error_peak,
                    "release": release,
                    "released_attribution_correct": (
                        release
                        and predicted == str(record["attribution_category"])
                    ),
                    "decision_input_fields": [
                        "ensemble model probabilities",
                        "body-fixed RGB",
                        "issued command",
                        "raw Go2 odometry velocity",
                    ],
                }
            )

    released = [row for row in decisions if row["release"]]
    compliant = [
        row
        for row in decisions
        if row["truth_attribution"] == "compliant_terrain"
    ]
    report = {
        "schema_version": "kinofail.realistic-c4-transfer-screen.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_complete_not_headline",
        "dataset_scope": "kinofail_realistic_scale_v8_test",
        "action_consequence_measured": False,
        "c4_endpoint_available": False,
        "method": (
            "byte-frozen structured-v2 thresholds with observable scale-v8 adapter "
            "and five-seed probability mean"
        ),
        "counts": {
            "test_anomaly_cases": len(decisions),
            "released_cases": len(released),
            "true_compliant_terrain_cases": len(compliant),
        },
        "decision_metrics": {
            "release_coverage": len(released) / len(decisions),
            "released_attribution_precision": (
                float(
                    np.mean(
                        [
                            row["released_attribution_correct"]
                            for row in released
                        ]
                    )
                )
                if released
                else 0.0
            ),
            "compliant_terrain_release_recall": (
                sum(row["release"] for row in compliant) / len(compliant)
                if compliant
                else 0.0
            ),
            "released_scene_clusters": sorted(
                {str(row["scene_cluster"]) for row in released}
            ),
        },
        "frozen_gate": str(gate_path.relative_to(ROOT)),
        "frozen_gate_sha256": _sha(gate_path),
        "inputs": {
            "schedule_sha256": _sha(schedule_path),
            "snapshot_records_sha256": _sha(
                snapshot_dir / "snapshot_records.jsonl"
            ),
            "snapshots_sha256": _sha(snapshot_dir / "snapshots.npz"),
            "predictions_sha256": _sha(predictions_path),
        },
        "adapter_boundary": (
            "The command and odometry channels are controller-observable. Ground truth, "
            "operator id, scene id, material id, and test cost are used only after the "
            "decision for audit metrics. The ensemble/adapter was specified after scale-v8 "
            "model outcomes, so this screen can guide a newly frozen independent C4 corpus "
            "but cannot itself close C4."
        ),
        "decisions": decisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "status": report["status"],
                "counts": report["counts"],
                "decision_metrics": report["decision_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
