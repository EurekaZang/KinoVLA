#!/usr/bin/env python3
"""Freeze the final O10 delay after a short-episode availability audit."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base_path = ROOT / "configs/eval/kinofail_realistic_snapshot_formal_v2.json"
    output = ROOT / "configs/eval/kinofail_realistic_snapshot_formal_v3.json"
    freeze_path = ROOT / "outputs/eval/realistic_a0_a7_v4/snapshot_o10_availability_amendment_freeze.json"
    if output.exists() or freeze_path.exists():
        raise FileExistsError("refusing to overwrite the O10 availability amendment")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    amended = deepcopy(base)
    amended["protocol_id"] = "realistic_snapshot_formal_scale_v7_o10_available_v3"
    amended["status"] = "frozen_after_half_scale_sensor_qa_before_formal_snapshot_extraction"
    amended["temporal_alignment"]["decision_delay_s_by_operator"]["O10_effort_decay"] = 0.7
    prior = amended.pop("amendment")
    amended["amendments"] = [
        prior,
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "base_protocol": str(base_path.relative_to(ROOT)),
            "base_protocol_sha256": _sha(base_path),
            "changed": "O10_effort_decay decision delay 0.8 s -> 0.7 s",
            "reason": "One short nominal episode ended before the 0.8 s decision frame; 0.7 s is the latest tested delay available for every observed O10 pair.",
            "sensor_qa_basis": {
                "formal_features_or_predictions_existed": False,
                "complete_o10_pairs_observed": 8,
                "candidate_results": {
                    "0.6_s": {"valid_pairs": 8, "minimum_mean_abs_proprio_delta": 0.3976878821849823, "median": 0.4587256461381912},
                    "0.7_s": {"valid_pairs": 8, "minimum_mean_abs_proprio_delta": 0.6518968343734741, "median": 1.0523911118507385},
                    "0.8_s": {"valid_pairs": 7, "invalid_pair": "cf_4a9e1b5338dfc930705e", "reason": "nominal RGB sequence ended at 2.22 s before the 2.34 s decision target"},
                },
                "selection_rule": "Use the latest tested delay available for all observed pairs; among available candidates it retains the stronger observable O10 consequence.",
                "labels_or_model_accuracy_used": False,
            },
        },
    ]
    output.write_text(json.dumps(amended, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    launcher = ROOT / "outputs/kinofail_realistic/corpus_scale_v7/launcher_audits/full_collection.json"
    attempts = len(json.loads(launcher.read_text(encoding="utf-8")).get("attempts", []))
    formal = [
        ROOT / "outputs/eval/realistic_a0_a7_v4/snapshots/snapshot_records.jsonl",
        ROOT / "outputs/eval/realistic_a0_a7_v4/features/features.npz",
        ROOT / "outputs/eval/realistic_a0_a7_v4/predictions.jsonl",
    ]
    freeze = {
        "schema_version": "kinofail.realistic-snapshot-availability-amendment-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "runtime_attempts_existing_at_freeze": attempts,
        "formal_artifacts_absent": {str(path.relative_to(ROOT)): not path.exists() for path in formal},
        "source_sha256": {
            str(base_path.relative_to(ROOT)): _sha(base_path),
            "kino_vla/data/realistic_snapshots.py": _sha(ROOT / "kino_vla/data/realistic_snapshots.py"),
        },
        "amended_protocol": str(output.relative_to(ROOT)),
        "amended_protocol_sha256": _sha(output),
        "labels_or_model_accuracy_used": False,
    }
    if not all(freeze["formal_artifacts_absent"].values()):
        output.unlink()
        raise RuntimeError("formal artifacts already exist; availability amendment is too late")
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(output), "sha256": _sha(output), "freeze": str(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
