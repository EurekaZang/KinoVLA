#!/usr/bin/env python3
"""Freeze the O10 event-to-decision timing correction before formal features exist."""

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
    base_path = ROOT / "configs/eval/kinofail_realistic_snapshot_formal_v1.json"
    output = ROOT / "configs/eval/kinofail_realistic_snapshot_formal_v2.json"
    freeze_path = ROOT / "outputs/eval/realistic_a0_a7_v4/snapshot_o10_timing_amendment_freeze.json"
    if output.exists() or freeze_path.exists():
        raise FileExistsError("refusing to overwrite the O10 timing amendment")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    amended = deepcopy(base)
    amended["protocol_id"] = "realistic_snapshot_formal_scale_v7_o10_timing_v2"
    amended["status"] = "frozen_after_sensor_qa_before_formal_snapshot_extraction"
    amended["temporal_alignment"]["decision_delay_s_by_operator"]["O10_effort_decay"] = 0.8
    amended["amendment"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "base_protocol": str(base_path.relative_to(ROOT)),
        "base_protocol_sha256": _sha(base_path),
        "changed": "O10_effort_decay decision delay 0.0 s -> 0.8 s",
        "unchanged": [
            "all collected episodes",
            "event adapter",
            "RGB offsets",
            "proprioception window",
            "all other operator delays",
            "model architecture and heldout splits",
        ],
        "sensor_qa_basis": {
            "formal_features_or_predictions_existed": False,
            "complete_o10_pairs_observed": 5,
            "metric": "mean absolute anomaly-minus-nominal proprioception difference in the frozen 21-sample decision window",
            "candidate_results": {
                "0.0_s": {"median": 0.0, "minimum": 0.0},
                "0.4_s": {"median": 0.19378569722175598, "minimum": 0.003966059070080519},
                "0.8_s": {"median": 1.486276626586914, "minimum": 1.3888788223266602},
                "1.2_s": {"valid": False, "reason": "requested RGB timestamps do not resolve to distinct 10 Hz frames"},
            },
            "selection_rule": "Use the shortest tested delay that exposes a non-negligible O10 sensor consequence in every observed pair while retaining five distinct RGB frames.",
            "labels_or_model_accuracy_used": False,
        },
    }
    output.write_text(json.dumps(amended, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    launcher = ROOT / "outputs/kinofail_realistic/corpus_scale_v7/launcher_audits/full_collection.json"
    attempts = len(json.loads(launcher.read_text(encoding="utf-8")).get("attempts", []))
    freeze = {
        "schema_version": "kinofail.realistic-snapshot-timing-amendment-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "runtime_attempts_existing_at_freeze": attempts,
        "formal_artifacts_absent": {
            str(path.relative_to(ROOT)): not path.exists()
            for path in (
                ROOT / "outputs/eval/realistic_a0_a7_v4/snapshots/snapshot_records.jsonl",
                ROOT / "outputs/eval/realistic_a0_a7_v4/features/features.npz",
                ROOT / "outputs/eval/realistic_a0_a7_v4/predictions.jsonl",
            )
        },
        "source_sha256": {
            str(base_path.relative_to(ROOT)): _sha(base_path),
            "kino_vla/data/realistic_snapshots.py": _sha(ROOT / "kino_vla/data/realistic_snapshots.py"),
        },
        "amended_protocol": str(output.relative_to(ROOT)),
        "amended_protocol_sha256": _sha(output),
        "reason": "Correct a conclusion-damaging O10 temporal-alignment bug detected by observable-input QA before feature extraction or model evaluation.",
    }
    if not all(freeze["formal_artifacts_absent"].values()):
        output.unlink()
        raise RuntimeError("formal artifacts already exist; timing amendment is too late")
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(output), "sha256": _sha(output), "freeze": str(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
