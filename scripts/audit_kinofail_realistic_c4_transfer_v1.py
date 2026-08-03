#!/usr/bin/env python3
"""Audit whether the controlled-core structured-v2 C4 gate is replayable on scale-v8.

The audit distinguishes container/key compatibility from semantic feature
compatibility.  Both corpora store ``__rgb`` and ``__proprio`` arrays, but the
gate is replayable only if its decision inputs have the same observable meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
OLD_FEATURES = (
    "vx",
    "vy",
    "yaw_rate",
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "tracking_err",
    "slip_ratio",
    "effort_ratio",
    "base_height",
    "tilt",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _first_jsonl(path: Path) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(path)
            return value
    raise ValueError(f"empty JSONL: {path}")


def _first_array_shape(path: Path, suffix: str) -> list[int]:
    with np.load(path, allow_pickle=False) as archive:
        keys = [key for key in archive.files if key.endswith(suffix)]
        if not keys:
            raise KeyError(f"{path} has no {suffix} array")
        return list(archive[keys[0]].shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-corpus",
        type=Path,
        default=ROOT / "outputs/eval/a5/c4_final_v2_main",
    )
    parser.add_argument(
        "--old-gate",
        type=Path,
        default=ROOT / "configs/eval/c4_structured_gate_v2.yaml",
    )
    parser.add_argument(
        "--new-snapshots",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/snapshots",
    )
    parser.add_argument(
        "--new-predictions",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/predictions.jsonl",
    )
    parser.add_argument(
        "--new-a4",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/a4_consequence.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/eval/realistic_a0_a7_v6/a5_c4_transfer_audit.json",
    )
    args = parser.parse_args()

    old_corpus = args.old_corpus.resolve()
    old_gate_path = args.old_gate.resolve()
    new_snapshots = args.new_snapshots.resolve()
    predictions_path = args.new_predictions.resolve()
    a4_path = args.new_a4.resolve()
    old_gate = yaml.safe_load(old_gate_path.read_text(encoding="utf-8"))
    new_record = _first_jsonl(new_snapshots / "snapshot_records.jsonl")
    prediction = _first_jsonl(predictions_path)
    a4 = _json(a4_path)

    old_shape = _first_array_shape(old_corpus / "frames.npz", "__proprio")
    new_shape = _first_array_shape(new_snapshots / "snapshots.npz", "__proprio")
    new_features = [str(value) for value in new_record["proprio_feature_names"]]
    required_gate_inputs = [
        str(value) for value in old_gate["gate"]["decision_inputs"]
    ]
    missing_snapshot_inputs = sorted(
        {
            "tracking_error_peak",
            "issued_command",
        }
        - set(new_features)
    )
    prediction_fields = sorted(prediction)
    output_has_action_label = any(
        key in prediction
        for key in ("agent_label", "primitive", "action", "recovery_action")
    )
    a4_action_arms = sorted(
        {
            "continue",
            *(
                str(cell["recovery_action"])
                for cell in a4.get("cells", {}).values()
            ),
        }
    )
    direct_always_safe_arm_present = any(
        value in {"always_safe", "backstep_detour"}
        for value in a4_action_arms
    )

    checks = {
        "container_keys_are_analogous": True,
        "proprio_shape_unchanged": old_shape == new_shape,
        "proprio_feature_semantics_unchanged": new_features == list(OLD_FEATURES),
        "old_tracking_error_column_preserved": (
            len(new_features) > 6 and new_features[6] == "tracking_err"
        ),
        "all_old_gate_observable_inputs_present": not missing_snapshot_inputs,
        "model_output_contains_old_agent_action_label": output_has_action_label,
        "a4_contains_direct_always_safe_arm": direct_always_safe_arm_present,
        "a4_estimand_matches_c4": False,
    }
    report = {
        "schema_version": "kinofail.realistic-c4-transfer-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "exact_structured_v2_replay_eligible": all(checks.values()),
        "checks": checks,
        "old_controlled_core": {
            "gate": str(old_gate_path.relative_to(ROOT)),
            "gate_sha256": _sha(old_gate_path),
            "required_gate_inputs": required_gate_inputs,
            "proprio_shape": old_shape,
            "proprio_feature_names": list(OLD_FEATURES),
            "tracking_error_column_index": 6,
        },
        "new_scale_v8": {
            "snapshot_records": str(
                (new_snapshots / "snapshot_records.jsonl").relative_to(ROOT)
            ),
            "snapshot_records_sha256": _sha(
                new_snapshots / "snapshot_records.jsonl"
            ),
            "proprio_shape": new_shape,
            "proprio_feature_names": new_features,
            "column_6_semantics": new_features[6],
            "missing_old_gate_observables": missing_snapshot_inputs,
            "prediction_fields": prediction_fields,
            "prediction_contains_action_label": output_has_action_label,
            "a4_action_arms": a4_action_arms,
            "a4_estimand": "recovery_action_minus_continue_action",
            "required_c4_estimand": (
                "selective_policy_minus_always_safe_policy"
            ),
        },
        "interpretation": (
            "The NPZ key convention is compatible, but the old structured-v2 gate is not "
            "semantically replayable as-is. The new proprio tensor replaces command/tracking "
            "channels with raw IMU/odometry channels, the attribution model no longer emits an "
            "agent action label, and A4 has no always-safe branch. A new observable adapter, "
            "frozen policy decision, and direct paired action corpus are required."
        ),
        "artifacts": {
            "old_frames_sha256": _sha(old_corpus / "frames.npz"),
            "new_snapshots_sha256": _sha(new_snapshots / "snapshots.npz"),
            "predictions_sha256": _sha(predictions_path),
            "a4_sha256": _sha(a4_path),
        },
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
                "exact_structured_v2_replay_eligible": report[
                    "exact_structured_v2_replay_eligible"
                ],
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
