#!/usr/bin/env python3
"""Assemble, predict, and score the sealed confirmatory extension exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
REGISTRY = ROOT / "outputs/kinofail_confirmatory_v1/scene_registry.json"
SCHEDULE_ROOT = (
    ROOT
    / "outputs/kinofail_confirmatory_v1/"
    "schedules_f2_scene_source_amendment"
)
EVAL_ROOT = ROOT / "outputs/eval/unified_moe_v3_confirmatory_v1"
SHARD_ROOT = EVAL_ROOT / "shards"
CONFLICT_FEATURES = EVAL_ROOT / "conflict/features"
F0 = (
    ROOT
    / "outputs/freeze/unified_moe_v3_confirmatory_f0_v6/"
    "freeze_manifest.json"
)
F1 = (
    ROOT
    / "outputs/freeze/unified_moe_v3_confirmatory_f1/"
    "seal_manifest.json"
)
FRESHNESS = ROOT / "outputs/kinofail_confirmatory_v1/freshness_audit.json"
SCALE_LANES = (
    ROOT / "outputs/kinofail_confirmatory_v1/scale_orchestration"
)
PRUNE_RECEIPTS = (
    ROOT / "outputs/kinofail_confirmatory_v1/prune_receipts"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _scenes() -> list[str]:
    scenes = [str(row["scene_id"]) for row in _json(REGISTRY)["scenes"]]
    if len(scenes) != 30 or len(set(scenes)) != 30:
        raise RuntimeError("confirmatory registry must contain 30 scenes")
    return scenes


def _scale_complete(scenes: list[str]) -> bool:
    expected = {"low": scenes[:15], "high": scenes[15:]}
    for lane_id, lane_scenes in expected.items():
        path = SCALE_LANES / f"{lane_id}.json"
        if not path.is_file():
            return False
        audit = _json(path)
        if (
            audit.get("scenes") != lane_scenes
            or len(audit.get("attempts", [])) != len(lane_scenes)
            or any(
                row.get("status")
                not in {
                    "complete_sealed_pruned",
                    "already_sealed_and_pruned",
                }
                for row in audit["attempts"]
            )
        ):
            return False
    return all(
        (PRUNE_RECEIPTS / scene_id / "completed.json").is_file()
        and _json(PRUNE_RECEIPTS / scene_id / "completed.json").get("state")
        == "completed"
        for scene_id in scenes
    )


def _active_collectors() -> int:
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        count += "isaac_collect_kinofail_confirmatory_" in command
    return count


def _wait(scenes: list[str], poll_seconds: int) -> None:
    last_report = 0.0
    while True:
        complete = _scale_complete(scenes)
        active = _active_collectors()
        if complete and active == 0:
            return
        now = time.monotonic()
        if now - last_report >= 600:
            print(
                json.dumps(
                    {
                        "stage": "waiting_for_scale",
                        "scale_complete": complete,
                        "active_collectors": active,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_report = now
        time.sleep(poll_seconds)


def _run(command: list[str], log_name: str) -> None:
    log_root = EVAL_ROOT / "finalization_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{log_name}.log"
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    with log_path.open("wb") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(
            f"finalization stage failed ({completed.returncode}): "
            f"{log_name}\n{tail}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError("poll interval is too small")
    final_audit_path = EVAL_ROOT / "finalization_audit.json"
    if final_audit_path.exists():
        raise FileExistsError(final_audit_path)
    scenes = _scenes()
    _wait(scenes, args.poll_seconds)

    inputs = EVAL_ROOT / "evaluation_inputs"
    input_command = [
        str(PYTHON),
        "scripts/build_kinofail_confirmatory_evaluation_bundle_inputs_v1.py",
        "--scale-schedule",
        str(SCHEDULE_ROOT / "scale_schedule.jsonl"),
    ]
    for scene_id in scenes:
        input_command.extend(
            [
                "--scale-snapshot-dir",
                str(SHARD_ROOT / scene_id / "scale/snapshots"),
                "--scale-unified-dir",
                str(SHARD_ROOT / scene_id / "scale/unified_features"),
            ]
        )
    input_command.extend(
        [
            "--conflict-schedule",
            str(SCHEDULE_ROOT / "conflict_schedule.jsonl"),
            "--conflict-feature-dir",
            str(CONFLICT_FEATURES),
            "--output",
            str(inputs),
        ]
    )
    _run(input_command, "01_evaluation_inputs")
    inputs_audit = _json(inputs / "audit.json")
    if inputs_audit.get("passed") is not True:
        raise RuntimeError("evaluation inputs did not pass")

    blind_bundle = EVAL_ROOT / "blind_bundle"
    _run(
        [
            str(PYTHON),
            "scripts/assemble_kinofail_confirmatory_blind_bundle_v1.py",
            "--f1-manifest",
            str(F1),
            "--feature-shard",
            str(inputs / "scale_features.npz"),
            "--feature-shard",
            str(inputs / "conflict_features.npz"),
            "--truth-ledger",
            str(inputs / "scale_truth.jsonl"),
            "--truth-ledger",
            str(inputs / "conflict_truth.jsonl"),
            "--out",
            str(blind_bundle),
        ],
        "02_blind_bundle",
    )
    bundle_manifest = _json(blind_bundle / "bundle_manifest.json")
    if (
        bundle_manifest.get("status")
        != "observable_features_and_separate_truth_key_sealed"
    ):
        raise RuntimeError("blind bundle did not seal")

    prediction_dir = EVAL_ROOT / "blind_predictions"
    _run(
        [
            str(PYTHON),
            "scripts/predict_kinofail_unified_moe_v3_blind.py",
            "--f0-manifest",
            str(F0),
            "--f1-manifest",
            str(F1),
            "--freshness-audit",
            str(FRESHNESS),
            "--features",
            str(blind_bundle / "blind_features.npz"),
            "--output-dir",
            str(prediction_dir),
        ],
        "03_blind_prediction",
    )
    prediction_manifest = _json(
        prediction_dir / "prediction_manifest.json"
    )
    if (
        prediction_manifest.get("status") != "blind_predictions_sealed"
        or prediction_manifest.get("labels_or_outcomes_read") is not False
        or prediction_manifest.get("fit_or_refit_called") is not False
    ):
        raise RuntimeError("blind prediction contract failed")

    protocol = EVAL_ROOT / "scoring_protocol.json"
    _run(
        [
            str(PYTHON),
            "scripts/build_kinofail_confirmatory_scoring_protocol_v1.py",
            "--blind-predictions",
            str(prediction_dir / "blind_predictions.jsonl"),
            "--truth-key",
            str(blind_bundle / "truth_key.jsonl"),
            "--f0-manifest",
            str(F0),
            "--f1-manifest",
            str(F1),
            "--feature-manifest",
            str(blind_bundle / "bundle_manifest.json"),
            "--out",
            str(protocol),
        ],
        "04_scoring_protocol",
    )
    if _json(protocol).get("status") != (
        "sealed_after_blind_prediction_before_scoring"
    ):
        raise RuntimeError("scoring protocol did not seal")

    report = EVAL_ROOT / "confirmatory_report.json"
    _run(
        [
            str(PYTHON),
            "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
            "--protocol",
            str(protocol),
            "--blind-predictions",
            str(prediction_dir / "blind_predictions.jsonl"),
            "--truth-key",
            str(blind_bundle / "truth_key.jsonl"),
            "--out",
            str(report),
        ],
        "05_score_once",
    )
    scored = _json(report)
    if scored.get("confirmatory_protocol_valid") is not True:
        raise RuntimeError("one-shot confirmatory protocol was invalid")

    audit = {
        "schema_version": "kinofail.confirmatory-finalization.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "score_once": True,
        "report_regardless_of_outcome": True,
        "passed_all_preregistered_gates": scored.get(
            "passed_all_preregistered_gates"
        ),
        "source_sha256": {
            "f0": _sha256(F0),
            "f1": _sha256(F1),
            "freshness": _sha256(FRESHNESS),
            "evaluation_inputs": _sha256(inputs / "audit.json"),
            "blind_bundle": _sha256(blind_bundle / "bundle_manifest.json"),
            "blind_predictions": _sha256(
                prediction_dir / "prediction_manifest.json"
            ),
            "scoring_protocol": _sha256(protocol),
            "confirmatory_report": _sha256(report),
        },
        "counts": {
            "evaluation_inputs": inputs_audit.get("counts"),
            "blind_sample_count": bundle_manifest.get("sample_count_valid"),
            "prediction_rows": prediction_manifest.get(
                "artifact", {}
            ).get("rows"),
        },
    }
    final_audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
