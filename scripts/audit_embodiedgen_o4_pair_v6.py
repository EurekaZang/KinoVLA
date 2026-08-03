#!/usr/bin/env python3
"""Adjudicate an immutable O4 pair with the phase-aware v6 visual contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from kino_vla.eval.o4_phase_aware import adjudicate_phase_aware_visuals


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=("development", "formal"), required=True)
    args = parser.parse_args()
    manifest_path = args.pair_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    frame_sets = {}
    artifact_checks = {}
    for take_name in ("nominal", "o4_adhesion"):
        take = manifest["takes"][take_name]
        frames_path = root / take["frames_manifest"]
        frames = _jsonl(frames_path)
        frame_sets[take_name] = frames
        artifact_checks[f"{take_name}_frames_manifest_hash"] = (
            _sha256(frames_path) == take["frames_manifest_sha256"]
        )
        artifact_checks[f"{take_name}_frame_count"] = len(frames) == int(take["rgb_frames"])
        artifact_checks[f"{take_name}_all_frame_hashes"] = all(
            (root / frame["path"]).is_file()
            and _sha256(root / frame["path"]) == frame["sha256"]
            for frame in frames
        )
    event_steps = [
        int(event["step"])
        for event in manifest["takes"]["o4_adhesion"]["event_trace"]
        if event["event"] == "attached"
    ]
    if not event_steps:
        raise ValueError("O4 pair has no attachment event")
    first_attachment = min(event_steps)
    anomaly = manifest["takes"]["o4_adhesion"]
    visual = adjudicate_phase_aware_visuals(
        frame_sets["nominal"],
        frame_sets["o4_adhesion"],
        first_attachment_step=first_attachment,
        anomaly_fell=bool(anomaly["fell"]),
        first_fall_step=anomaly["first_fall_step"],
    )
    inherited_checks = {
        name: bool(value)
        for name, value in manifest["checks"].items()
        if name != "both_sequences_visually_valid"
    }
    checks = {
        "input_pair_result_is_boolean": isinstance(manifest.get("passed"), bool),
        "input_pair_role_known": manifest.get("protocol_role") in {"calibration", "formal"},
        "all_artifact_hashes_verified": all(artifact_checks.values()),
        "all_nonvisual_pair_checks_passed": all(inherited_checks.values()),
        "phase_aware_visual_contract_passed": visual["passed"],
    }
    result = {
        "schema_version": "kinofail.embodiedgen-o4-pair-adjudication.v6",
        "created_utc": datetime.now(UTC).isoformat(),
        "adjudication_role": args.role,
        "input_pair_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "scene_id": manifest["scene_id"],
        "seed": manifest["seed"],
        "input_pair_passed": manifest["passed"],
        "input_visual_check_passed": manifest["checks"]["both_sequences_visually_valid"],
        "artifact_checks": artifact_checks,
        "inherited_nonvisual_checks": inherited_checks,
        "phase_aware_visuals": visual,
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation": (
            "formal only when this adjudicator and its thresholds were frozen before collection"
            if args.role == "formal"
            else "development replay only; never relabels the sealed input batch"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "scene_id": result["scene_id"],
                "input_pair_passed": result["input_pair_passed"],
                "phase_aware_visual_passed": visual["passed"],
                "passed": result["passed"],
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
