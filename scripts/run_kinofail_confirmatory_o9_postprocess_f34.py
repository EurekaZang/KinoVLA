#!/usr/bin/env python3
"""Build direct-event snapshots and frozen observable features for F33 O9."""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.build_kinofail_confirmatory_o9_pilot_f32 import ROOT, read_jsonl, sha256
from scripts.run_kinofail_t3_replenishment_f28 import atomic_json, read_json


OUTPUT = ROOT / "outputs/kinofail_confirmatory_o9_postprocess_f34"
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")


def validate_seal(path: Path) -> dict[str, Any]:
    seal = read_json(path)
    sidecar = path.with_name("seal_manifest.sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text().split()[0] != sha256(path)
        or seal.get("status") != "sealed_before_model_blind_feature_extraction"
        or seal.get("passed") is not True
        or seal.get("model_prediction_feature_label_outcome_or_score_read") is not False
        or int(seal.get("accepted_physical_pairs", 0)) < 750
    ):
        raise RuntimeError("invalid F34 seal")
    for key in ("f33_seal", "f33_audit", "accepted_schedule", "snapshot_protocol"):
        artifact = ROOT / str(seal[key])
        if not artifact.is_file() or sha256(artifact) != seal[f"{key}_sha256"]:
            raise RuntimeError(f"F34 sealed artifact drift: {artifact}")
    for row in seal["dependencies"]:
        artifact = Path(str(row["path"]))
        if not artifact.is_file() or sha256(artifact) != row["sha256"]:
            raise RuntimeError(f"F34 dependency drift: {artifact}")
    return seal


def run(command: list[str], log_path: Path) -> None:
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"F34 stage failed ({completed.returncode}): {log_path}")


def main() -> int:
    seal_path = OUTPUT / "seal_manifest.json"
    seal = validate_seal(seal_path)
    eval_root = ROOT / str(seal["eval_root"])
    if eval_root.exists():
        raise FileExistsError(f"refusing to overwrite F34 evaluation: {eval_root}")
    snapshot_dir = eval_root / "snapshots"
    visual_dir = eval_root / "visual_features"
    unified_dir = eval_root / "unified_features"
    protocol = ROOT / str(seal["snapshot_protocol"])
    logs = OUTPUT / "logs"
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts/build_kinofail_o9_direct_snapshots_f34.py"),
            "--protocol",
            str(protocol.relative_to(ROOT)),
        ],
        logs / "01_direct_snapshots.log",
    )
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts/extract_kinofail_realistic_features.py"),
            "--snapshot-dir",
            str(snapshot_dir),
            "--output-dir",
            str(visual_dir),
            "--batch-size",
            "64",
        ],
        logs / "02_visual_features.log",
    )
    run(
        [
            str(PYTHON),
            str(ROOT / "scripts/build_kinofail_unified_invariant_features_v1.py"),
            "--snapshot-dir",
            str(snapshot_dir),
            "--visual-feature-dir",
            str(visual_dir),
            "--output-dir",
            str(unified_dir),
        ],
        logs / "03_unified_features.log",
    )

    snapshot_audit_path = snapshot_dir / "extraction_audit.json"
    snapshot_records_path = snapshot_dir / "snapshot_records.jsonl"
    snapshot_arrays_path = snapshot_dir / "snapshots.npz"
    visual_manifest_path = visual_dir / "feature_manifest.json"
    visual_features_path = visual_dir / "features.npz"
    unified_manifest_path = unified_dir / "feature_manifest.json"
    unified_features_path = unified_dir / "features.npz"
    snapshot_audit = read_json(snapshot_audit_path)
    visual_manifest = read_json(visual_manifest_path)
    unified_manifest = read_json(unified_manifest_path)
    records = read_jsonl(snapshot_records_path)
    schedule_rows = read_jsonl(ROOT / str(seal["accepted_schedule"]))
    anomaly_by_pair = {
        str(row["counterfactual_group_id"]): row
        for row in schedule_rows
        if row["condition"] == "anomaly"
    }
    valid_ids = {str(row["counterfactual_group_id"]) for row in records}
    if len(records) != 6 * len(valid_ids):
        raise RuntimeError("F34 snapshots do not contain complete six-sample pairs")
    if not valid_ids <= set(anomaly_by_pair):
        raise RuntimeError("F34 emitted a pair outside the accepted F33 schedule")
    by_scene = Counter(str(anomaly_by_pair[pair_id]["scene_id"]) for pair_id in valid_ids)
    by_domain = Counter(str(anomaly_by_pair[pair_id]["domain"]) for pair_id in valid_ids)
    by_lambda = Counter(
        f"{float(anomaly_by_pair[pair_id]['parameter_interpolation']['source_lambda']):.8f}"
        for pair_id in valid_ids
    )
    valid_pairs = len(valid_ids)
    effective_scale = 9283 + valid_pairs
    remaining = 10560 - effective_scale
    attrition_rate = remaining / 10560
    gates = {
        "snapshot_audit_passed": snapshot_audit.get("passed") is True,
        "visual_manifest_complete": visual_manifest.get("status") == "complete",
        "unified_manifest_complete": unified_manifest.get("status") == "complete",
        "at_least_750_direct_event_pairs": valid_pairs >= 750,
        "effective_scale_attrition_strictly_below_five_percent": attrition_rate < 0.05,
        "each_scene_has_at_least_24_pairs": len(by_scene) == 30 and min(by_scene.values(), default=0) >= 24,
        "each_domain_has_at_least_250_pairs": len(by_domain) == 3 and min(by_domain.values(), default=0) >= 250,
        "each_parameter_point_has_at_least_45_pairs": len(by_lambda) == 16 and min(by_lambda.values(), default=0) >= 45,
        "all_records_are_o9": {str(row["target_operator"]) for row in records} == {"O9_high_centering"},
        "all_records_use_direct_event_protocol": all(
            float(row["event_time_s_from_anomaly_privileged_telemetry"]) >= 0.0 for row in records
        ),
    }
    audit = {
        "schema_version": "kinofail.confirmatory-o9-postprocess-f34-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "final",
        "passed": all(gates.values()),
        "model_prediction_feature_label_outcome_or_score_read": False,
        "event_alignment": "first cumulative telemetry frame passing strict direct O9 semantics; nominal reuses the exact time",
        "counts": {
            "f33_accepted_physical_pairs": int(seal["accepted_physical_pairs"]),
            "direct_event_snapshot_pairs": valid_pairs,
            "temporal_pair_exclusions": int(snapshot_audit["counts"]["temporally_infeasible_counterfactual_pairs"]),
            "snapshot_records": len(records),
            "effective_non_o9_scale_pairs": 9283,
            "effective_total_scale_pairs": effective_scale,
            "remaining_scale_attrition_pairs": remaining,
        },
        "effective_scale_attrition_rate": attrition_rate,
        "gates": gates,
        "valid_pair_ids": sorted(valid_ids),
        "counts_by_scene": dict(sorted(by_scene.items())),
        "counts_by_domain": dict(sorted(by_domain.items())),
        "counts_by_source_lambda": dict(sorted(by_lambda.items())),
        "source_sha256": {
            "seal": sha256(seal_path),
            "snapshot_records": sha256(snapshot_records_path),
            "snapshot_arrays": sha256(snapshot_arrays_path),
            "snapshot_audit": sha256(snapshot_audit_path),
            "visual_features": sha256(visual_features_path),
            "visual_manifest": sha256(visual_manifest_path),
            "unified_features": sha256(unified_features_path),
            "unified_manifest": sha256(unified_manifest_path),
        },
    }
    atomic_json(OUTPUT / "final_audit.json", audit)
    print(json.dumps({"passed": audit["passed"], "counts": audit["counts"], "gates": gates}, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
