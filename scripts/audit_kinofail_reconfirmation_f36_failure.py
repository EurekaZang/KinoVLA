#!/usr/bin/env python3
"""Record the result-blind F36 loader failure before the F37 amendment."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
F36 = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f36"
SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f36/seal_manifest.json"
STATE = ROOT / "outputs/kinofail_after_a4_pilot_to_a0_a7_v2/state.json"
LOG = ROOT / "outputs/kinofail_after_a4_pilot_to_a0_a7_v2/logs/04_finalize_f36_once.log"
OUTPUT = ROOT / "outputs/kinofail_reconfirmation_f36_failure_audit_v1/audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    required = (
        SEAL,
        STATE,
        LOG,
        F36 / "conflict/c2_base/feature_manifest.json",
        F36 / "conflict/c2_base/geometry_manifest.json",
        F36 / "conflict/c2_base/records.jsonl",
        F36 / "conflict/c2_base/features.npz",
        F36 / "conflict/c2_base/geometry.npz",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    state = json.loads(STATE.read_text())
    feature_manifest = json.loads((F36 / "conflict/c2_base/feature_manifest.json").read_text())
    forbidden = (
        F36 / "blind_predictions",
        F36 / "blind_bundle/truth_key.jsonl",
        F36 / "confirmatory_report.json",
        F36 / "finalization_audit.json",
    )
    checks = {
        "terminal_failure_recorded": state.get("state") == "terminal_failure",
        "failed_at_finalize_f36": state.get("stage") == "04_finalize_f36_once",
        "base_features_completed_model_blind": feature_manifest.get("passed") is True,
        "no_blind_predictions_exist": not forbidden[0].exists(),
        "no_truth_key_unsealed": not forbidden[1].exists(),
        "no_confirmatory_report_exists": not forbidden[2].exists(),
        "no_finalization_audit_exists": not forbidden[3].exists(),
        "loader_keyerror_recorded": "KeyError: 'cf_75f2161542aef1e86f7f_anomaly'" in LOG.read_text(),
    }
    report = {
        "schema_version": "kinofail.reconfirmation-f36-failure-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "status": "result_blind_loader_failure_preserved",
        "checks": checks,
        "failure_class": "F30 union episode leaves are directory symlinks; Path.rglob did not descend into them",
        "scientific_attempt_started": False,
        "model_prediction_or_score_read": False,
        "partial_base_features_count_as_confirmation": False,
        "f36_root_preserved": str(F36.relative_to(ROOT)),
        "sha256": {
            "f36_seal": sha256(SEAL),
            "state": sha256(STATE),
            "log": sha256(LOG),
            "base_feature_manifest": sha256(F36 / "conflict/c2_base/feature_manifest.json"),
            "base_geometry_manifest": sha256(F36 / "conflict/c2_base/geometry_manifest.json"),
            "base_records": sha256(F36 / "conflict/c2_base/records.jsonl"),
            "base_features": sha256(F36 / "conflict/c2_base/features.npz"),
            "base_geometry": sha256(F36 / "conflict/c2_base/geometry.npz"),
            "audit_script": sha256(Path(__file__).resolve()),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
