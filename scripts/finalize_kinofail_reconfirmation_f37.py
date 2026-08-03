#!/usr/bin/env python3
"""Run the sealed F37 result-blind symlink-loader amendment exactly once."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts import finalize_kinofail_reconfirmation_f36 as base


ROOT = Path(__file__).resolve().parents[1]
F37_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f37/seal_manifest.json"
TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f37"


def validate_seal() -> dict[str, Any]:
    seal = base.read_json(F37_SEAL)
    sidecar = F37_SEAL.with_name("seal_manifest.sha256")
    source = seal.get("source_sha256", {})
    scripts = source.get("execution_scripts", {})
    relative_self = str(Path(__file__).resolve().relative_to(ROOT))
    if (
        not sidecar.is_file()
        or sidecar.read_text().split()[0] != base.sha256(F37_SEAL)
        or seal.get("status") != "sealed_before_symlink_loader_amendment_blind_prediction"
        or seal.get("passed") is not True
        or seal.get("model_prediction_or_score_read") is not False
        or seal.get("parent_f36_status") != "result_blind_loader_failure_preserved"
        or source.get("f36_failure_audit")
        != base.sha256(ROOT / "outputs/kinofail_reconfirmation_f36_failure_audit_v1/audit.json")
        or source.get("fixed_v5_loader")
        != base.sha256(ROOT / "scripts/build_kinofail_realistic_c2_v5_features.py")
        or scripts.get(relative_self) != base.sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("invalid F37 pre-prediction amendment seal")
    return seal


def main() -> int:
    # Reuse the already audited F36 one-shot pipeline while replacing only the
    # seal validator and write-once destination.  Corpus, model, features,
    # threshold, and scoring code remain those bound in the F37 amendment seal.
    base.F36_SEAL = F37_SEAL
    base.TARGET = TARGET
    base.validate_seal = validate_seal
    result = int(base.main())

    audit_path = TARGET / "finalization_audit.json"
    audit = base.read_json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.f37.v1"
    audit["f37_symlink_loader_amendment"] = {
        "seal": str(F37_SEAL.relative_to(ROOT)),
        "seal_sha256": base.sha256(F37_SEAL),
        "parent_f36_seal_sha256": base.sha256(
            ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f36/seal_manifest.json"
        ),
        "f36_failure_audit_sha256": base.sha256(
            ROOT / "outputs/kinofail_reconfirmation_f36_failure_audit_v1/audit.json"
        ),
        "only_amendment": "fixed-depth enumeration of F30 episode-directory symlinks",
        "failure_was_result_blind": True,
        "model_features_threshold_statistics_and_schedule_unchanged": True,
    }
    audit["f37_finalized_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
