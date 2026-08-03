#!/usr/bin/env python3
"""Preserve the result-blind F37 temporal-window incompatibility."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kino_vla.eval.c2_temporal_v5 import geometry_aligned_invariant_summary  # noqa: E402
from scripts.build_kinofail_realistic_c2_v5_features import index_t3_episode_dirs  # noqa: E402


F37 = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f37"
SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f37/seal_manifest.json"
STATE = ROOT / "outputs/kinofail_after_a4_pilot_to_a0_a7_v3/state.json"
T3 = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_replenishment_f30/union")
OUTPUT = ROOT / "outputs/kinofail_reconfirmation_f37_failure_audit_v1/audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    seal = read_json(SEAL)
    state = read_json(STATE)
    log = F37 / "finalization_logs/04_conflict_v5_features.log"
    records = F37 / "conflict/c2_base/records.jsonl"
    if (
        seal.get("status") != "sealed_before_symlink_loader_amendment_blind_prediction"
        or seal.get("model_prediction_or_score_read") is not False
        or state.get("state") != "terminal_failure"
        or state.get("stage") != "04_finalize_f37_once"
        or "short post-interaction window" not in log.read_text()
    ):
        raise RuntimeError("F37 failure boundary is invalid")
    forbidden = [
        path
        for pattern in ("*prediction*", "*truth*", "*score*", "confirmatory_report.json", "finalization_audit.json")
        for path in F37.rglob(pattern)
    ]
    if forbidden:
        raise RuntimeError(f"F37 is not result-blind: {forbidden[:3]}")
    rows = [json.loads(line) for line in records.read_text().splitlines() if line]
    episode_ids = sorted(
        {
            str(row.get("proprio_source_episode_id") or f"{row['source_physics_group_id']}_anomaly")
            for row in rows
            if row["cell"] == "T3_proprio_decisive"
        }
    )
    index = index_t3_episode_dirs(T3)
    reasons: dict[str, int] = {}
    passed = 0
    for episode_id in episode_ids:
        try:
            geometry_aligned_invariant_summary(index[episode_id])
        except (KeyError, OSError, TypeError, ValueError) as exc:
            key = str(exc).split(":", 1)[0]
            reasons[key] = reasons.get(key, 0) + 1
        else:
            passed += 1
    report = {
        "schema_version": "kinofail.reconfirmation-f37-failure-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "result_blind_systematic_temporal_contract_failure_preserved",
        "passed": True,
        "scientific_attempt_started": False,
        "model_prediction_truth_key_or_score_read": False,
        "failure_stage": "04_conflict_v5_features",
        "diagnosis": (
            "the reachable-region T3 collector places the robot footprint in the "
            "operator region at the first telemetry sample, so the frozen 21-sample "
            "window ending 0.30 s after encounter has insufficient prehistory"
        ),
        "counts": {
            "t3_source_episodes_checked": len(episode_ids),
            "temporally_extractable": passed,
            "temporally_incompatible": len(episode_ids) - passed,
        },
        "failure_reasons": reasons,
        "prohibited_shortcuts": [
            "repeat first sample padding",
            "post-hoc decision-time shift",
            "silent exclusion of all T3 cases",
        ],
        "required_remedy": (
            "result-blind physical T3 recollection with a pre-frozen run-in region, "
            "while retaining the original v5 feature function, model, threshold, and statistics"
        ),
        "source_sha256": {
            "f37_seal": sha256(SEAL),
            "f37_state": sha256(STATE),
            "f37_failure_log": sha256(log),
            "f37_base_records": sha256(records),
            "v5_feature_function": sha256(ROOT / "kino_vla/eval/c2_temporal_v5.py"),
            "v5_feature_builder": sha256(ROOT / "scripts/build_kinofail_realistic_c2_v5_features.py"),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
