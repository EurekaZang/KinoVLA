#!/usr/bin/env python3
"""Seal exact F0 nuisance-table validation after the legacy-range incident."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F2 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f2_collector_provenance_amendment1/"
    "amendment_manifest.json"
)
DESIGN = ROOT / "configs/data/kinofail_unified_reconfirmation_design_v2.json"
V5 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v5.py"
V7 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v7.py"
SCHEDULE_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2/schedules"
SCENE = "confirm_v2_life_scene_00"
SCRATCH_SCENE = Path(
    "/media/eureka/FC28565528560ED0/tmp/"
    f"KinoVLA_reconfirmation_v2/corpus/{SCENE}"
)
ARCHIVE_RECEIPT = (
    SCRATCH_SCENE
    / "incidents/f3_nuisance_contract_interruption/archive_receipt.json"
)
EVAL_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
OUTPUT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f3_nuisance_contract_amendment1/"
    "amendment_manifest.json"
)
EXPECTED_F2_SHA256 = (
    "ab0da5f6e2572e380784dcc2534099a550d54a939f65f37e35c90d1c10ce6702"
)
EXPECTED_DESIGN_SHA256 = (
    "1a1fd3c53590dd745f12b7ca4582c6b186514077ca168f712ae8338938d428c2"
)
EXPECTED_V5_SHA256 = (
    "88f8e3abe6805bb1e96a87f333dad29d285b49ec1dbf40e4b626b020bb36419d"
)
EXPECTED_ARCHIVE_RECEIPT_SHA256 = (
    "f8039fe07d4afe704562627cb639ae4bb77543e3b7c580607ce26df36e059126"
)
GUARD_FAILURE_PAIR = "cf_0181579aad8dd1e2599c"
INTERRUPTED_PAIR_IDS = {
    "cf_025108cfb719cb716b63",
    "cf_04678ef990b4ff57de94",
    "cf_07365e997113eacd4de6",
}
RECOVERED_COMPLETE_PAIR = "cf_0348b2a192f86369ee8a"
ACTUAL_QA_FAILURE_PAIR = "cf_00c828424846ba775cdb"


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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path, expected in (
        (F2, EXPECTED_F2_SHA256),
        (DESIGN, EXPECTED_DESIGN_SHA256),
        (V5, EXPECTED_V5_SHA256),
        (ARCHIVE_RECEIPT, EXPECTED_ARCHIVE_RECEIPT_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"F3 dependency hash mismatch: {path}")

    design = _json(DESIGN)
    contract = design["scale"]["physical_nuisance"]
    profiles = {
        int(row["profile_index"]): dict(row)
        for row in contract["profile_table"]
    }
    if set(profiles) != set(range(16)):
        raise RuntimeError("F0 nuisance table does not contain 16 profiles")
    affected_profiles = sorted(
        index
        for index, row in profiles.items()
        if not 0.0 <= float(row["start_progress_m"]) <= 0.14
        or not 0.21 <= float(row["forward_speed_mps"]) <= 0.27
    )
    if affected_profiles != [2, 6, 9]:
        raise RuntimeError("legacy/F0 nuisance mismatch changed")

    affected_counts = {}
    for battery in ("scale", "c2_t3"):
        pairs = set()
        rows = 0
        for schedule in sorted(
            (SCHEDULE_ROOT / "scenes").glob(f"*/{battery}/schedule.jsonl")
        ):
            for record in _jsonl(schedule):
                if int(record["physical_nuisance"]["profile_index"]) in affected_profiles:
                    rows += 1
                    pairs.add(str(record["counterfactual_group_id"]))
        affected_counts[battery] = {
            "schedule_rows": rows,
            "counterfactual_pairs": len(pairs),
        }

    audits = sorted(
        (SCRATCH_SCENE / "launcher_audits").glob(
            "scale_f2v6_partition_*_of_4.json"
        )
    )
    if len(audits) != 4:
        raise RuntimeError("F3 needs all four interrupted v6 partition audits")
    attempts = {}
    for audit_path in audits:
        for row in _json(audit_path)["attempts"]:
            attempts[str(row["counterfactual_group_id"])] = {
                **row,
                "audit": str(audit_path),
                "audit_sha256": _sha256(audit_path),
                "log_sha256": _sha256(Path(str(row["log"]))),
            }

    guard_attempt = attempts.get(GUARD_FAILURE_PAIR)
    if (
        not guard_attempt
        or guard_attempt.get("state") != "terminal"
        or guard_attempt.get("summary_exists") is not False
        or "confirmatory start progress is outside the frozen band"
        not in Path(str(guard_attempt["log"])).read_text(
            encoding="utf-8", errors="replace"
        )
    ):
        raise RuntimeError("F0 nuisance guard incident is not exact")
    for pair_id in INTERRUPTED_PAIR_IDS:
        if attempts.get(pair_id, {}).get("state") != "started":
            raise RuntimeError("expected an explicitly interrupted v6 pair")
        if (SCRATCH_SCENE / "pair_summaries" / f"{pair_id}.json").exists():
            raise RuntimeError("interrupted pair unexpectedly has a summary")

    recovered_summary = _json(
        SCRATCH_SCENE
        / "pair_summaries"
        / f"{RECOVERED_COMPLETE_PAIR}.json"
    )
    actual_failure = _json(
        SCRATCH_SCENE
        / "pair_summaries"
        / f"{ACTUAL_QA_FAILURE_PAIR}.json"
    )
    if recovered_summary.get("passed") is not True:
        raise RuntimeError("completed interrupted pair was not recovered")
    if actual_failure.get("passed") is not False:
        raise RuntimeError("actual acquisition failure must remain terminal")

    archive = _json(ARCHIVE_RECEIPT)
    if (
        archive.get("passed") is not True
        or set(archive.get("interrupted_pair_ids", []))
        != INTERRUPTED_PAIR_IDS
    ):
        raise RuntimeError("partial-pair archive receipt is invalid")
    eval_artifacts = (
        [path for path in EVAL_ROOT.rglob("*") if path.is_file()]
        if EVAL_ROOT.exists()
        else []
    )
    if eval_artifacts:
        raise RuntimeError("features or predictions exist before F3")

    authorized_reexecution = sorted(
        INTERRUPTED_PAIR_IDS | {GUARD_FAILURE_PAIR}
    )
    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-f0-nuisance-contract-amendment.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_model_blind_feature_extraction",
        "passed": True,
        "cause": {
            "classification": "legacy_guard_narrower_than_f0_schedule",
            "legacy_guard": {
                "start_progress_m": [0.0, 0.14],
                "forward_speed_mps": [0.21, 0.27],
            },
            "f0_affected_profile_indices": affected_profiles,
            "affected_frozen_schedule_counts": affected_counts,
            "identified_from": (
                "deterministic comparison of the exact F0 table with the "
                "legacy source guard after its pre-acquisition exception"
            ),
            "model_or_endpoint_outcomes_used": False,
        },
        "correction": {
            "wrapper": str(V7),
            "wrapper_sha256": _sha256(V7),
            "formal_protocol_predecessor": str(V5),
            "formal_protocol_predecessor_sha256": _sha256(V5),
            "f0_design": str(DESIGN),
            "f0_design_sha256": _sha256(DESIGN),
            "validation_rule": (
                "exact profile-index lookup and exact equality to all values "
                "in the F0 16-profile table"
            ),
            "simulation_logic_changed": False,
            "operator_parameters_changed": False,
            "scheduled_nuisance_values_changed": False,
            "sensor_or_feature_logic_changed": False,
            "threshold_or_analysis_changed": False,
            "result_dependent_selection_changed": False,
        },
        "incident_disposition": {
            "authorized_reexecution_pair_ids": authorized_reexecution,
            "guard_rejected_before_episode_creation": [GUARD_FAILURE_PAIR],
            "recoverably_archived_interrupted_pairs": sorted(
                INTERRUPTED_PAIR_IDS
            ),
            "partial_archive_receipt": str(ARCHIVE_RECEIPT),
            "partial_archive_receipt_sha256": _sha256(ARCHIVE_RECEIPT),
            "completed_pair_retained_without_reexecution": [
                RECOVERED_COMPLETE_PAIR
            ],
            "actual_physics_qa_failure_retained_without_reexecution": [
                ACTUAL_QA_FAILURE_PAIR
            ],
            "v6_attempts": attempts,
        },
        "timing": {
            "some_scale_acquisition_outcomes_existed": True,
            "outcomes_did_not_determine_the_correction": True,
            "model_blind_features_existed": False,
            "model_predictions_existed": False,
            "architecture_thresholds_features_and_statistics_changed": False,
        },
        "predecessor_f2": str(F2),
        "predecessor_f2_sha256": _sha256(F2),
        "amendment_source": str(Path(__file__).resolve()),
        "amendment_source_sha256": _sha256(Path(__file__).resolve()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "output": str(OUTPUT),
                "sha256": _sha256(OUTPUT),
                "v7_sha256": _sha256(V7),
                "authorized_reexecution_pair_ids": authorized_reexecution,
                "affected_counts": affected_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
