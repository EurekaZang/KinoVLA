#!/usr/bin/env python3
"""Seal the v5-to-v6 pre-acquisition provenance-binding correction."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F0 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f0_transitive_amendment1/"
    "freeze_manifest.json"
)
F1 = (
    ROOT
    / "outputs/freeze/unified_moe_v3_reconfirmation_f1/"
    "seal_manifest.json"
)
SCHEDULE_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2/schedules"
SCRATCH_SCENE = Path(
    "/media/eureka/FC28565528560ED0/tmp/KinoVLA_reconfirmation_v2/"
    "corpus/confirm_v2_life_scene_00"
)
EVAL_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
V5 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v5.py"
V6 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v6.py"
OUTPUT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f2_collector_provenance_amendment1/"
    "amendment_manifest.json"
)
EXPECTED_F0_SHA256 = (
    "a2b11e281fd7a75dac3c2b443a47dc80609d5080a2a73139df6e5b6a5a9c68a3"
)
EXPECTED_F1_SHA256 = (
    "f59870031e9f22944e80451c99d5ccb55cb46b3951993ba4929d3587056f6a0f"
)
EXPECTED_V5_SHA256 = (
    "88f8e3abe6805bb1e96a87f333dad29d285b49ec1dbf40e4b626b020bb36419d"
)
PROVENANCE_ERROR = "RuntimeError: formal protocol collector_sha256 mismatch"


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


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path, expected in (
        (F0, EXPECTED_F0_SHA256),
        (F1, EXPECTED_F1_SHA256),
        (V5, EXPECTED_V5_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"predecessor hash mismatch: {path}")

    audit_paths = sorted(
        (SCRATCH_SCENE / "launcher_audits").glob("scale_partition_*_of_4.json")
    )
    if len(audit_paths) != 3:
        raise RuntimeError("expected exactly three affected scale partitions")
    attempts: list[dict[str, Any]] = []
    audit_hashes = {}
    for audit_path in audit_paths:
        audit = _json(audit_path)
        audit_hashes[str(audit_path)] = _sha256(audit_path)
        for attempt in audit["attempts"]:
            log_path = Path(str(attempt["log"]))
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            terminal = attempt.get("state") == "terminal"
            if terminal and (
                int(attempt.get("returncode", -1)) != 1
                or attempt.get("summary_exists") is not False
                or PROVENANCE_ERROR not in log_text
            ):
                raise RuntimeError("affected terminal attempt has another failure mode")
            attempts.append(
                {
                    "counterfactual_group_id": str(
                        attempt["counterfactual_group_id"]
                    ),
                    "state_at_interruption": str(attempt["state"]),
                    "returncode": attempt.get("returncode"),
                    "summary_exists": bool(attempt.get("summary_exists", False)),
                    "log": str(log_path),
                    "log_sha256": _sha256(log_path),
                    "provenance_guard_error_verified": (
                        PROVENANCE_ERROR in log_text
                    ),
                }
            )
    pair_ids = [row["counterfactual_group_id"] for row in attempts]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("affected pair IDs are not unique")

    pair_summaries = list((SCRATCH_SCENE / "pair_summaries").glob("*.json"))
    scale_episode_manifests = list(
        (SCRATCH_SCENE / "scale").glob("**/manifest.json")
    )
    t3_episode_manifests = list(
        (SCRATCH_SCENE / "c2_t3").glob("**/manifest.json")
    )
    eval_artifacts = (
        [path for path in EVAL_ROOT.rglob("*") if path.is_file()]
        if EVAL_ROOT.exists()
        else []
    )
    if pair_summaries or scale_episode_manifests or t3_episode_manifests:
        raise RuntimeError("incident is no longer pre-acquisition")
    if eval_artifacts:
        raise RuntimeError("model-blind or model outputs already exist")

    terminal_attempts = [
        row for row in attempts if row["state_at_interruption"] == "terminal"
    ]
    interrupted_attempts = [
        row for row in attempts if row["state_at_interruption"] == "started"
    ]
    if not terminal_attempts or len(interrupted_attempts) != 3:
        raise RuntimeError("unexpected incident attempt state")

    affected_protocols = []
    for battery in ("scale", "c2_t3"):
        for path in sorted(
            (SCHEDULE_ROOT / "scenes").glob(
                f"*/{battery}/collection_protocol.json"
            )
        ):
            protocol = _json(path)
            if protocol.get("collector_sha256") != EXPECTED_V5_SHA256:
                raise RuntimeError("formal protocol is not bound to v5")
            affected_protocols.append(
                {"path": str(path), "sha256": _sha256(path)}
            )

    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-collector-provenance-amendment.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_scale_or_t3_episode_acquisition",
        "passed": True,
        "incident": {
            "classification": "pre_acquisition_provenance_guard",
            "cause": (
                "v5 executes the exact v4 scientific wrapper, whose dynamically "
                "loaded module retained v4 __file__; the inner guard therefore "
                "compared the formal v5 hash against v4 before _collect_one."
            ),
            "episode_manifests_written": 0,
            "pair_summaries_written": 0,
            "scale_or_t3_features_written": 0,
            "model_predictions_written": 0,
            "terminal_guard_failures": len(terminal_attempts),
            "interrupted_pre_acquisition_processes": len(interrupted_attempts),
            "affected_pair_count": len(attempts),
            "affected_pair_ids": sorted(pair_ids),
            "attempts": attempts,
            "launcher_audit_sha256": audit_hashes,
        },
        "correction": {
            "wrapper": str(V6),
            "wrapper_sha256": _sha256(V6),
            "exact_hash_predecessor": str(V5),
            "exact_hash_predecessor_sha256": _sha256(V5),
            "only_change": (
                "set the dynamically loaded scientific wrapper's __file__ to "
                "the F0-bound v5 path before entering its unchanged main"
            ),
            "simulation_logic_changed": False,
            "operator_parameters_changed": False,
            "sensor_or_feature_logic_changed": False,
            "threshold_or_analysis_changed": False,
            "result_dependent_selection_changed": False,
            "authorized_reexecution": (
                "only the enumerated pairs rejected or interrupted before "
                "episode creation; same frozen schedule, seeds, and protocols"
            ),
        },
        "timing_and_scope": {
            "t2_acquisition_had_started": True,
            "t2_collector_or_design_changed": False,
            "t2_data_were_not_read_to_design_the_correction": True,
            "scale_or_t3_acquisition_had_started": False,
            "model_blind_feature_extraction_had_started": False,
            "model_inference_had_started": False,
        },
        "predecessors": {
            "f0": str(F0),
            "f0_sha256": _sha256(F0),
            "f1": str(F1),
            "f1_sha256": _sha256(F1),
        },
        "immutable_formal_protocols": affected_protocols,
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
                "affected_pair_count": len(attempts),
                "v6_sha256": _sha256(V6),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
