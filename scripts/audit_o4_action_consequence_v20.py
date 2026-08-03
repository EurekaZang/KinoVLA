#!/usr/bin/env python3
"""Independently seal the frozen O4 v20 Continue/Backstep action consequence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"expected JSON objects in: {path}")
    return rows


def _canonical_rows_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _locked(record: dict[str, Any]) -> bool:
    path = _resolve(record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        return False
    return "json_passed" not in record or _json(path).get("passed") is record["json_passed"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    manifest_path = args.manifest.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    manifest = _json(manifest_path)
    case = config["cases"][0]
    runtime = config["runtime"]

    summaries: dict[str, dict[str, Any]] = {}
    rows_by_take: dict[str, list[dict[str, Any]]] = {}
    artifact_checks: dict[str, bool] = {}
    for take in ("continue", "backstep"):
        take_dir = manifest_path.parent / take
        summary_path = take_dir / "summary.json"
        summary = _json(summary_path)
        telemetry_path = _resolve(summary["telemetry"])
        rows = _jsonl(telemetry_path)
        summaries[take] = summary
        rows_by_take[take] = rows
        artifact_checks[f"{take}_summary_matches_manifest"] = (
            summary == manifest.get("takes", {}).get(take)
        )
        artifact_checks[f"{take}_telemetry_hash_matches"] = (
            _sha256(telemetry_path) == summary.get("telemetry_sha256")
        )
        artifact_checks[f"{take}_step_count_matches"] = len(rows) == summary.get("steps")
        for capture in summary.get("captures", []):
            image_path = take_dir / capture["image"]
            artifact_checks[f"{take}_{capture['label']}_image_hash_matches"] = (
                image_path.is_file() and _sha256(image_path) == capture.get("image_sha256")
            )

    decision_step = int(summaries["continue"]["predecision_step"])
    branch_step = decision_step + 1
    continue_rows = rows_by_take["continue"]
    backstep_rows = rows_by_take["backstep"]
    continue_pre = continue_rows[: decision_step + 1]
    backstep_pre = backstep_rows[: decision_step + 1]
    pre_hash = _canonical_rows_hash(continue_pre)
    outcome_checks = {
        "collector_reported_all_checks_passed": manifest.get("passed") is True
        and bool(manifest.get("checks"))
        and all(manifest["checks"].values()),
        "predecision_rows_exactly_equal": continue_pre == backstep_pre,
        "predecision_hash_recomputed": pre_hash
        == _canonical_rows_hash(backstep_pre)
        == summaries["continue"].get("predecision_rows_sha256")
        == summaries["backstep"].get("predecision_rows_sha256"),
        "branch_step_is_exact": summaries["continue"].get("action_branch_step")
        == summaries["backstep"].get("action_branch_step")
        == branch_step,
        "actions_diverge_only_after_decision": continue_rows[branch_step]["action_phase"]
        == "matched_forward"
        and backstep_rows[branch_step]["action_phase"] == "backstep",
        "continue_falls_after_branch": summaries["continue"].get("fell") is True
        and int(summaries["continue"]["first_fall_step"]) > branch_step,
        "backstep_completes_fixed_horizon": summaries["backstep"].get("fell") is False
        and summaries["backstep"].get("steps") == runtime["horizon_steps"],
        "backstep_has_physical_peel": summaries["backstep"].get(
            "postdecision_peel_events", 0
        )
        >= 1,
        "backstep_retreat_distance_passes": summaries["backstep"].get(
            "backward_recovery_distance_m", -1.0
        )
        >= runtime["minimum_recovery_distance_m"],
        "both_routes_contained": all(
            summaries[take].get("maximum_absolute_route_lateral_offset_m", float("inf"))
            <= runtime["maximum_route_deviation_m"]
            for take in ("continue", "backstep")
        ),
        "command_agnostic_physics_readback": all(
            row["adhesion"].get("mode")
            == "command_agnostic_contact_surface_adhesion_v3"
            and row["adhesion"].get("command_conditioned_release") is False
            for take in ("continue", "backstep")
            for row in rows_by_take[take]
        ),
    }
    provenance = manifest.get("provenance", {})
    integrity_checks = {
        "preflight_passed": preflight.get("passed") is True,
        "preflight_config_hash_matches": preflight.get("config_sha256") == _sha256(config_path),
        "locked_files_unchanged": all(_locked(record) for record in config["locked_files"]),
        "single_train_only_case": len(config.get("cases", [])) == 1
        and config.get("evidence_policy", {}).get("train_only_scene") is True,
        "manifest_identity_matches": manifest.get("schema_version")
        == "kinofail.o4-action-consequence.v20"
        and manifest.get("scene_id") == case["source_scene_id"]
        and manifest.get("operator", {}).get("id") == "O4"
        and manifest.get("operator", {}).get("subtype") == "adhesive_contact",
        "evidence_quarantine_preserved": manifest.get("development_only") is True
        and manifest.get("counts_as_a0_a7_evidence") is False
        and manifest.get("realistic_a0_a7_readiness") == "0/8",
        "collector_hash_matches_lock": provenance.get("collector_sha256")
        == config["provenance_contract"]["collector_sha256"],
        "surface_hashes_match_lock": provenance.get("surface_interface_sha256")
        == config["provenance_contract"]["surface_interface_sha256"]
        and provenance.get("surface_model_sha256")
        == config["provenance_contract"]["surface_model_sha256"],
        "backend_hashes_match_lock": provenance.get("backend_adapter_sha256")
        == config["provenance_contract"]["backend_adapter_sha256"]
        and provenance.get("parent_backend_adapter_sha256")
        == config["provenance_contract"]["parent_backend_adapter_sha256"],
        "all_artifact_hashes_match": all(artifact_checks.values()),
    }
    integrity_passed = all(integrity_checks.values())
    outcome_passed = all(outcome_checks.values())
    audit = {
        "schema_version": "kinofail.o4-action-consequence-v20-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "artifact_checks": artifact_checks,
        "integrity_checks": integrity_checks,
        "outcome_checks": outcome_checks,
        "integrity_passed": integrity_passed,
        "outcome_passed": outcome_passed,
        "passed": integrity_passed and outcome_passed,
        "sealed": integrity_passed,
        "development_only": True,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "interpretation": (
            "A passing result authorizes fresh held-out O4 confirmation only. It is not an "
            "A0-A7 result and cannot be combined with legacy-domain metrics."
        ),
        "next_gate": (
            "Freeze newly generated, never-calibrated realistic scenes and repeat the matched "
            "Continue/Backstep contract once per scene without parameter changes."
        ),
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"out": str(out), "passed": audit["passed"], "sealed": audit["sealed"]},
            indent=2,
        )
    )
    return 0 if audit["sealed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
