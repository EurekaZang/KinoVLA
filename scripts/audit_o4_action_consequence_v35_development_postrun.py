#!/usr/bin/env python3
"""Seal v35 adaptive-Backstep development under the v34 phase-aware route gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_o4_action_consequence_v27_development_postrun import (  # noqa: E402
    _audit_case as _audit_v27_case,
    _locked,
)
from audit_o4_action_consequence_v32_development_postrun import (  # noqa: E402
    _early_branch_checks,
    _state_aware_visual_checks,
)
from audit_o4_action_consequence_v34_confirmation_postrun import (  # noqa: E402
    _phase_aware_route_checks,
)


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
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _adaptive_contract_checks(
    manifest: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, bool]:
    decision = manifest.get("decision_contract", {})
    guard = decision.get("adaptive_backstep_guard", {})
    expected_action = (
        f"reverse {runtime['backstep_speed_mps']} m/s, latching "
        f"{runtime['emergency_backstep_speed_mps']} m/s when physical urgency or "
        f"{runtime['maximum_postdecision_forward_penetration_m']} m forward penetration "
        f"is observed, until retreat reaches {runtime['target_recovery_distance_m']} m "
        f"and active adhesion remains clear for {runtime['adhesion_clearance_dwell_steps']} "
        f"steps, capped at {runtime['maximum_backstep_steps']} steps, then hold"
    )
    backstep = manifest.get("takes", {}).get("backstep", {})
    escalated = backstep.get("backstep_escalated")
    step = backstep.get("backstep_escalation_step")
    reason = backstep.get("backstep_escalation_reason")
    escalation_record_valid = (
        escalated is False and step is None and reason is None
    ) or (
        escalated is True
        and isinstance(step, int)
        and step > backstep.get("predecision_step", step)
        and reason in {"physical_urgency", "forward_penetration_guard"}
    )
    return {
        "adaptive_backstep_action_exact": decision.get("backstep_action") == expected_action,
        "adaptive_guard_parameters_exact": guard == {
            "emergency_backstep_speed_mps": runtime["emergency_backstep_speed_mps"],
            "maximum_postdecision_forward_penetration_m": runtime[
                "maximum_postdecision_forward_penetration_m"
            ],
            "physical_urgency_uses_frozen_decision_thresholds": True,
            "latches_after_first_trigger": True,
            "scene_identity_used": False,
            "command_conditioned_release_used": False,
        },
        "adaptive_escalation_record_valid": escalation_record_valid,
        "full_horizon_backstep_contract": runtime["maximum_backstep_steps"]
        == runtime["horizon_steps"]
        == 300,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    output_root = _resolve(config["operator_output_root"])
    runner_path = output_root / "runner_audit.json"
    runner = _json(runner_path)
    launchers = {row.get("scene_id"): row for row in runner.get("cases", [])}

    cases = []
    for frozen_case in config["cases"]:
        case = _audit_v27_case(frozen_case, config["runtime"])
        manifest_path = Path(case["manifest"])
        manifest = _json(manifest_path)
        launcher = launchers.get(frozen_case["scene_id"], {})
        collector_checks = manifest.get("checks", {})
        unreplaced_collector_checks = {
            key: value
            for key, value in collector_checks.items()
            if key not in {"all_frames_non_degenerate", "both_lanes_within_route_budget"}
        }
        outcome_checks = {
            key: value
            for key, value in case["outcome_checks"].items()
            if key not in {"collector_reported_all_checks_passed", "both_routes_contained"}
        }
        outcome_checks["collector_unreplaced_checks_pass"] = (
            bool(unreplaced_collector_checks)
            and all(value is True for value in unreplaced_collector_checks.values())
        )
        outcome_checks.update(_early_branch_checks(manifest, config["runtime"]))
        outcome_checks.update(
            _state_aware_visual_checks(manifest, config["visual_qa_contract"])
        )
        outcome_checks.update(_adaptive_contract_checks(manifest, config["runtime"]))
        route_checks, route_metrics = _phase_aware_route_checks(
            manifest, config["runtime"]
        )
        outcome_checks.update(route_checks)

        identity = (
            manifest.get("schema_version") == "kinofail.o4-action-consequence.v35"
            and manifest.get("scene_id") == frozen_case["scene_id"]
            and manifest.get("room_family") == frozen_case["room_family"]
            and manifest.get("material_id") == frozen_case["material_id"]
            and manifest.get("seed") == frozen_case["runtime_seed"]
        )
        case["integrity_checks"]["manifest_identity_matches"] = identity
        case["integrity_checks"].pop("collector_hash_matches_v27", None)
        case["integrity_checks"]["collector_hash_matches_v35"] = (
            manifest.get("provenance", {}).get("collector_sha256")
            == config["runtime"]["collector_sha256"]
        )
        case["integrity_checks"].update({
            "launcher_terminal": launcher.get("collector_terminal_returncode") is True,
            "launcher_manifest_hash_matches": manifest_path.is_file()
            and launcher.get("manifest_sha256") == _sha256(manifest_path),
        })
        case["integrity_passed"] = all(case["integrity_checks"].values())
        case["outcome_checks"] = outcome_checks
        case["outcome_passed"] = bool(outcome_checks) and all(outcome_checks.values())
        case["sealed"] = case["integrity_passed"]
        case["passed"] = case["integrity_passed"] and case["outcome_passed"]
        backstep = manifest.get("takes", {}).get("backstep", {})
        case["metrics"].update(route_metrics)
        case["metrics"].update({
            "backstep_escalated": backstep.get("backstep_escalated"),
            "backstep_escalation_step": backstep.get("backstep_escalation_step"),
            "backstep_escalation_reason": backstep.get("backstep_escalation_reason"),
            "decision_progress_m": backstep.get("decision_progress_m"),
            "final_progress_m": backstep.get("final_progress_m"),
            "legacy_collector_both_lanes_within_route_budget": collector_checks.get(
                "both_lanes_within_route_budget"
            ),
            "legacy_collector_all_frames_non_degenerate": collector_checks.get(
                "all_frames_non_degenerate"
            ),
        })
        cases.append(case)

    frozen_ids = [case["scene_id"] for case in config["cases"]]
    integrity_checks = {
        "preflight_passed_and_bound": preflight.get("passed") is True
        and preflight.get("execution_authorized") is True
        and preflight.get("config_sha256") == _sha256(config_path),
        "locked_files_unchanged": all(_locked(item) for item in config["locked_files"]),
        "runner_bound_to_config_and_preflight": runner.get("config_sha256")
        == _sha256(config_path)
        and runner.get("preflight_sha256") == _sha256(preflight_path),
        "every_frozen_case_attempted_in_order": runner.get("attempted_scene_ids")
        == frozen_ids
        and runner.get("all_cases_attempted") is True,
        "every_case_terminal_with_manifest": runner.get("all_cases_terminal") is True
        and runner.get("all_manifests_present") is True,
        "all_case_integrity_passed": all(case["integrity_passed"] for case in cases),
    }
    integrity_passed = all(integrity_checks.values())
    outcome_passed = all(case["outcome_passed"] for case in cases)
    passed_count = sum(case["passed"] for case in cases)
    result = {
        "schema_version": "kinofail.o4-action-consequence-v35-development-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "runner_audit": str(runner_path),
        "runner_audit_sha256": _sha256(runner_path) if runner_path.is_file() else None,
        "integrity_checks": integrity_checks,
        "integrity_passed": integrity_passed,
        "outcome_passed": outcome_passed,
        "passed": integrity_passed and outcome_passed,
        "sealed": integrity_passed,
        "total_scenes": len(cases),
        "passed_scenes": passed_count,
        "room_families": sorted({case["room_family"] for case in cases}),
        "cases": cases,
        "development_only": True,
        "counts_as_realistic_o4_operator_confirmation": False,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "interpretation": (
            "This seals one-shot v35 architecture development on the complete v36 cohort. "
            "It is neither held-out O4 confirmation nor A0-A7 evidence."
        ),
        "next_gate": (
            "If v35 passes, acquire a fresh operator-blind cohort and execute the unchanged "
            "architecture once. Never rerun these development scenes."
        ),
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite postrun: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "passed": result["passed"],
        "sealed": result["sealed"],
        "passed_scenes": passed_count,
        "total_scenes": len(cases),
    }, indent=2))
    return 0 if result["sealed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
