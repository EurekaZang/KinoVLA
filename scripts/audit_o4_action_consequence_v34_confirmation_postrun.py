#!/usr/bin/env python3
"""Seal prospective v34 O4 confirmation with phase-aware route containment."""

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
    _horizon_contract_checks,
    _state_aware_visual_checks,
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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows if all(isinstance(row, dict) for row in rows) else []


def _telemetry_rows(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for take in ("continue", "backstep"):
        path = Path(manifest.get("takes", {}).get(take, {}).get("telemetry", ""))
        if path and not path.is_absolute():
            path = ROOT / path
        result[take] = _jsonl(path)
    return result


def _maximum_abs_offset(rows: list[dict[str, Any]]) -> float:
    return max(
        (abs(float(row.get("route_lateral_offset_m", float("inf")))) for row in rows),
        default=float("inf"),
    )


def _phase_aware_route_checks(
    manifest: dict[str, Any],
    runtime: dict[str, Any],
    rows_by_take: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, bool], dict[str, Any]]:
    rows_by_take = rows_by_take or _telemetry_rows(manifest)
    takes = manifest.get("takes", {})
    continue_summary = takes.get("continue", {})
    backstep_summary = takes.get("backstep", {})
    continue_rows = rows_by_take.get("continue", [])
    backstep_rows = rows_by_take.get("backstep", [])
    decision_step = continue_summary.get("predecision_step")
    height_reference = continue_summary.get("preattachment_base_height_m")
    onset = None
    if isinstance(decision_step, int) and isinstance(height_reference, (int, float)):
        onset = next(
            (
                row
                for row in continue_rows
                if row.get("step", -1) > decision_step
                and float(row.get("tilt_rad", -1.0))
                >= runtime["decision_tilt_urgency_rad"]
                and float(height_reference) - float(row.get("base_height_m", float("inf")))
                >= runtime["decision_height_drop_urgency_m"]
            ),
            None,
        )
    prefix = (
        [row for row in continue_rows if row.get("step", -1) <= decision_step]
        if isinstance(decision_step, int)
        else []
    )
    through_onset = (
        [row for row in continue_rows if row.get("step", -1) <= onset["step"]]
        if onset is not None
        else []
    )
    first_over_budget = next(
        (
            row
            for row in continue_rows
            if abs(float(row.get("route_lateral_offset_m", float("inf"))))
            > runtime["maximum_route_deviation_m"]
        ),
        None,
    )
    prefix_max = _maximum_abs_offset(prefix)
    onset_max = _maximum_abs_offset(through_onset)
    backstep_max = _maximum_abs_offset(backstep_rows)
    checks = {
        "matched_prefix_within_route_budget": bool(prefix)
        and prefix_max <= runtime["maximum_route_deviation_m"],
        "backstep_full_horizon_within_route_budget": bool(backstep_rows)
        and backstep_max <= runtime["maximum_route_deviation_m"],
        "physical_failure_onset_observed_after_decision": onset is not None,
        "continue_through_failure_onset_within_route_budget": bool(through_onset)
        and onset_max <= runtime["maximum_route_deviation_m"],
        "route_deviation_not_used_to_define_failure_onset": True,
    }
    metrics = {
        "matched_prefix_max_route_deviation_m": prefix_max,
        "backstep_full_horizon_max_route_deviation_m": backstep_max,
        "physical_failure_onset_step": onset.get("step") if onset else None,
        "physical_failure_onset_route_deviation_m": (
            abs(float(onset["route_lateral_offset_m"])) if onset else None
        ),
        "continue_max_route_deviation_through_failure_onset_m": onset_max,
        "continue_first_over_route_budget_step": (
            first_over_budget.get("step") if first_over_budget else None
        ),
        "continue_first_over_route_budget_after_failure_onset": (
            first_over_budget is None
            or onset is not None
            and first_over_budget.get("step", -1) > onset.get("step", -1)
        ),
        "continue_full_horizon_max_route_deviation_m": _maximum_abs_offset(
            continue_rows
        ),
    }
    return checks, metrics


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
        replaced_collector_checks = {
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
            bool(replaced_collector_checks)
            and all(value is True for value in replaced_collector_checks.values())
        )
        outcome_checks.update(_early_branch_checks(manifest, config["runtime"]))
        outcome_checks.update(_horizon_contract_checks(manifest, config["runtime"]))
        outcome_checks.update(
            _state_aware_visual_checks(manifest, config["visual_qa_contract"])
        )
        route_checks, route_metrics = _phase_aware_route_checks(
            manifest, config["runtime"]
        )
        outcome_checks.update(route_checks)
        launcher_checks = {
            "launcher_terminal": launcher.get("collector_terminal_returncode") is True,
            "launcher_manifest_hash_matches": manifest_path.is_file()
            and launcher.get("manifest_sha256") == _sha256(manifest_path),
        }
        case["integrity_checks"].update(launcher_checks)
        case["integrity_passed"] = all(case["integrity_checks"].values())
        case["outcome_checks"] = outcome_checks
        case["outcome_passed"] = bool(outcome_checks) and all(outcome_checks.values())
        case["sealed"] = case["integrity_passed"]
        case["passed"] = case["integrity_passed"] and case["outcome_passed"]
        take = manifest.get("takes", {}).get("continue", {})
        case["metrics"].update(route_metrics)
        case["metrics"].update(
            {
                "decision_trigger_reason": take.get("decision_trigger_reason"),
                "first_attachment_step": take.get("first_attachment_step"),
                "predecision_step": take.get("predecision_step"),
                "action_branch_step": take.get("action_branch_step"),
                "legacy_collector_both_lanes_within_route_budget": collector_checks.get(
                    "both_lanes_within_route_budget"
                ),
                "legacy_collector_all_frames_non_degenerate": collector_checks.get(
                    "all_frames_non_degenerate"
                ),
            }
        )
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
    passed = integrity_passed and outcome_passed
    result = {
        "schema_version": "kinofail.o4-action-consequence-v34-heldout-postrun.v1",
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
        "passed": passed,
        "sealed": integrity_passed,
        "total_scenes": len(cases),
        "passed_scenes": passed_count,
        "room_families": sorted({case["room_family"] for case in cases}),
        "cases": cases,
        "counts_as_realistic_o4_operator_confirmation": passed,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "interpretation": (
            "This prospectively tests the unchanged v32 action architecture on a new "
            "operator-blind cohort using a preregistered phase-aware nuisance-control gate. "
            "It does not complete A0-A7."
        ),
        "next_gate": (
            "If passed, extend realistic operator coverage through O5-O11, registry-bind the "
            "corpus, retrain/infer, and rerun A0-A7 independently."
        ),
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite postrun: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "passed": passed,
                "sealed": result["sealed"],
                "passed_scenes": passed_count,
                "total_scenes": len(cases),
            },
            indent=2,
        )
    )
    return 0 if result["sealed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
