#!/usr/bin/env python3
"""Seal the frozen v32 early-branch O4 development cohort."""

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


def _capture_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    captures = summary.get("captures", [])
    if not isinstance(captures, list):
        return {}
    mapped = {
        capture.get("label"): capture
        for capture in captures
        if isinstance(capture, dict) and isinstance(capture.get("label"), str)
    }
    return mapped if len(mapped) == len(captures) else {}


def _strict_frame_passes(capture: dict[str, Any], contract: dict[str, Any]) -> bool:
    metrics = capture.get("metrics", {})
    return (
        isinstance(metrics, dict)
        and metrics.get("std_luminance", -1.0)
        >= contract["strict_minimum_std_luminance"]
        and metrics.get("black_fraction", 1.0)
        < contract["strict_maximum_black_fraction"]
        and metrics.get("quantized_color_count_5bit", -1)
        >= contract["strict_minimum_quantized_color_count_5bit"]
        and contract["minimum_mean_luminance"]
        <= metrics.get("mean_luminance", -1.0)
        <= contract["maximum_mean_luminance"]
    )


def _fallen_frame_passes(capture: dict[str, Any], contract: dict[str, Any]) -> bool:
    metrics = capture.get("metrics", {})
    return (
        capture.get("fallen") is True
        and isinstance(metrics, dict)
        and metrics.get("black_fraction", 1.0)
        < contract["fallen_continue_outcome_maximum_black_fraction"]
        and metrics.get("quantized_color_count_5bit", -1)
        >= contract["fallen_continue_outcome_minimum_quantized_color_count_5bit"]
        and contract["minimum_mean_luminance"]
        <= metrics.get("mean_luminance", -1.0)
        <= contract["maximum_mean_luminance"]
    )


def _state_aware_visual_checks(
    manifest: dict[str, Any], contract: dict[str, Any]
) -> dict[str, bool]:
    takes = manifest.get("takes", {})
    continue_summary = takes.get("continue", {})
    backstep_summary = takes.get("backstep", {})
    continue_captures = _capture_map(continue_summary)
    backstep_captures = _capture_map(backstep_summary)
    required = {"initial", "decision", "outcome"}
    checks = {
        "continue_has_exact_capture_labels": set(continue_captures) == required,
        "backstep_has_exact_capture_labels": set(backstep_captures) == required,
    }
    checks.update(
        {
            f"continue_{label}_strict_visual_quality": _strict_frame_passes(
                continue_captures.get(label, {}), contract
            )
            for label in ("initial", "decision")
        }
    )
    checks.update(
        {
            f"backstep_{label}_strict_visual_quality": _strict_frame_passes(
                backstep_captures.get(label, {}), contract
            )
            for label in ("initial", "decision", "outcome")
        }
    )
    checks["fallen_continue_outcome_sensor_valid"] = (
        continue_summary.get("fell") is True
        and _fallen_frame_passes(continue_captures.get("outcome", {}), contract)
    )
    return checks


def _early_branch_checks(manifest: dict[str, Any], runtime: dict[str, Any]) -> dict[str, bool]:
    takes = manifest.get("takes", {})
    summaries = [takes.get(name, {}) for name in ("continue", "backstep")]
    predecision_steps = [summary.get("predecision_step") for summary in summaries]
    first_attachment_steps = [summary.get("first_attachment_step") for summary in summaries]
    return {
        "both_takes_branch_one_step_after_attachment": all(
            isinstance(pre, int)
            and isinstance(first, int)
            and pre == first + 1
            for pre, first in zip(predecision_steps, first_attachment_steps)
        ),
        "both_takes_use_maximum_dwell_trigger": all(
            summary.get("decision_trigger_reason") == "maximum_dwell"
            for summary in summaries
        ),
        "decision_steps_match_between_takes": len(set(predecision_steps)) == 1
        and len(set(first_attachment_steps)) == 1,
        "manifest_contract_uses_one_step_dwell": manifest.get("decision_contract", {}).get(
            "minimum_decision_dwell_steps"
        )
        == runtime["minimum_decision_dwell_steps"]
        == 1
        and manifest.get("decision_contract", {}).get("maximum_decision_dwell_steps")
        == runtime["maximum_decision_dwell_steps"]
        == 1,
    }


def _horizon_contract_checks(
    manifest: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, bool]:
    decision = manifest.get("decision_contract", {})
    expected = (
        f"reverse {runtime['backstep_speed_mps']} m/s until retreat reaches "
        f"{runtime['target_recovery_distance_m']} m and active adhesion remains clear for "
        f"{runtime['adhesion_clearance_dwell_steps']} steps, capped at "
        f"{runtime['maximum_backstep_steps']} steps, then hold"
    )
    return {
        "backstep_contract_allows_full_horizon": runtime["maximum_backstep_steps"]
        == runtime["horizon_steps"]
        == 300,
        "manifest_backstep_contract_exact": decision.get("backstep_action") == expected,
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
        nonvisual_checks = {
            key: value
            for key, value in collector_checks.items()
            if key != "all_frames_non_degenerate"
        }
        replacement_outcome_checks = {
            key: value
            for key, value in case["outcome_checks"].items()
            if key != "collector_reported_all_checks_passed"
        }
        replacement_outcome_checks["collector_nonvisual_checks_pass"] = (
            bool(nonvisual_checks) and all(value is True for value in nonvisual_checks.values())
        )
        replacement_outcome_checks.update(
            _early_branch_checks(manifest, config["runtime"])
        )
        replacement_outcome_checks.update(
            _horizon_contract_checks(manifest, config["runtime"])
        )
        visual_checks = _state_aware_visual_checks(
            manifest, config["visual_qa_contract"]
        )
        replacement_outcome_checks.update(visual_checks)
        launcher_checks = {
            "launcher_terminal": launcher.get("collector_terminal_returncode") is True,
            "launcher_manifest_hash_matches": manifest_path.is_file()
            and launcher.get("manifest_sha256") == _sha256(manifest_path),
        }
        case["integrity_checks"].update(launcher_checks)
        case["integrity_passed"] = all(case["integrity_checks"].values())
        case["outcome_checks"] = replacement_outcome_checks
        case["outcome_passed"] = bool(replacement_outcome_checks) and all(
            replacement_outcome_checks.values()
        )
        case["sealed"] = case["integrity_passed"]
        case["passed"] = case["integrity_passed"] and case["outcome_passed"]
        take = manifest.get("takes", {}).get("continue", {})
        case["metrics"].update(
            {
                "decision_trigger_reason": take.get("decision_trigger_reason"),
                "first_attachment_step": take.get("first_attachment_step"),
                "predecision_step": take.get("predecision_step"),
                "action_branch_step": take.get("action_branch_step"),
                "collector_all_frames_non_degenerate": collector_checks.get(
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
    result = {
        "schema_version": "kinofail.o4-action-consequence-v32-development-postrun.v1",
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
            "This seals one-shot v32 architecture development on the complete v33 cohort. "
            "It is neither held-out O4 confirmation nor A0-A7 evidence."
        ),
        "next_gate": (
            "If the frozen v32 architecture passes, acquire a completely new operator-blind "
            "cohort and execute one unchanged held-out confirmation. Never rerun these scenes."
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
                "passed": result["passed"],
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
