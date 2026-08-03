#!/usr/bin/env python3
"""Build the final model-blind A4-v8 supported-recovery schedule."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.build_kinofail_reconfirmation_a4_v7_schedule import (
    ACTIONS,
    F25_AUDIT,
    F25_SCHEDULE_ROOT,
    ROOT,
    SCALE,
    jsonl,
    load_f35_primary_decisions,
    sha256,
    stable_key,
)


OUTPUT = ROOT / "outputs/kinofail_reconfirmation_a4_v8"
CELLS = (
    ("O4_tether", "hard", "adhesion_high_step_peel"),
    ("O5_payload", "moderate", "overload_braced_safe_stop"),
)
PILOT_DESIGNS = tuple(
    [ROOT / f"outputs/kinofail_reconfirmation_a4_v7_pilot_p{i}/design_audit.json" for i in (1, 2, 3)]
    + [ROOT / f"outputs/kinofail_reconfirmation_a4_v8_pilot_p{i}/design_audit.json" for i in (4, 5)]
)
PILOT_POSTRUN = tuple(
    ROOT / f"outputs/kinofail_reconfirmation_a4_v8_pilot_p{i}/postrun_audit.json"
    for i in (4, 5)
)
PILOT_FAILURES = tuple(
    ROOT / f"outputs/kinofail_reconfirmation_a4_v7_pilot_p{i}/failure_audit.json"
    for i in (1, 2, 3)
)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (*PILOT_DESIGNS, *PILOT_POSTRUN, *PILOT_FAILURES, F25_AUDIT):
        if not path.is_file():
            raise FileNotFoundError(path)
    postrun = [json.loads(path.read_text()) for path in PILOT_POSTRUN]
    if not all(
        row.get("passed") is True
        and row.get("development_only") is True
        and row.get("counts_as_a0_a7_evidence") is False
        for row in postrun
    ):
        raise RuntimeError("A4-v8 requires passed, permanently excluded pilots")
    failures = [json.loads(path.read_text()) for path in PILOT_FAILURES]
    if not all(
        row.get("passed") is True
        and row.get("pilot_passed") is False
        and row.get("counts_as_a0_a7_evidence") is False
        for row in failures
    ):
        raise RuntimeError("A4-v8 requires preserved P1--P3 negative pilot audits")
    pilot_designs = [json.loads(path.read_text()) for path in PILOT_DESIGNS]
    excluded_scenes = {str(row["scene_id"]) for row in pilot_designs}
    if len(excluded_scenes) != 5:
        raise RuntimeError("A4-v8 requires five disjoint exposed pilot scenes")

    valid, decisions = load_f35_primary_decisions()
    anomaly = {
        str(row["counterfactual_group_id"]): row
        for row in jsonl(SCALE)
        if row["condition"] == "anomaly"
    }
    f25 = json.loads(F25_AUDIT.read_text())
    replacements = {
        str(row["original_counterfactual_group_id"]): str(
            row["replacement_counterfactual_group_id"]
        )
        for row in f25["accepted"]
        if str(row["target_operator"]) != "O9_high_centering"
    }
    replacement_rows = {
        str(row["counterfactual_group_id"]): row
        for path in sorted(F25_SCHEDULE_ROOT.glob("*/scale/schedule.jsonl"))
        for row in jsonl(path)
        if row["condition"] == "anomaly"
    }
    all_scenes = sorted({str(anomaly[group_id]["scene_id"]) for group_id in valid})
    scenes = [scene for scene in all_scenes if scene not in excluded_scenes]
    if len(all_scenes) != 30 or len(scenes) != 25:
        raise RuntimeError(
            f"A4-v8 scene partition mismatch: all={len(all_scenes)} formal={len(scenes)}"
        )

    rows = []
    for scene in scenes:
        for operator, severity, endpoint in CELLS:
            eligible = []
            for group_id in valid:
                original = anomaly[group_id]
                if (
                    original["scene_id"] == scene
                    and original["target_operator"] == operator
                    and original["severity_id"] == severity
                ):
                    values = decisions[group_id]["invariant_proprio_80"]
                    height, tilt = float(values[60]), abs(float(values[61]))
                    if height >= 0.25 and tilt < 0.55:
                        eligible.append((group_id, height, tilt))
            if not eligible:
                raise RuntimeError(f"no A4-v8 source: {scene}/{operator}/{severity}")
            group_id, height, tilt = min(
                eligible,
                key=lambda row: stable_key("a4-v8-formal", scene, operator, severity, row[0]),
            )
            original = anomaly[group_id]
            replacement_id = replacements.get(group_id)
            source = replacement_rows[replacement_id] if replacement_id else original
            nuisance = source["physical_nuisance"]
            rows.append(
                {
                    "schema_version": "kinofail.reconfirmation-a4-v8-schedule.v1",
                    "case_id": f"a4v8__{scene}__{operator}__{severity}",
                    "scene_id": scene,
                    "domain": str(original["domain"]),
                    "operator": operator,
                    "severity_id": severity,
                    "source_counterfactual_group_id": group_id,
                    "source_slot_episode_id": str(original["episode_id"]),
                    "source_physical_episode_id": str(source["episode_id"]),
                    "source_record": source,
                    "source_is_f33_direct_o9": False,
                    "source_is_f25_replacement": replacement_id is not None,
                    "reset_seed": int(nuisance["physics_seed"]),
                    "source_physical_nuisance": nuisance,
                    "source_f35_decision": decisions[group_id],
                    "actions": list(ACTIONS),
                    "operator_recovery": endpoint,
                    "pairing": "same frozen F35 source state with seven checkpoint branches",
                    "source_observation_replay_required": True,
                    "selection_used_model_predictions_or_outcomes": False,
                    "decision_state_eligibility": {
                        "last_base_height_m": height,
                        "last_absolute_tilt_rad": tilt,
                        "minimum_base_height_m": 0.25,
                        "maximum_absolute_tilt_rad": 0.55,
                        "uses_only_frozen_f35_physical_observation": True,
                    },
                    "development_only": False,
                    "counts_as_a0_a7_evidence": True,
                }
            )
    if len(rows) != 50:
        raise AssertionError(len(rows))
    if {row["scene_id"] for row in rows} & excluded_scenes:
        raise RuntimeError("exposed pilot scene leaked into A4-v8")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    schedule = OUTPUT / "schedule.jsonl"
    schedule.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    audit = {
        "schema_version": "kinofail.reconfirmation-a4-v8-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_model_blind_design_before_f36_prediction_and_a4_outcomes",
        "passed": True,
        "model_prediction_or_outcome_read": False,
        "selection_used_model_predictions_or_outcomes": False,
        "counts": {
            "source_scenes": 30,
            "excluded_pilot_scenes": 5,
            "formal_scenes": 25,
            "domains": len({row["domain"] for row in rows}),
            "operators": 2,
            "severity_strata": 2,
            "physical_cases": 50,
            "action_arms": 7,
            "physical_episodes": 350,
        },
        "supported_recovery_cells": {
            operator: {"severity": severity, "endpoint": endpoint}
            for operator, severity, endpoint in CELLS
        },
        "unsupported_actor_boundaries": {
            "O2_compliance": "P5 showed zero successful arms on a fresh hard case",
            "O8_invisible_collider": "F35 attribution decision occurs after recoverable detour timing",
            "O9_high_centering": "current actor cannot self-extract after sustained belly support",
        },
        "action_arms": list(ACTIONS),
        "excluded_pilot_scenes": sorted(excluded_scenes),
        "formal_scenes": scenes,
        "source_observation_replay": {
            "required": True,
            "window": "same 21x19 raw proprio timestamps as F35 anomaly-primary",
            "absolute_tolerance": 0.005,
            "relative_tolerance": 0.005,
            "failure_policy": "terminal case failure; no scientific retry",
        },
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule),
        "source_sha256": {
            "scale_schedule": sha256(SCALE),
            "f25_final_audit": sha256(F25_AUDIT),
            "pilot_designs": {str(path.relative_to(ROOT)): sha256(path) for path in PILOT_DESIGNS},
            "pilot_postrun": {str(path.relative_to(ROOT)): sha256(path) for path in PILOT_POSTRUN},
            "pilot_failures": {str(path.relative_to(ROOT)): sha256(path) for path in PILOT_FAILURES},
            "builder": sha256(Path(__file__).resolve()),
        },
    }
    audit_path = OUTPUT / "design_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
