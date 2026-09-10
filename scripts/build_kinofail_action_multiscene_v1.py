#!/usr/bin/env python3
"""Build cross-scene validation or formal full-matrix action protocols."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_kinofail_action_full_v1 as base


SOURCE_PROTOCOL = ROOT / "outputs/kinofail_action_full_v1_pilot_p5/protocol.json"
VALIDATION_SCHEDULE = (
    ROOT / "outputs/kinofail_action_multiscene_v1_validation_v1/schedule.jsonl"
)
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_action_full_v1.py"
BASE_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
RUNNER = ROOT / "scripts/run_kinofail_action_full_v1.py"
ANALYZER = ROOT / "scripts/analyze_kinofail_action_multiscene_v1.py"


FINAL_PARAMETERS: dict[str, dict[str, Any]] = {
    "O1_mu_field": {"mu_s": 0.06, "mu_d": 0.04, "restitution": 0.0},
    "O2_compliance": {"sink_depth_m": 0.045, "shear_gain": 0.66, "stiffness_n_per_m": 120.0},
    "O3_collapse": {"damage_threshold_ns": 70.0, "drop_m": 0.030, "residual_support": 0.68},
    "O4_tether": {"attachment_enabled": 1.0, "tangential_force_cap_n": 30.0, "normal_force_cap_n": 12.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3.0},
    "O5_payload": {"mass_kg": 8.0, "com_offset_x_m": 0.08, "com_offset_y_m": 0.0},
    "O6_push": {"impulse_ns": 3.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]},
    "O7_visual_remap": {"mu_s": 0.06, "mu_d": 0.04, "depth_bias_m": 0.32},
    "O8_invisible_collider": {"collision_enabled": 1.0, "obstacle_height_m": 0.12, "optical_transmission": 0.92},
    "O9_high_centering": {"ridge_height_m": 0.19, "ridge_width_m": 0.24, "residual_support": 0.46},
    "O10_effort_decay": {"effort_floor": 0.72, "decay_rate_per_s": 0.48, "onset_s": 0.80},
    "O11_obs_bias": {"tilt_bias_rad": 0.32, "random_walk_rad_sqrt_s": 0.018, "latency_s": 0.18},
}


O9_SCENE_PARAMETERS: dict[str, dict[str, Any]] = {
    # production_004 has a higher local support surface: 0.19/0.205 m did not
    # reach the chassis, whereas 0.22 m produced strict high-centering.  The
    # remaining eleven scenes use the lower 0.19 m ridge, which preserved
    # approach motion and passed strict semantics in every development case.
    "kino4c_production_004": {
        "ridge_height_m": 0.22,
        "ridge_width_m": 0.28,
        "residual_support": 0.25,
    }
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def select_sources(
    candidates: list[dict[str, Any]], mode: str, operators: tuple[str, ...]
) -> list[tuple[dict[str, Any], str]]:
    reserved_development_sources = (
        {
            str(row["source_counterfactual_group_id"])
            for row in base.jsonl(VALIDATION_SCHEDULE)
        }
        if mode == "formal"
        else set()
    )
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_cell[(str(row["scene_cluster"]), str(row["target_operator"]))].append(row)
    scenes = sorted({scene for scene, _operator in by_cell})
    if len(scenes) != 12:
        raise RuntimeError(f"expected 12 scenes, found {len(scenes)}")
    selected: list[tuple[dict[str, Any], str]] = []
    missing: list[str] = []
    for scene in scenes:
        for operator in operators:
            rows = sorted(
                [
                    row
                    for row in by_cell.get((scene, operator), [])
                    if str(row["counterfactual_group_id"])
                    not in reserved_development_sources
                ],
                key=lambda row: base.stable_key(
                    "action-multiscene-v1",
                    scene,
                    operator,
                    row["counterfactual_group_id"],
                ),
            )
            required = (
                1
                if mode == "validation"
                else 3
                if operator in {"O4_tether", "O9_high_centering"}
                else 2
            )
            if len(rows) < required:
                missing.append(f"{scene}/{operator}:{len(rows)}<{required}")
                continue
            if mode == "validation":
                selected.append((rows[0], "validation"))
            else:
                # Development source IDs are explicitly excluded above.  The
                # first two remaining stable-key rows are formal replicates.
                # O4/O9 receive one additional predeclared rank because
                # attachment/chassis-support semantics depend on gait phase;
                # every rank and semantic attrition remains in the audit.
                selected.extend(((rows[0], "replicate_a"), (rows[1], "replicate_b")))
                if operator in {"O4_tether", "O9_high_centering"}:
                    selected.append((rows[2], "replicate_c"))
    if missing:
        raise RuntimeError("source coverage incomplete: " + ", ".join(missing[:30]))
    expected = (
        12 * len(operators)
        if mode == "validation"
        else 12
        * (
            2 * len(operators)
            + int("O4_tether" in operators)
            + int("O9_high_centering" in operators)
        )
    )
    if len(selected) != expected:
        raise AssertionError((len(selected), expected))
    return selected


def make_case(
    record: dict[str, Any], replicate: str, mode: str
) -> dict[str, Any]:
    row = base.build_case(record, "formal" if mode == "formal" else "pilot")
    operator = str(row["operator"])
    row["schema_version"] = "kinofail.action-multiscene-v1-schedule.v1"
    row["case_id"] = (
        f"actionmsv1__{mode}__{row['scene_id']}__{operator}__{replicate}__"
        f"{row['source_counterfactual_group_id']}"
    )
    row["severity_id"] = "capability_band"
    parameters = (
        O9_SCENE_PARAMETERS.get(str(row["scene_id"]), FINAL_PARAMETERS[operator])
        if operator == "O9_high_centering"
        else FINAL_PARAMETERS[operator]
    )
    row["source_record"]["physics_parameters"] = copy.deepcopy(parameters)
    row["operator_engagement_dwell_steps"] = (
        2 if operator in {"O1_mu_field", "O7_visual_remap"} else 1
    )
    if operator == "O11_obs_bias":
        row["operator_engagement_dwell_steps"] = 20
    if operator == "O4_tether":
        row["operator_engagement_dwell_steps"] = 3
    row["actions"] = (
        list(base.ACTIONS)
        if mode == "formal"
        else ["continue", "always_safe_halt", base.CORRECT_ACTION[operator]]
    )
    row["pairing"] = (
        "one exact checkpoint restored across all eleven arms"
        if mode == "formal"
        else "one exact checkpoint restored across three validation arms"
    )
    row["development_only"] = mode != "formal"
    row["counts_as_publication_evidence"] = mode == "formal"
    row["source_rank_contract"] = replicate
    row["capability_band_parameters"] = copy.deepcopy(parameters)
    if operator in {"O4_tether", "O9_high_centering"}:
        # The full-action collector replaces this placeholder with the first
        # measured attachment/chassis-contact time.  A finite value is still required by
        # the underlying collector before its dynamic-boundary branch runs.
        row["source_f35_decision"]["decision_time_s"] = 0.0
        row["source_f35_decision"]["event_time_s"] = 0.0
    if operator == "O9_high_centering":
        row["strict_o9_semantic_certificate_role"] = (
            "must be observed during the unmodified continue arm"
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validation", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--operators",
        help="optional comma-separated operator subset for development repair audits",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    operators = tuple(
        token.strip()
        for token in (args.operators or ",".join(base.OPERATORS)).split(",")
        if token.strip()
    )
    unknown = sorted(set(operators) - set(base.OPERATORS))
    if unknown or not operators:
        raise ValueError(f"invalid operator subset: {unknown or operators}")
    if args.mode == "formal" and operators != tuple(base.OPERATORS):
        raise ValueError("formal protocol must include all eleven operators")
    candidates, rejected = base.eligible_candidates(
        relaxed_o9_action_precursor=True,
        relaxed_o4_action_precursor=True,
    )
    selected = select_sources(candidates, args.mode, operators)
    cases = [make_case(row, replicate, args.mode) for row, replicate in selected]
    registry_object = base.scene_registry([row for row, _replicate in selected])
    output.mkdir(parents=True)
    schedule = output / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases),
        encoding="utf-8",
    )
    registry = output / "scene_registry.json"
    registry.write_text(
        json.dumps(registry_object, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_protocol = load(SOURCE_PROTOCOL)
    action_count = 11 if args.mode == "formal" else 3
    protocol = copy.deepcopy(source_protocol)
    subset_suffix = "" if operators == tuple(base.OPERATORS) else "-" + "-".join(operators)
    protocol.update(
        {
            "schema_version": "kinofail.action-multiscene-v1-protocol.v1",
            "protocol_id": (
                f"kinofail-action-multiscene-v1-{args.mode}{subset_suffix}-20260809"
            ),
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "formal_frozen" if args.mode == "formal" else "development_validation_frozen",
            "development_only": args.mode != "formal",
            "schedule": str(schedule),
            "schedule_sha256": sha256(schedule),
            "scene_registry": str(registry),
            "scene_registry_sha256": sha256(registry),
            "collector_sha256": sha256(COLLECTOR),
            "base_collector_sha256": sha256(BASE_COLLECTOR),
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": sha256(RUNNER),
            "analyzer": str(ANALYZER.relative_to(ROOT)),
            "analyzer_sha256": sha256(ANALYZER),
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "counts": {
                "scenes": 12,
                "operators": len(operators),
                "physical_cases": len(cases),
                "action_arms": action_count,
                "recovery_action_arms": action_count - 2,
                # The nine ontology-level recovery labels contain eight
                # numerically distinct control programs: the O5/O11
                # hold/request label and the O9 braced hold label intentionally
                # execute the same immediate safe-hold controller.  The labels
                # remain separate because their downstream recovery requests
                # and causal meanings differ.
                "unique_recovery_control_programs": (
                    8 if args.mode == "formal" else 1
                ),
                "physical_episodes": len(cases) * action_count,
            },
            "action_arms": (
                list(base.ACTIONS)
                if args.mode == "formal"
                else "continue, always_safe_halt, and registered action"
            ),
            "operator_to_registered_action": {
                operator: base.CORRECT_ACTION[operator] for operator in operators
            },
            "action_control_equivalence": {
                "recover_as_hold_request": ["O5_payload", "O11_obs_bias"],
                "recover_as_O9_high_centering": ["O9_high_centering"],
                "shared_numeric_control_program": "braced stationary hold",
                "reason_labels_remain_distinct": True,
            },
            "capability_band_parameters": {
                operator: FINAL_PARAMETERS[operator] for operator in operators
            },
            "o9_scene_parameter_overrides": O9_SCENE_PARAMETERS,
            "source_rank_contract": {
                "validation": "stable-key rank 0",
                "formal": (
                    "all validation source IDs explicitly excluded; first two "
                    "remaining stable-key rows for every operator, plus the "
                    "third remaining row for O4 and O9"
                ),
                "explicit_validation_source_id_exclusion": True,
                "selection_reads_model_predictions": False,
                "selection_reads_action_outcomes": False,
            },
            "o9_contract": {
                "branch_trigger": "first measured chassis contact with crossbar",
                "continue_arm_must_observe_strict_high_centering": True,
                "strict_semantics": "sustained belly contact plus partial foot unloading",
                "predeclared_formal_source_ranks": 3,
                "semantic_attrition_is_retained": True,
                "maximum_overall_semantic_attrition_rate": 0.05,
            },
            "o4_contract": {
                "branch_trigger": "first telemetry-proven foot attachment",
                "predeclared_formal_source_ranks": 3,
                "semantic_attrition_is_retained": True,
                "maximum_overall_semantic_attrition_rate": 0.05,
            },
            "statistical_analysis_plan": {
                "analysis_unit": "one physical case; all action arms paired at one restored checkpoint",
                "rendered_views_are_independent_observations": False,
                "primary_contrast": "registered cause-matched action versus mean mismatched recovery-program cost",
                "reference_contrasts": [
                    "registered action versus continue",
                    "registered action versus always-safe halt",
                    "registered action versus oracle minimum-cost wrong recovery",
                ],
                "primary_endpoint": "terminal_cost",
                "secondary_endpoints": [
                    "operator_recovery_success",
                    "fell",
                    "safety_exposure_auc",
                ],
                "confidence_interval": "20,000-draw scene-cluster bootstrap",
                "cost_test": "paired one-sided sign-flip randomization test",
                "success_test": "exact paired McNemar test",
                "operator_multiplicity": "Holm adjustment across eleven operators",
                "o9_required_gate": "continue arm observes strict high-centering semantics",
                "unfavorable_outcomes_retained": True,
            },
        }
    )
    protocol_path = output / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rejection_counts: dict[str, int] = defaultdict(int)
    for row in rejected:
        for reason in row["reasons"]:
            rejection_counts[str(reason)] += 1
    audit = {
        "schema_version": "kinofail.action-multiscene-v1-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "mode": args.mode,
        "development_only": args.mode != "formal",
        "selection_used_model_predictions_or_action_outcomes": False,
        "counts": protocol["counts"],
        "source_candidates_admitted": len(candidates),
        "source_candidates_rejected": len(rejected),
        "source_rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "capability_band_parameters": FINAL_PARAMETERS,
        "hashes": {
            "schedule": sha256(schedule),
            "scene_registry": sha256(registry),
            "protocol": sha256(protocol_path),
            "collector": sha256(COLLECTOR),
            "base_collector": sha256(BASE_COLLECTOR),
        },
    }
    (output / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
