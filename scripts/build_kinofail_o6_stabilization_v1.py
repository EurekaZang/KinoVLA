#!/usr/bin/env python3
"""Build O6 action-profile development and source-disjoint confirmation."""

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
from scripts import build_kinofail_action_multiscene_v1 as multi


COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_o6_stabilization_v1.py"
BASE_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
RUNNER = ROOT / "scripts/run_kinofail_action_full_v1.py"
ANALYZER = ROOT / "scripts/analyze_kinofail_action_multiscene_v1.py"
VALIDATION_SCHEDULE = ROOT / "outputs/kinofail_action_multiscene_v1_validation_v1/schedule.jsonl"
FORMAL_V4_SCHEDULE = ROOT / "outputs/kinofail_action_multiscene_v1_formal_v4/schedule.jsonl"
FORMAL_V4_ANALYSIS = ROOT / "outputs/kinofail_action_multiscene_v1_formal_v4/analysis_final/analysis.json"

DEVELOPMENT_PROFILES = (
    "O6_resume_024",
    "O6_brace_low",
    "O6_brace_deep",
    "O6_counter_low",
    "O6_counter_deep",
    "O6_catch_then_brace",
    "O6_brace_then_resume",
    "O6_directional_reflex",
)


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


def known_source_ids() -> set[str]:
    rows = base.jsonl(VALIDATION_SCHEDULE) + base.jsonl(FORMAL_V4_SCHEDULE)
    return {str(row["source_counterfactual_group_id"]) for row in rows}


def registry_sources(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates, _rejected = base.eligible_candidates(
        relaxed_o9_action_precursor=True,
        relaxed_o4_action_precursor=True,
    )
    by_id = {
        str(row["counterfactual_group_id"]): row for row in candidates
    }
    missing = sorted(
        str(row["source_counterfactual_group_id"])
        for row in cases
        if str(row["source_counterfactual_group_id"]) not in by_id
    )
    if missing:
        raise RuntimeError(f"registry sources missing: {missing[:5]}")
    return [
        by_id[str(row["source_counterfactual_group_id"])] for row in cases
    ]


def development_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for source in base.jsonl(VALIDATION_SCHEDULE):
        if source["operator"] != "O6_push":
            continue
        row = copy.deepcopy(source)
        row["schema_version"] = "kinofail.o6-stabilization-v1-schedule.v1"
        row["case_id"] = (
            f"actiono6v1__development__{row['scene_id']}__"
            f"{row['source_counterfactual_group_id']}"
        )
        row["actions"] = [
            "continue",
            "always_safe_halt",
            *(f"recover_as_{profile}" for profile in DEVELOPMENT_PROFILES),
        ]
        row["pairing"] = "one exact checkpoint restored across all nine development arms"
        row["development_only"] = True
        row["counts_as_publication_evidence"] = False
        row["source_rank_contract"] = "pre-existing rank-0 development source"
        cases.append(row)
    if len(cases) != 12 or len({row["scene_id"] for row in cases}) != 12:
        raise RuntimeError("O6 development source coverage must be exactly 12 scenes")
    return sorted(cases, key=lambda row: row["case_id"])


def confirmation_cases(profile: str) -> list[dict[str, Any]]:
    excluded = known_source_ids()
    candidates, _rejected = base.eligible_candidates(
        relaxed_o9_action_precursor=True,
        relaxed_o4_action_precursor=True,
    )
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["target_operator"] != "O6_push":
            continue
        if str(row["counterfactual_group_id"]) in excluded:
            continue
        by_scene[str(row["scene_cluster"])].append(row)
    if len(by_scene) != 12:
        raise RuntimeError(f"expected 12 O6 scenes, found {len(by_scene)}")

    actions = [
        action if action != "recover_as_O6_push" else f"recover_as_{profile}"
        for action in base.ACTIONS
    ]
    cases: list[dict[str, Any]] = []
    for scene in sorted(by_scene):
        rows = sorted(
            by_scene[scene],
            key=lambda row: base.stable_key(
                "action-o6-stabilization-confirmation-v1",
                scene,
                row["counterfactual_group_id"],
            ),
        )
        if len(rows) < 2:
            raise RuntimeError(f"insufficient untouched O6 sources: {scene}:{len(rows)}")
        for replicate, source in zip(("replicate_a", "replicate_b"), rows[:2], strict=True):
            row = multi.make_case(source, replicate, "formal")
            row["schema_version"] = "kinofail.o6-stabilization-v1-schedule.v1"
            row["case_id"] = (
                f"actiono6v1__confirmation__{row['scene_id']}__{replicate}__"
                f"{row['source_counterfactual_group_id']}"
            )
            row["actions"] = actions
            row["pairing"] = "one exact checkpoint restored across all eleven frozen arms"
            row["development_only"] = False
            row["counts_as_publication_evidence"] = True
            row["source_rank_contract"] = (
                "first two stable-key sources after explicit exclusion of rank-0 "
                "development and formal_v4 sources"
            )
            cases.append(row)
    if len(cases) != 24:
        raise AssertionError(len(cases))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=DEVELOPMENT_PROFILES)
    parser.add_argument("--development-analysis", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.mode == "confirmation" and (not args.profile or not args.development_analysis):
        raise ValueError("confirmation requires --profile and --development-analysis")

    cases = (
        development_cases()
        if args.mode == "development"
        else confirmation_cases(str(args.profile))
    )
    action_count = len(cases[0]["actions"])
    if any(len(row["actions"]) != action_count for row in cases):
        raise RuntimeError("inconsistent action-arm counts")
    registry = base.scene_registry(registry_sources(cases))
    output.mkdir(parents=True)
    schedule_path = output / "schedule.jsonl"
    schedule_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases),
        encoding="utf-8",
    )
    registry_path = output / "scene_registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_protocol = load(ROOT / "outputs/kinofail_action_multiscene_v1_formal_v4/protocol.json")
    selected_profile = str(args.profile) if args.profile else None
    registered_action = (
        f"recover_as_{selected_profile}"
        if selected_profile
        else "selected only after development analysis"
    )
    protocol = copy.deepcopy(source_protocol)
    protocol.update(
        {
            "schema_version": "kinofail.o6-stabilization-v1-protocol.v1",
            "protocol_id": f"kinofail-o6-stabilization-v3-{args.mode}-20260810",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "formal_frozen" if args.mode == "confirmation" else "development_frozen",
            "development_only": args.mode == "development",
            "schedule": str(schedule_path),
            "schedule_sha256": sha256(schedule_path),
            "scene_registry": str(registry_path),
            "scene_registry_sha256": sha256(registry_path),
            "collector": str(COLLECTOR.relative_to(ROOT)),
            "collector_sha256": sha256(COLLECTOR),
            "base_collector_sha256": sha256(BASE_COLLECTOR),
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": sha256(RUNNER),
            "analyzer": str(ANALYZER.relative_to(ROOT)),
            "analyzer_sha256": sha256(ANALYZER),
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "operators": ["O6_push"],
            "counts": {
                "scenes": 12,
                "operators": 1,
                "physical_cases": len(cases),
                "action_arms": action_count,
                "recovery_action_arms": action_count - 2,
                # The direction-conditioned program intentionally becomes
                # numerically equivalent to brace-low or nominal ride-out in a
                # given case. Seven distinct numeric programs remain required.
                "unique_recovery_control_programs": 7,
                "physical_episodes": len(cases) * action_count,
            },
            "action_arms": cases[0]["actions"],
            "operator_to_registered_action": {"O6_push": registered_action},
            "o4_contract": {"maximum_overall_semantic_attrition_rate": 0.0},
            "o9_contract": {"maximum_overall_semantic_attrition_rate": 0.0},
            "o6_recovery_contract": {
                "operator_dose_unchanged_from_formal_v4": True,
                "decision_boundary_unchanged_from_formal_v4": True,
                "endpoint_and_terminal_cost_unchanged_from_formal_v4": True,
                "development_profiles": list(DEVELOPMENT_PROFILES),
                "selected_profile": selected_profile,
                "selection_rule": (
                    "minimum fall rate, then maximum operator recovery success, "
                    "then minimum mean terminal cost across 12 development scenes"
                ),
                "formal_v4_unfavorable_o6_result_retained": True,
            },
            "source_rank_contract": {
                "development": "the pre-existing 12-scene rank-0 validation sources",
                "confirmation": (
                    "two stable-key sources per scene after explicit exclusion of "
                    "development and formal_v4 source group IDs"
                ),
                "selection_reads_confirmation_action_outcomes": False,
            },
            "statistical_analysis_plan": {
                "analysis_unit": "one physical case with paired same-checkpoint action arms",
                "primary_endpoint": "terminal_cost",
                "primary_contrast": "frozen O6 stabilization versus mean mismatched recovery-program cost",
                "reference_contrasts": [
                    "frozen O6 stabilization versus continue",
                    "frozen O6 stabilization versus always-safe halt",
                    "frozen O6 stabilization versus oracle minimum-cost wrong recovery",
                ],
                "confidence_interval": "20,000-draw scene-cluster bootstrap",
                "cost_test": "paired one-sided sign-flip randomization test",
                "success_test": "exact paired McNemar test",
                "unfavorable_outcomes_retained": True,
            },
            "formal_v4_analysis_sha256": sha256(FORMAL_V4_ANALYSIS),
            "development_analysis": (
                str(args.development_analysis.resolve()) if args.development_analysis else None
            ),
            "development_analysis_sha256": (
                sha256(args.development_analysis.resolve()) if args.development_analysis else None
            ),
        }
    )
    protocol_path = output / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    excluded = known_source_ids()
    selected_ids = {str(row["source_counterfactual_group_id"]) for row in cases}
    overlap = selected_ids & excluded if args.mode == "confirmation" else set()
    audit = {
        "schema_version": "kinofail.o6-stabilization-v1-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": not overlap,
        "mode": args.mode,
        "development_only": args.mode == "development",
        "source_overlap_with_development_or_formal_v4": sorted(overlap),
        "selection_used_confirmation_outcomes": False,
        "counts": protocol["counts"],
        "source_sha256": {
            "builder": sha256(Path(__file__).resolve()),
            "collector": sha256(COLLECTOR),
            "protocol": sha256(protocol_path),
            "schedule": sha256(schedule_path),
            "scene_registry": sha256(registry_path),
        },
    }
    (output / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
