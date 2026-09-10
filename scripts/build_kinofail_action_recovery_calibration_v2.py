#!/usr/bin/env python3
"""Build the pre-formal cross-scene recovery calibration after validation v1."""

from __future__ import annotations

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

from scripts import build_kinofail_action_full_v1 as source
from scripts import build_kinofail_action_multiscene_v1 as multiscene


COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_action_full_v1.py"
BASE_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
SOURCE_PROTOCOL = ROOT / "outputs/kinofail_action_full_v1_pilot_p5/protocol.json"

# O3/O4/O8 retain the physical band from validation v1 and test only the
# geometry-robust recovery program.  O6 predeclares two less destructive doses
# because 6 N*s exceeded the recoverable region across validation-v1 scenes.
PROFILES: dict[str, tuple[tuple[str, dict[str, Any]], ...]] = {
    "O3_collapse": (("geometry_robust", multiscene.FINAL_PARAMETERS["O3_collapse"]),),
    "O4_tether": (("tangent_preserving", multiscene.FINAL_PARAMETERS["O4_tether"]),),
    "O6_push": (
        (
            "impulse_2ns",
            {
                "impulse_ns": 2.0,
                "duration_s": 0.12,
                "application_point_body_m": [0.0, -0.085, 0.075],
            },
        ),
        (
            "impulse_3ns",
            {
                "impulse_ns": 3.0,
                "duration_s": 0.12,
                "application_point_body_m": [0.0, -0.085, 0.075],
            },
        ),
        (
            "impulse_4ns",
            {
                "impulse_ns": 4.0,
                "duration_s": 0.12,
                "application_point_body_m": [0.0, -0.085, 0.075],
            },
        ),
        (
            "impulse_5ns",
            {
                "impulse_ns": 5.0,
                "duration_s": 0.12,
                "application_point_body_m": [0.0, -0.085, 0.075],
            },
        ),
    ),
    "O8_invisible_collider": (
        ("stable_backoff", multiscene.FINAL_PARAMETERS["O8_invisible_collider"]),
    ),
    "O9_high_centering": (
        (
            "ridge_0205m",
            {
                "ridge_height_m": 0.205,
                "ridge_width_m": 0.26,
                "residual_support": 0.34,
            },
        ),
        (
            "ridge_0220m",
            {
                "ridge_height_m": 0.22,
                "ridge_width_m": 0.28,
                "residual_support": 0.25,
            },
        ),
    ),
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


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile-select",
        help="optional comma-separated operator:profile subset",
    )
    parser.add_argument(
        "--protocol-id",
        default="kinofail-action-recovery-calibration-v2-20260809",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    active_profiles = PROFILES
    if args.profile_select:
        requested: dict[str, set[str]] = defaultdict(set)
        for token in args.profile_select.split(","):
            operator, separator, profile = token.strip().partition(":")
            if not separator:
                raise ValueError(f"invalid profile selector: {token}")
            requested[operator].add(profile)
        active_profiles = {
            operator: tuple(
                (profile, parameters)
                for profile, parameters in profiles
                if profile in requested.get(operator, set())
            )
            for operator, profiles in PROFILES.items()
            if operator in requested
        }
        missing = sorted(
            f"{operator}:{profile}"
            for operator, profiles in requested.items()
            for profile in profiles
            if not any(
                candidate == profile
                for candidate, _parameters in active_profiles.get(operator, ())
            )
        )
        if missing:
            raise ValueError(f"unknown profile selectors: {missing}")

    candidates, rejected = source.eligible_candidates(
        relaxed_o9_action_precursor=True
    )
    operators = tuple(active_profiles)
    selected = multiscene.select_sources(candidates, "validation", operators)
    cases: list[dict[str, Any]] = []
    for record, _replicate in selected:
        operator = str(record["target_operator"])
        for profile_id, parameters in active_profiles[operator]:
            # make_case retains a source-record reference for provenance.  A
            # deep copy is required when two parameter profiles share a source
            # episode, otherwise the later profile mutates the earlier row.
            row = multiscene.make_case(
                copy.deepcopy(record), profile_id, "validation"
            )
            row["schema_version"] = "kinofail.action-recovery-calibration-v2-schedule.v1"
            row["case_id"] = (
                f"actioncalv2__{row['scene_id']}__{operator}__{profile_id}__"
                f"{row['source_counterfactual_group_id']}"
            )
            row["severity_id"] = profile_id
            row["source_record"]["physics_parameters"] = copy.deepcopy(parameters)
            row["capability_band_parameters"] = copy.deepcopy(parameters)
            row["development_only"] = True
            row["counts_as_publication_evidence"] = False
            cases.append(row)
    expected = 12 * sum(len(rows) for rows in active_profiles.values())
    if len(cases) != expected:
        raise AssertionError((len(cases), expected))

    registry_object = source.scene_registry(
        [record for record, _replicate in selected]
    )
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

    protocol = copy.deepcopy(load(SOURCE_PROTOCOL))
    protocol.update(
        {
            "schema_version": "kinofail.action-recovery-calibration-v2-protocol.v1",
            "protocol_id": args.protocol_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "development_calibration_frozen",
            "development_only": True,
            "schedule": str(schedule),
            "schedule_sha256": sha256(schedule),
            "scene_registry": str(registry),
            "scene_registry_sha256": sha256(registry),
            "collector": str(COLLECTOR.relative_to(ROOT)),
            "collector_sha256": sha256(COLLECTOR),
            "base_collector": str(BASE_COLLECTOR.relative_to(ROOT)),
            "base_collector_sha256": sha256(BASE_COLLECTOR),
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "counts": {
                "scenes": 12,
                "operators": len(active_profiles),
                "physical_cases": len(cases),
                "action_arms": 3,
                "recovery_action_arms": 1,
                "physical_episodes": len(cases) * 3,
            },
            "action_arms": "continue, always_safe_halt, and registered action",
            "operator_to_registered_action": {
                operator: source.CORRECT_ACTION[operator] for operator in operators
            },
            "predeclared_profiles": active_profiles,
            "source_rank_contract": "development stable-key rank 0",
            "selection_reads_model_predictions": False,
            "selection_reads_action_outcomes": False,
            "unfavorable_outcomes_retained": True,
            "result_dependent_retry_permitted": False,
        }
    )
    protocol_path = output / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rejection_counts: dict[str, int] = defaultdict(int)
    for row in rejected:
        for reason in row["reasons"]:
            rejection_counts[str(reason)] += 1
    audit = {
        "schema_version": "kinofail.action-recovery-calibration-v2-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "development_only": True,
        "counts": protocol["counts"],
        "profiles": active_profiles,
        "selection_used_model_predictions_or_action_outcomes": False,
        "source_candidates_admitted": len(candidates),
        "source_candidates_rejected": len(rejected),
        "source_rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "hashes": {
            "schedule": sha256(schedule),
            "scene_registry": sha256(registry),
            "protocol": sha256(protocol_path),
            "collector": sha256(COLLECTOR),
            "base_collector": sha256(BASE_COLLECTOR),
        },
    }
    (output / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
