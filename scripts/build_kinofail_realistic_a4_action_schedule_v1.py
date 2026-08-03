#!/usr/bin/env python3
"""Freeze the minimal realistic A4 actual-action schedule.

Five scene clusters × two independent reset seeds × five headline operators × two
paired actions gives 100 physical rollouts and 10 seeds per operator/action cell.
The schedule reuses the already frozen severe scale-v7 operator realization in each
scene; no outcome is consulted while selecting scenes, seeds, or actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATORS = (
    "O2_compliance",
    "O4_tether",
    "O5_payload",
    "O8_invisible_collider",
    "O9_high_centering",
)
SCENES = (
    "indoor_livingroom_139",
    "indoor_office_162",
    "indoor_office_186",
    "forest_trail_metric_g03",
    "forest_river_walk_metric_g07",
)
ACTIONS = {
    "O2_compliance": "slow_high_step",
    "O4_tether": "backstep_release",
    "O5_payload": "hold_and_request",
    "O8_invisible_collider": "backstep_detour_replan",
    "O9_high_centering": "raise_body_slow_cross",
}


def _stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return 300_000_000 + int(hashlib.sha256(raw).hexdigest()[:8], 16) % 1_700_000_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/design_scale_v2/pilot_schedule.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/design_a4_actual_action_v1/schedule.jsonl",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    selected: list[dict[str, object]] = []
    for scene in SCENES:
        for operator in OPERATORS:
            candidates = [
                row
                for row in rows
                if row["scene_family"] == scene
                and row["target_operator"] == operator
                and row["condition"] == "anomaly"
            ]
            if not candidates:
                raise RuntimeError(f"missing scale-v7 source row: {scene}/{operator}")
            source_row = max(candidates, key=lambda row: int(row["severity_rank"]))
            if int(source_row["severity_rank"]) != 3 and operator != "O8_invisible_collider":
                raise RuntimeError(f"severe source unavailable: {scene}/{operator}")
            for replicate in range(2):
                case_id = f"{scene}__{operator}__r{replicate}"
                selected.append(
                    {
                        "schema_version": "kinofail.realistic-a4-actual-action-schedule.v1",
                        "case_id": case_id,
                        "scene_cluster": scene,
                        "domain": source_row["domain"],
                        "operator": operator,
                        "source_episode_id": source_row["episode_id"],
                        "source_counterfactual_group_id": source_row["counterfactual_group_id"],
                        "source_record": source_row,
                        "replicate": replicate,
                        "reset_seed": _stable_seed("a4-v1", scene, operator, replicate),
                        "actions": ["continue", ACTIONS[operator]],
                        "pairing": "same scene/operator/reset seed and byte-identical predecision policy",
                    }
                )
    if len(selected) != 50:
        raise AssertionError(len(selected))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    args.out.write_text(payload, encoding="utf-8")
    audit = {
        "schema_version": "kinofail.realistic-a4-actual-action-design-audit.v1",
        "passed": True,
        "source_schedule": str(source.relative_to(ROOT)),
        "source_schedule_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "schedule": str(args.out.resolve().relative_to(ROOT)),
        "schedule_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "operators": list(OPERATORS),
        "scene_clusters": list(SCENES),
        "independent_seeds_per_action_cell": 10,
        "physical_episode_count": 100,
        "selection_used_outcomes": False,
    }
    audit_path = args.out.with_name("design_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
