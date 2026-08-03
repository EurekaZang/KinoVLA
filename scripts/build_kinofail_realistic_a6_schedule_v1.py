#!/usr/bin/env python3
"""Build the targeted five-family, four-level realistic A6 boundary schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.realistic_benchmark import build_realistic_corpus  # noqa: E402


TARGET_SCENES = {
    "O1_mu_field": "forest_trail_metric_g03",
    "O2_compliance": "indoor_bedroom_118",
    "O4_tether": "indoor_office_162",
    "O5_payload": "forest_slope_metric_g06",
    "O10_effort_decay": "indoor_livingroom_139",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/data/kinofail_realistic.yaml")
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/kinofail_realistic/design_a6_boundary_v2")
    args = parser.parse_args()
    spec = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    domains = defaultdict(lambda: {"scenes": []})
    for row in registry["scenes"]:
        domains[row["domain"]]["scenes"].append({
            "id": row["scene_id"], "split": row["split"], "source": row["source"]
        })
    for value in domains.values():
        value["scenes"].sort(key=lambda row: ("train", "val", "test").index(row["split"]))
    spec["domains"] = dict(domains)
    spec["benchmark_id"] = "kinofail_realistic_a6_boundary_v2"
    # Keep the command-agnostic contact implementation used by the scale-v7 collector.
    o4 = next(row for row in spec["operators"] if row["id"] == "O4_tether")
    o4["nominal"] = {
        "attachment_enabled": 0.0, "tangential_force_cap_n": 0.0,
        "normal_force_cap_n": 0.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3,
    }
    o4["severities"] = [
        {"id": "mild", "rank": 1, "parameters": {"attachment_enabled": 1.0, "tangential_force_cap_n": 12.0, "normal_force_cap_n": 6.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3}},
        {"id": "moderate", "rank": 2, "parameters": {"attachment_enabled": 1.0, "tangential_force_cap_n": 18.0, "normal_force_cap_n": 8.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3}},
        {"id": "severe", "rank": 3, "parameters": {"attachment_enabled": 1.0, "tangential_force_cap_n": 24.0, "normal_force_cap_n": 10.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3}},
    ]
    full = build_realistic_corpus(spec, mode="full")
    first_realization = {
        row["id"]: row["physical_realizations"][0]
        for row in spec["operators"] if row["id"] in TARGET_SCENES
    }
    selected = [
        dict(row) for row in full.records
        if row["target_operator"] in TARGET_SCENES
        and row["scene_family"] == TARGET_SCENES[row["target_operator"]]
        and row["physical_realization"] == first_realization[row["target_operator"]]
    ]
    pair_groups = defaultdict(list)
    for row in selected:
        pair_groups[row["counterfactual_group_id"]].append(row)
    seed_cells = defaultdict(list)
    for pair_id, pair in pair_groups.items():
        representative = pair[0]
        key = (representative["target_operator"], representative["severity_id"])
        seed_cells[key].append(pair_id)
    scene_seed_index = {}
    for operator in TARGET_SCENES:
        values = sorted({
            int(row["scene_seed"]) for row in selected if row["target_operator"] == operator
        })
        if len(values) != 5:
            raise RuntimeError(f"{operator} does not have five shared scene seeds")
        scene_seed_index.update({(operator, value): index for index, value in enumerate(values)})
    for row in selected:
        row["boundary_seed_index"] = scene_seed_index[(row["target_operator"], int(row["scene_seed"]))]

    args.out.mkdir(parents=True, exist_ok=True)
    schedule = args.out / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8"
    )
    pairs = defaultdict(list)
    for row in selected:
        pairs[row["counterfactual_group_id"]].append(row)
    checks = {
        "150_records": len(selected) == 150,
        "75_pairs": len(pairs) == 75,
        "all_pairs_complete": all(len(rows) == 2 and {row["condition"] for row in rows} == {"anomaly", "nominal_counterfactual"} for rows in pairs.values()),
        "five_parameter_families": set(TARGET_SCENES) == {row["target_operator"] for row in selected},
        "three_anomaly_levels": {row["severity_id"] for row in selected} == {"mild", "moderate", "severe"},
        "five_seeds_per_operator_level": all(len(pair_ids) == 5 for pair_ids in seed_cells.values()),
        "scene_seed_paired_across_all_levels": all(
            all(
                len({
                    int(row["scene_seed"]) for row in selected
                    if row["target_operator"] == operator
                    and row["boundary_seed_index"] == seed_index
                }) == 1
                and {row["severity_id"] for row in selected
                     if row["condition"] == "anomaly"
                     and row["target_operator"] == operator
                     and row["boundary_seed_index"] == seed_index} == {"mild", "moderate", "severe"}
                for seed_index in range(5)
            )
            for operator in TARGET_SCENES
        ),
        "one_realization_per_operator": all(len({row["physical_realization"] for row in selected if row["target_operator"] == operator}) == 1 for operator in TARGET_SCENES),
        "three_domains_covered": {row["domain"] for row in selected} == {"life", "production", "wild"},
        "three_views": all(len(row["appearance_views"]) == 3 for row in selected),
        "a8_excluded": all("A8" not in row["target_operator"] for row in selected),
    }
    audit = {
        "schema_version": "kinofail.realistic-a6-boundary-design-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "records": len(selected),
        "pairs": len(pairs),
        "anomaly_levels_plus_nominal": 4,
        "operator_pair_counts": dict(sorted(Counter(row["target_operator"] for row in selected if row["condition"] == "anomaly").items())),
        "target_scene_by_operator": TARGET_SCENES,
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "source_full_design_audit_passed": full.audit["passed"],
        "statistical_unit": "independent operator seed within a measured parameter level; duplicate nominal controls from moderate/severe pairs are retained for QA but excluded from the level-0 estimator",
    }
    audit_path = args.out / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "records": len(selected), "pairs": len(pairs), "schedule": str(schedule), "sha256": _sha(schedule)}, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
