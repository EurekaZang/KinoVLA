#!/usr/bin/env python3
"""Report measured coverage/randomization of evaluation-eligible Kino-Fail episodes."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eligible",
        default="outputs/kinofail_realistic/runtime_audit/formal_pilot_partial/evaluation_eligible.jsonl",
    )
    parser.add_argument(
        "--runtime-audit",
        default="outputs/kinofail_realistic/runtime_audit/formal_pilot_partial/runtime_audit.json",
    )
    parser.add_argument(
        "--corpus-root", default="outputs/kinofail_realistic/corpus_v1_formal"
    )
    parser.add_argument(
        "--out",
        default="outputs/kinofail_realistic/runtime_audit/formal_pilot_partial/coverage.json",
    )
    args = parser.parse_args()

    eligible_path = Path(args.eligible).resolve()
    audit = json.loads(Path(args.runtime_audit).read_text(encoding="utf-8"))
    corpus_root = Path(args.corpus_root).resolve()
    records = _jsonl(eligible_path)
    manifests: list[dict[str, Any]] = []
    for record in records:
        relative = Path(record["required_outputs"]["episode_manifest"])
        manifests.append(json.loads((corpus_root / relative).read_text(encoding="utf-8")))

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pairs[str(record["counterfactual_group_id"])].append(record)
    complete_pairs = {
        group_id
        for group_id, rows in pairs.items()
        if len(rows) == 2
        and {row["condition"] for row in rows}
        == {"anomaly", "nominal_counterfactual"}
    }
    paired_appearance_identical = all(
        next(row for row in rows if row["condition"] == "anomaly")["appearance_views"]
        == next(row for row in rows if row["condition"] == "nominal_counterfactual")[
            "appearance_views"
        ]
        for group_id, rows in pairs.items()
        if group_id in complete_pairs
    )

    l1_values: list[float] = []
    total_primary_frames = 0
    total_rendered_frames = 0
    total_proprio_samples = 0
    sequence_counts: list[int] = []
    for manifest in manifests:
        measured = manifest["runtime_validation"]["measured"]
        frames = int(measured["rgb_frames"])
        views = int(measured["appearance_views"])
        total_primary_frames += frames
        total_rendered_frames += frames * views
        total_proprio_samples += int(measured["proprio_shape"][0])
        sequence_counts.append(int(measured["distinct_appearance_view_sequences"]))
        l1_values.extend(float(value) for value in measured["mean_appearance_pair_rgb_l1"].values())

    material_families = {
        str(view["material_family"])
        for row in records
        for view in row["appearance_views"]
    }
    appearance_ids = {
        str(view["appearance_id"])
        for row in records
        for view in row["appearance_views"]
    }
    surface_states = {
        str(view["surface_state"])
        for row in records
        for view in row["appearance_views"]
    }
    uv_rotations = {
        float(view["uv_rotation_deg"])
        for row in records
        for view in row["appearance_views"]
    }
    uv_scales = {
        float(view["uv_scale"])
        for row in records
        for view in row["appearance_views"]
    }

    observed = {
        "physical_episodes": len(records),
        "counterfactual_pairs": len(complete_pairs),
        "appearance_view_sequences": len(records) * 3,
        "rendered_rgb_frames": total_rendered_frames,
        "primary_rgb_frames": total_primary_frames,
        "proprio_samples": total_proprio_samples,
        "operators": len({row["target_operator"] for row in records}),
        "domains": len({row["domain"] for row in records}),
        "scene_families": len({row["scene_family"] for row in records}),
        "camera_profiles": len({row["camera_profile"] for row in records}),
        "severity_ids": len({row["severity_id"] for row in records}),
        "material_families": len(material_families),
        "appearance_ids": len(appearance_ids),
        "surface_states": len(surface_states),
        "uv_rotations": len(uv_rotations),
        "uv_scales": len(uv_scales),
    }
    target = {
        "physical_episodes": int(audit["scheduled_records"]),
        "counterfactual_pairs": int(audit["counterfactual_pairs"]),
        "appearance_view_sequences": int(audit["scheduled_records"]) * 3,
        "operators": 11,
        "domains": 3,
        "scene_families": 9,
        "camera_profiles": 3,
        "severity_ids": 2,
        "material_families": 18,
    }
    coverage = {
        key: {
            "observed": observed[key],
            "target": value,
            "fraction": observed[key] / value if value else 0.0,
        }
        for key, value in target.items()
    }
    payload = {
        "schema_version": "kinofail.formal-runtime-coverage.v1",
        "publication_ready": bool(audit["publication_freeze_ready"]),
        "runtime_audit_passed": bool(audit["passed"]),
        "evaluation_eligible_records": int(audit["evaluation_eligible_records"]),
        "observed": observed,
        "coverage": coverage,
        "groups": {
            "operators": dict(sorted(Counter(row["target_operator"] for row in records).items())),
            "domains": dict(sorted(Counter(row["domain"] for row in records).items())),
            "scene_families": dict(
                sorted(Counter(row["scene_family"] for row in records).items())
            ),
            "camera_profiles": dict(
                sorted(Counter(row["camera_profile"] for row in records).items())
            ),
            "severity_ids": dict(sorted(Counter(row["severity_id"] for row in records).items())),
            "conditions": dict(sorted(Counter(row["condition"] for row in records).items())),
            "material_families": sorted(material_families),
            "surface_states": sorted(surface_states),
        },
        "counterfactual_integrity": {
            "complete_pair_count": len(complete_pairs),
            "paired_appearance_assignments_identical": paired_appearance_identical,
        },
        "measured_texture_swap": {
            "threshold": 0.015,
            "comparisons": len(l1_values),
            "min_rgb_l1": min(l1_values) if l1_values else None,
            "mean_rgb_l1": statistics.fmean(l1_values) if l1_values else None,
            "max_rgb_l1": max(l1_values) if l1_values else None,
            "all_pass": bool(l1_values) and min(l1_values) >= 0.015,
            "min_distinct_sequences_per_episode": min(sequence_counts) if sequence_counts else 0,
        },
        "next_coverage_bottlenecks": [
            "remaining_392_physical_episodes",
            "remaining_10_operators",
            "remaining_8_scene_families",
            "remaining_2_domains",
            "remaining_2_camera_profiles",
            "real_go2_fixture_anchor_not_collected",
        ],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
