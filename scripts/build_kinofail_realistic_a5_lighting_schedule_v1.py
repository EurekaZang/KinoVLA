#!/usr/bin/env python3
"""Freeze the one-pass A5 held-out-lighting extension from scale-v7 pairs.

The extension keeps the physical seed, scene, operator, severity, camera and PBR appearance
byte-identical to the moderate scale-v7 cell.  The collector changes only authored USD light
intensities by one global factor.  There is deliberately no per-scene tuning.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:length]}"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _output_paths(row: dict, episode_id: str) -> dict:
    stem = f"heldout_dim_055/{row['domain']}/{row['target_operator']}/{episode_id}"
    return {
        "rgb": f"{stem}/rgb/",
        "rgb_views": {
            "primary": f"{stem}/rgb/",
            "swap_01": f"{stem}/rgb_views/swap_01/",
            "swap_02": f"{stem}/rgb_views/swap_02/",
        },
        "depth_optional": f"{stem}/depth/",
        "proprio": f"{stem}/proprio.npz",
        "telemetry": f"{stem}/privileged.jsonl",
        "episode_manifest": f"{stem}/manifest.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-schedule",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/design_scale_v2/pilot_schedule.jsonl",
    )
    parser.add_argument(
        "--scene-registry",
        type=Path,
        default=ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
    )
    parser.add_argument(
        "--collector",
        type=Path,
        default=ROOT / "scripts/isaac_collect_kinofail_realistic_a5_lighting_v1.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/design_a5_lighting_v1",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/data/kinofail_realistic_a5_lighting_formal_v1.json",
    )
    args = parser.parse_args()
    source = args.source_schedule.resolve()
    registry = args.scene_registry.resolve()
    collector = args.collector.resolve()
    if not collector.is_file():
        raise FileNotFoundError(collector)
    selected = [row for row in _rows(source) if row["severity_id"] == "moderate"]
    groups: dict[str, list[dict]] = {}
    for row in selected:
        groups.setdefault(str(row["counterfactual_group_id"]), []).append(row)
    if len(groups) != 99:
        raise RuntimeError(f"expected 99 moderate pairs, got {len(groups)}")

    records: list[dict] = []
    for old_group, pair in sorted(groups.items()):
        if len(pair) != 2 or {row["condition"] for row in pair} != {
            "anomaly", "nominal_counterfactual"
        }:
            raise RuntimeError(f"invalid source pair {old_group}")
        representative = pair[0]
        new_group = _stable_id(
            "cf", "a5-lighting-v1", representative["scene_family"],
            representative["target_operator"], old_group,
        )
        for raw in pair:
            row = copy.deepcopy(raw)
            episode_id = f"{new_group}_{row['condition']}"
            row.update({
                "benchmark_id": "kinofail_realistic_a5_lighting_v1",
                "counterfactual_group_id": new_group,
                "episode_id": episode_id,
                "lighting_profile": "heldout_dim_055_v1",
                "lighting_intensity_scale": 0.55,
                "lighting_intervention_only": True,
                "source_scale_v7_counterfactual_group_id": old_group,
                "source_scale_v7_episode_id": raw["episode_id"],
                "artifact_state": "planned",
                "evaluation_eligible": False,
                "required_outputs": _output_paths(row, episode_id),
            })
            records.append(row)

    checks = {
        "records_198": len(records) == 198,
        "pairs_99": len({row["counterfactual_group_id"] for row in records}) == 99,
        "scenes_9": len({row["scene_family"] for row in records}) == 9,
        "operators_11": len({row["target_operator"] for row in records}) == 11,
        "one_pair_per_scene_operator": all(
            sum(
                row["scene_family"] == scene and row["target_operator"] == operator
                for row in records
            ) == 2
            for scene in {row["scene_family"] for row in records}
            for operator in {row["target_operator"] for row in records}
        ),
        "moderate_only": {row["severity_id"] for row in records} == {"moderate"},
        "single_global_lighting_factor": {
            row["lighting_intensity_scale"] for row in records
        } == {0.55},
        "a8_excluded": all("A8" not in row["target_operator"] for row in records),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    schedule = args.out_dir / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "kinofail.a5-lighting-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "records": len(records),
        "pairs": len(records) // 2,
        "source_schedule": str(source.relative_to(ROOT)),
        "source_schedule_sha256": _sha256(source),
        "schedule_sha256": _sha256(schedule),
        "policy": "one global held-out light scale; no per-scene tuning; outcome retained",
    }
    audit_path = args.out_dir / "design_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.protocol.exists():
        raise FileExistsError(f"refusing to overwrite {args.protocol}")
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail_realistic_a5_lighting_198_v1",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": "kinofail_realistic_a5_lighting_v1",
        "scope": "A5 held-out lighting: 198 episodes / 99 pairs / 9 scenes / O1-O11; A8 excluded.",
        "schedule_path": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha256(schedule),
        "scene_registry_path": str(registry.relative_to(ROOT)),
        "scene_registry_sha256": _sha256(registry),
        "collector_path": str(collector.relative_to(ROOT)),
        "collector_sha256": _sha256(collector),
        "runtime_manifest_path": "kino_vla/data/runtime_manifest.py",
        "runtime_manifest_sha256": _sha256(ROOT / "kino_vla/data/runtime_manifest.py"),
        "allowed": {
            "counterfactual_group_ids": sorted({row["counterfactual_group_id"] for row in records}),
            "target_operators": sorted({row["target_operator"] for row in records}),
            "scene_families": sorted({row["scene_family"] for row in records}),
            "physical_realizations": sorted({row["physical_realization"] for row in records}),
            "geometry_profiles": sorted({row["geometry_profile"] for row in records}),
            "severity_ids": ["moderate"],
            "conditions": ["anomaly", "nominal_counterfactual"],
        },
        "collection_contract": {
            "physical_episodes": 198,
            "counterfactual_pairs": 99,
            "scene_families": 9,
            "operators": 11,
            "appearance_views_per_episode": 3,
            "lighting_profile": "heldout_dim_055_v1",
            "lighting_intensity_scale": 0.55,
            "no_per_scene_tuning": True,
            "blocking_bug_policy": "Only evidence corruption, inactive mechanisms, label leakage, or lighting intervention failure blocks collection.",
            "a8_in_scope": False,
        },
    }
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    args.protocol.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "schedule": str(schedule), "schedule_sha256": _sha256(schedule),
        "protocol": str(args.protocol), "protocol_sha256": _sha256(args.protocol),
        "pairs": 99, "episodes": 198, "passed": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
