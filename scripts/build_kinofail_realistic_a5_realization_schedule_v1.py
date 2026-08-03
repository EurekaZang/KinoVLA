#!/usr/bin/env python3
"""Freeze the supported O8/O9 alternate-physical-realization OOD batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.realistic_benchmark import build_realistic_corpus  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config", type=Path,
        default=ROOT / "configs/data/kinofail_realistic.yaml",
    )
    parser.add_argument(
        "--scene-registry", type=Path,
        default=ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
    )
    parser.add_argument(
        "--collector", type=Path,
        default=ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v6.py",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "outputs/kinofail_realistic/design_a5_realization_v1",
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "configs/data/kinofail_realistic_a5_realization_formal_v1.json",
    )
    args = parser.parse_args()
    base_path = args.base_config.resolve()
    registry_path = args.scene_registry.resolve()
    collector_path = args.collector.resolve()
    spec = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    spec["benchmark_id"] = "kinofail_realistic_a5_realization_v1"
    spec["corpus"].update({
        "full_seeds": 5, "full_severities": 2, "full_realizations": 2,
        "expected_full_records": 3960,
    })
    for operator in spec["operators"]:
        operator["severities"] = operator["severities"][-2:]
    domains: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in registry["scenes"]:
        episode = ROOT / row["episode_usd"]
        audit = ROOT / row["compiled_audit"]
        if _sha(episode) != row["episode_sha256"] or _sha(audit) != row["compiled_audit_sha256"]:
            raise RuntimeError(f"scene registry mismatch: {row['scene_id']}")
        domains.setdefault(row["domain"], {"scenes": []})["scenes"].append({
            "id": row["scene_id"], "split": row["split"], "source": row["source"],
        })
    for domain in domains.values():
        domain["scenes"].sort(key=lambda row: ("train", "val", "test").index(row["split"]))
    spec["domains"] = domains
    build = build_realistic_corpus(spec, mode="full")
    alternate = {
        "O8_invisible_collider": "occluded_low_bar",
        "O9_high_centering": "pallet_edge",
    }
    rows = [
        dict(row) for row in build.records
        if row["target_operator"] in alternate
        and row["physical_realization"] == alternate[row["target_operator"]]
    ]
    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pairs[row["counterfactual_group_id"]].append(row)
    identity_fields = (
        "domain", "scene_family", "geometry_id", "target_operator",
        "physical_realization", "severity_id", "material_family", "appearance_id",
        "surface_state", "uv_scale", "uv_rotation_deg", "camera_profile", "scene_seed",
        "operator_seed", "appearance_seed", "appearance_views",
    )
    malformed = []
    for pair_id, pair in pairs.items():
        same = all(
            len({json.dumps(row[field], sort_keys=True) for row in pair}) == 1
            for field in identity_fields
        )
        if len(pair) != 2 or {row["condition"] for row in pair} != {
            "anomaly", "nominal_counterfactual"
        } or not same:
            malformed.append(pair_id)
    checks = {
        "records_360": len(rows) == 360,
        "pairs_180": len(pairs) == 180,
        "no_malformed_pairs": not malformed,
        "scenes_9": len({row["scene_family"] for row in rows}) == 9,
        "operators_exactly_o8_o9": set(alternate) == {row["target_operator"] for row in rows},
        "alternate_realizations_exact": set(alternate.values()) == {
            row["physical_realization"] for row in rows
        },
        "two_severities": {row["severity_id"] for row in rows} == {"moderate", "severe"},
        "five_pairs_per_scene_operator_severity": all(
            sum(
                row["scene_family"] == scene
                and row["target_operator"] == operator
                and row["severity_id"] == severity
                for row in rows
            ) == 10
            for scene in {row["scene_family"] for row in rows}
            for operator in alternate
            for severity in {"moderate", "severe"}
        ),
        "a8_excluded": all("A8" not in row["target_operator"] for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    args.out.mkdir(parents=True, exist_ok=True)
    schedule = args.out / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    audit = {
        "schema_version": "kinofail.a5-realization-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True, "checks": checks,
        "records": len(rows), "counterfactual_pairs": len(pairs),
        "appearance_sequences": len(rows) * 3,
        "supported_scope": alternate,
        "unsupported_operator_realizations_not_collected": sorted(
            {row["id"] for row in spec["operators"]} - set(alternate)
        ),
        "schedule_sha256": _sha(schedule),
        "claim_boundary": (
            "This targeted physical-realization OOD axis covers O8/O9 geometry mechanisms only; "
            "it is not evidence for alternate implementations of all eleven causes."
        ),
    }
    (args.out / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.protocol.exists():
        raise FileExistsError(f"refusing to overwrite {args.protocol}")
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail_realistic_a5_realization_360_v1",
        "status": "frozen", "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": spec["benchmark_id"],
        "scope": "O8/O9 alternate physical realization: 360 episodes / 180 pairs / 9 scenes / 2 severities / 5 seeds; A8 excluded.",
        "schedule_path": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "scene_registry_path": str(registry_path.relative_to(ROOT)),
        "scene_registry_sha256": _sha(registry_path),
        "collector_path": str(collector_path.relative_to(ROOT)),
        "collector_sha256": _sha(collector_path),
        "runtime_manifest_path": "kino_vla/data/runtime_manifest.py",
        "runtime_manifest_sha256": _sha(ROOT / "kino_vla/data/runtime_manifest.py"),
        "allowed": {
            "counterfactual_group_ids": sorted(pairs),
            "target_operators": sorted(alternate),
            "scene_families": sorted({row["scene_family"] for row in rows}),
            "physical_realizations": sorted(alternate.values()),
            "geometry_profiles": sorted({row["geometry_profile"] for row in rows}),
            "severity_ids": ["moderate", "severe"],
            "conditions": ["anomaly", "nominal_counterfactual"],
        },
        "collection_contract": {
            "physical_episodes": 360, "counterfactual_pairs": 180,
            "appearance_views_per_episode": 3, "scene_families": 9,
            "operators": 2, "severity_levels": 2, "physical_seeds_per_cell": 5,
            "supported_scope": alternate,
            "per_scene_parameter_tuning_forbidden": True,
            "outcome_strength_is_reported_not_gated": True,
            "a8_in_scope": False,
        },
    }
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    args.protocol.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schedule": str(schedule), "schedule_sha256": _sha(schedule),
        "protocol": str(args.protocol), "protocol_sha256": _sha(args.protocol),
        "episodes": 360, "pairs": 180, "appearance_sequences": 1080,
        "passed": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
