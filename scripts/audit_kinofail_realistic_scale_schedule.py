#!/usr/bin/env python3
"""Fail-closed static audit before spending GPU hours on a scale schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    schedule_path = args.schedule.resolve()
    registry_path = args.registry.resolve()
    rows = [json.loads(line) for line in schedule_path.read_text(encoding="utf-8").splitlines() if line]
    registry = _json(registry_path)
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["counterfactual_group_id"])].append(row)
    scene_ids = {str(row["scene_id"]) for row in registry["scenes"]}
    domains = {str(row["domain"]) for row in rows}
    operators = {str(row["target_operator"]) for row in rows}
    cameras = {str(row["camera_profile"]) for row in rows}
    outputs = [str(row["required_outputs"]["episode_manifest"]) for row in rows]
    pair_valid = all(
        len(pair) == 2
        and {str(row["condition"]) for row in pair} == {"anomaly", "nominal_counterfactual"}
        and len({str(row["target_operator"]) for row in pair}) == 1
        and len({str(row["scene_family"]) for row in pair}) == 1
        and len({str(row["camera_profile"]) for row in pair}) == 1
        and len({json.dumps(row["appearance_views"], sort_keys=True) for row in pair}) == 1
        for pair in groups.values()
    )
    artifact_checks = []
    for scene in registry["scenes"]:
        episode = ROOT / scene["episode_usd"]
        compiled = ROOT / scene["compiled_audit"]
        artifact_checks.append(
            episode.is_file() and compiled.is_file()
            and _sha256(episode) == scene["episode_sha256"]
            and _sha256(compiled) == scene["compiled_audit_sha256"]
            and _json(compiled).get("passed") is True
        )
    checks = {
        "records_396": len(rows) == 396,
        "pairs_198": len(groups) == 198,
        "all_pairs_complete_and_nuisance_matched": pair_valid,
        "required_outputs_unique": len(outputs) == len(set(outputs)),
        "three_domains": len(domains) == 3,
        "nine_registry_scenes": {str(row["scene_family"]) for row in rows} == scene_ids and len(scene_ids) == 9,
        "eleven_operators": len(operators) == 11,
        "all_three_camera_profiles": cameras == {"go2_front_calib_a", "go2_front_calib_b", "go2_front_calib_c"},
        "each_domain_covers_all_cameras": all(
            {str(row["camera_profile"]) for row in rows if row["domain"] == domain} == cameras
            for domain in domains
        ),
        "each_operator_covers_all_domains": all(
            {str(row["domain"]) for row in rows if row["target_operator"] == operator} == domains
            for operator in operators
        ),
        "each_operator_has_two_severities": all(
            len({str(row["severity_id"]) for row in rows if row["target_operator"] == operator}) == 2
            for operator in operators
        ),
        "three_appearance_views_each": all(len(row.get("appearance_views", [])) == 3 for row in rows),
        "all_registry_artifacts_hash_valid": all(artifact_checks),
        "a8_excluded": all("A8" not in str(row) for row in rows),
    }
    audit = {
        "schema_version": "kinofail.realistic-scale-static-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "schedule": str(schedule_path.relative_to(ROOT)),
        "schedule_sha256": _sha256(schedule_path),
        "registry": str(registry_path.relative_to(ROOT)),
        "registry_sha256": _sha256(registry_path),
        "counts": {
            "records": len(rows), "pairs": len(groups), "domains": len(domains),
            "scenes": len(scene_ids), "operators": len(operators), "camera_profiles": len(cameras),
        },
        "distributions": {
            key: dict(sorted(Counter(str(row[key]) for row in rows).items()))
            for key in ("domain", "scene_family", "target_operator", "camera_profile", "severity_id")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
