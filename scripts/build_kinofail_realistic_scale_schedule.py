#!/usr/bin/env python3
"""Bind the 396-record pilot design to nine existing realistic scene packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.realistic_benchmark import build_realistic_corpus, write_corpus_build  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/data/kinofail_realistic.yaml")
    parser.add_argument("--scene-registry", type=Path, default=ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/kinofail_realistic/design_scale_v1")
    parser.add_argument("--benchmark-id", default="kinofail_realistic_scale_v1")
    args = parser.parse_args()
    base_path = args.base_config.resolve()
    registry_path = args.scene_registry.resolve()
    spec = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    # The scale corpus uses the current command-agnostic, per-foot O4 contact model rather than
    # the legacy body-tether parameter names retained by the controlled-core design.
    o4 = next(row for row in spec["operators"] if row["id"] == "O4_tether")
    o4["nominal"] = {
        "attachment_enabled": 0.0,
        "tangential_force_cap_n": 0.0,
        "normal_force_cap_n": 0.0,
        "peel_height_m": 0.025,
        "unload_steps_to_peel": 3,
    }
    o4["severities"] = [
        {"id": "mild", "rank": 1, "parameters": {"attachment_enabled": 1.0, "tangential_force_cap_n": 12.0, "normal_force_cap_n": 6.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3}},
        {"id": "moderate", "rank": 2, "parameters": {"attachment_enabled": 1.0, "tangential_force_cap_n": 18.0, "normal_force_cap_n": 8.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3}},
        {"id": "severe", "rank": 3, "parameters": {"attachment_enabled": 1.0, "tangential_force_cap_n": 24.0, "normal_force_cap_n": 10.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3}},
    ]

    scene_checks = []
    domains: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in registry["scenes"]:
        episode = ROOT / row["episode_usd"]
        audit_path = ROOT / row["compiled_audit"]
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
        checks = {
            "episode_hash": episode.is_file() and _sha256(episode) == row["episode_sha256"],
            "audit_hash": audit_path.is_file() and _sha256(audit_path) == row["compiled_audit_sha256"],
            "audit_passed": audit.get("passed") is True,
            "scene_id": audit.get("scene_id") == row["scene_id"],
            "route_has_waypoints": len(audit.get("route", {}).get("waypoints_xy_m", [])) >= 5,
        }
        scene_checks.append({"scene_id": row["scene_id"], "checks": checks, "passed": all(checks.values())})
        domains.setdefault(row["domain"], {"scenes": []})["scenes"].append({
            "id": row["scene_id"],
            "split": row["split"],
            "source": row["source"],
        })
    for value in domains.values():
        value["scenes"].sort(key=lambda row: ("train", "val", "test").index(row["split"]))
    spec["benchmark_id"] = args.benchmark_id
    spec["domains"] = domains
    build = build_realistic_corpus(spec, mode="pilot")
    paths = write_corpus_build(build, args.out.resolve())
    schedule_scenes = {row["scene_family"] for row in build.records}
    registered_scenes = {row["scene_id"] for row in registry["scenes"]}
    operators = {row["target_operator"] for row in build.records}
    checks = {
        "all_scene_artifacts_valid": all(row["passed"] for row in scene_checks),
        "design_gate_passed": build.audit.get("passed") is True,
        "exactly_396_records": len(build.records) == 396,
        "all_registry_scenes_bound": schedule_scenes == registered_scenes,
        "three_domains": {row["domain"] for row in build.records} == {"life", "production", "wild"},
        "all_11_operators": len(operators) == 11,
        "all_three_camera_profiles": len({row["camera_profile"] for row in build.records}) == 3,
        "each_domain_has_all_three_camera_profiles": all(
            len({row["camera_profile"] for row in build.records if row["domain"] == domain}) == 3
            for domain in {row["domain"] for row in build.records}
        ),
        "a8_excluded": "A8" not in operators,
    }
    audit = {
        "schema_version": "kinofail.realistic-scale-schedule-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "scene_registry": str(registry_path),
        "scene_registry_sha256": _sha256(registry_path),
        "base_config": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "scene_checks": scene_checks,
        "records": len(build.records),
        "counterfactual_pairs": len(build.records) // 2,
        "scene_families": len(schedule_scenes),
        "operators": sorted(operators),
        "outputs": paths,
        "old_a0_a7_metrics_mixed": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    audit_path = args.out / "scale_schedule_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "passed": audit["passed"], "records": len(build.records), "audit": str(audit_path)}, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
