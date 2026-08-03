#!/usr/bin/env python3
"""Freeze the 1,980-episode, five-seed realistic Kino-Fail replication corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
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
        default=ROOT / "outputs/kinofail_realistic/design_scale_v8_replication_v1",
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "configs/data/kinofail_realistic_scale_v8_replication_formal_v1.json",
    )
    args = parser.parse_args()
    base_path = args.base_config.resolve()
    registry_path = args.scene_registry.resolve()
    collector_path = args.collector.resolve()
    spec = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    # Use only runtime-supported first realizations.  Five stochastic seeds and two headline
    # severities provide scale without mislabelling conceptual alternate mechanisms as collected.
    spec["benchmark_id"] = "kinofail_realistic_scale_v8_replication_v1"
    spec["corpus"].update({
        "full_seeds": 5,
        "full_severities": 2,
        "full_realizations": 1,
        "expected_full_records": 1980,
    })
    for operator in spec["operators"]:
        operator["severities"] = operator["severities"][-2:]

    # Keep the admitted per-foot adhesion implementation used by scale-v7.
    o4 = next(row for row in spec["operators"] if row["id"] == "O4_tether")
    o4["nominal"] = {
        "attachment_enabled": 0.0,
        "tangential_force_cap_n": 0.0,
        "normal_force_cap_n": 0.0,
        "peel_height_m": 0.025,
        "unload_steps_to_peel": 3,
    }
    o4["severities"] = [
        {"id": "moderate", "rank": 2, "parameters": {
            "attachment_enabled": 1.0, "tangential_force_cap_n": 18.0,
            "normal_force_cap_n": 8.0, "peel_height_m": 0.025,
            "unload_steps_to_peel": 3,
        }},
        {"id": "severe", "rank": 3, "parameters": {
            "attachment_enabled": 1.0, "tangential_force_cap_n": 24.0,
            "normal_force_cap_n": 10.0, "peel_height_m": 0.025,
            "unload_steps_to_peel": 3,
        }},
    ]
    domains: dict[str, dict[str, list[dict[str, str]]]] = {}
    scene_checks = []
    for row in registry["scenes"]:
        episode = ROOT / row["episode_usd"]
        compiled = ROOT / row["compiled_audit"]
        audit = json.loads(compiled.read_text(encoding="utf-8"))
        checks = {
            "episode_hash": episode.is_file() and _sha256(episode) == row["episode_sha256"],
            "audit_hash": compiled.is_file() and _sha256(compiled) == row["compiled_audit_sha256"],
            "audit_passed": audit.get("passed") is True,
        }
        scene_checks.append({"scene_id": row["scene_id"], "checks": checks, "passed": all(checks.values())})
        domains.setdefault(row["domain"], {"scenes": []})["scenes"].append({
            "id": row["scene_id"], "split": row["split"], "source": row["source"],
        })
    for domain in domains.values():
        domain["scenes"].sort(key=lambda row: ("train", "val", "test").index(row["split"]))
    spec["domains"] = domains
    build = build_realistic_corpus(spec, mode="full")
    if not build.audit.get("passed"):
        raise RuntimeError(f"v8 replication design audit failed: {build.audit.get('issues')}")
    paths = write_corpus_build(build, args.out.resolve())
    schedule = Path(paths["schedule"])
    rows = list(build.records)
    checks = {
        "all_scene_artifacts_valid": all(row["passed"] for row in scene_checks),
        "design_gate_passed": build.audit.get("passed") is True,
        "exactly_1980_records": len(rows) == 1980,
        "exactly_990_pairs": len({row["counterfactual_group_id"] for row in rows}) == 990,
        "nine_scenes": len({row["scene_family"] for row in rows}) == 9,
        "three_domains": {row["domain"] for row in rows} == {"life", "production", "wild"},
        "all_11_operators": len({row["target_operator"] for row in rows}) == 11,
        "five_pairs_per_scene_operator_severity": all(
            sum(
                row["scene_family"] == scene
                and row["target_operator"] == operator
                and row["severity_id"] == severity
                for row in rows
            ) == 10
            for scene in {row["scene_family"] for row in rows}
            for operator in {row["target_operator"] for row in rows}
            for severity in {"moderate", "severe"}
        ),
        "moderate_and_severe": {row["severity_id"] for row in rows} == {"moderate", "severe"},
        "one_supported_realization_per_operator": all(
            len({row["physical_realization"] for row in rows if row["target_operator"] == operator}) == 1
            for operator in {row["target_operator"] for row in rows}
        ),
        "all_three_camera_profiles": len({row["camera_profile"] for row in rows}) == 3,
        "a8_excluded": all("A8" not in row["target_operator"] for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    extension_audit = {
        "schema_version": "kinofail.realistic-scale-v8-replication-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "checks": checks,
        "records": len(rows),
        "counterfactual_pairs": len(rows) // 2,
        "appearance_sequences": len(rows) * 3,
        "base_config": str(base_path.relative_to(ROOT)),
        "base_config_sha256": _sha256(base_path),
        "scene_registry": str(registry_path.relative_to(ROOT)),
        "scene_registry_sha256": _sha256(registry_path),
        "schedule_sha256": _sha256(schedule),
        "design_audit": build.audit,
        "policy": "freeze-build-run; no per-scene or per-operator outcome tuning",
    }
    audit_path = args.out / "scale_v8_replication_audit.json"
    audit_path.write_text(json.dumps(extension_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.protocol.exists():
        raise FileExistsError(f"refusing to overwrite {args.protocol}")
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail_realistic_scale_1980_v8_replication_v1",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": spec["benchmark_id"],
        "scope": "1,980 episodes / 990 pairs / 9 scenes / 11 operators / 2 severities / 5 seeds; A8 excluded.",
        "schedule_path": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha256(schedule),
        "scene_registry_path": str(registry_path.relative_to(ROOT)),
        "scene_registry_sha256": _sha256(registry_path),
        "collector_path": str(collector_path.relative_to(ROOT)),
        "collector_sha256": _sha256(collector_path),
        "runtime_manifest_path": "kino_vla/data/runtime_manifest.py",
        "runtime_manifest_sha256": _sha256(ROOT / "kino_vla/data/runtime_manifest.py"),
        "allowed": {
            "counterfactual_group_ids": sorted({row["counterfactual_group_id"] for row in rows}),
            "target_operators": sorted({row["target_operator"] for row in rows}),
            "scene_families": sorted({row["scene_family"] for row in rows}),
            "physical_realizations": sorted({row["physical_realization"] for row in rows}),
            "geometry_profiles": sorted({row["geometry_profile"] for row in rows}),
            "severity_ids": ["moderate", "severe"],
            "conditions": ["anomaly", "nominal_counterfactual"],
        },
        "collection_contract": {
            "physical_episodes": 1980,
            "counterfactual_pairs": 990,
            "appearance_views_per_episode": 3,
            "appearance_sequences": 5940,
            "scene_families": 9,
            "domains": 3,
            "operators": 11,
            "severity_levels": 2,
            "physical_seeds_per_scene_operator_severity": 5,
            "per_scene_parameter_tuning_forbidden": True,
            "outcome_strength_is_reported_not_gated": True,
            "blocking_bug_policy": "Only evidence corruption, inactive anomaly mechanisms, hash mismatch, or counterfactual leakage blocks collection.",
            "a8_in_scope": False,
        },
    }
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    args.protocol.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schedule": str(schedule), "schedule_sha256": _sha256(schedule),
        "protocol": str(args.protocol), "protocol_sha256": _sha256(args.protocol),
        "episodes": len(rows), "pairs": len(rows) // 2,
        "appearance_sequences": len(rows) * 3, "passed": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
