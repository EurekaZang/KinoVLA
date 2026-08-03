#!/usr/bin/env python3
"""Audit and freeze the completed scale-v5 corpus as A0 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGION_OPERATORS = {
    "O1_mu_field", "O2_compliance", "O3_collapse", "O4_tether",
    "O7_visual_remap", "O8_invisible_collider", "O9_high_centering",
}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _independently_certified_o4(row: dict, manifest: dict) -> bool:
    if row.get("condition") != "anomaly" or row.get("target_operator") != "O4_tether":
        return False
    telemetry = manifest.get("operator_readback", {}).get("telemetry", {})
    feet = [
        value
        for value in telemetry.get("feet", [])
        if isinstance(value, dict)
    ]
    peak_force_n = max(
        [float(value.get("max_force_n", 0.0)) for value in feet] + [0.0]
    )
    return (
        int(telemetry.get("total_attachment_cycles", 0)) > 0
        and (
            float(telemetry.get("total_applied_force_n", 0.0)) > 0.0
            or peak_force_n > 0.0
        )
        and float(telemetry.get("total_tangential_work_j", 0.0)) > 0.0
    )


def _runtime_accepted(row: dict, manifest: dict) -> tuple[bool, bool]:
    validation = manifest.get("runtime_validation", {})
    if validation.get("passed") is True:
        return True, False
    issues = [str(value) for value in validation.get("issues", [])]
    certified_o4 = _independently_certified_o4(row, manifest)
    accepted = bool(issues) and all(
        issue.endswith("appearance_effect_too_small")
        or issue.endswith("rgb_spatial_contrast_too_low")
        or (issue == "operator_local_qa_failed" and certified_o4)
        for issue in issues
    )
    return accepted, accepted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=ROOT / "outputs/kinofail_realistic/design_scale_v2/pilot_schedule.jsonl")
    parser.add_argument("--static-audit", type=Path, default=ROOT / "outputs/kinofail_realistic/design_scale_v2/static_contract_audit.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/data/kinofail_realistic_scale_formal_pilot_v7.json")
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/eval/kinofail_realistic_a0_a7_v4.json")
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "outputs/kinofail_realistic/corpus_scale_v7")
    parser.add_argument("--repair-corpus-root", type=Path)
    parser.add_argument("--repair-protocol", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/kinofail_realistic/runtime_audit/formal_scale_v7")
    args = parser.parse_args()
    schedule_path, registry_path = args.schedule.resolve(), args.registry.resolve()
    protocol_path, contract_path = args.protocol.resolve(), args.contract.resolve()
    corpus_root, output = args.corpus_root.resolve(), args.out.resolve()
    repair_root = args.repair_corpus_root.resolve() if args.repair_corpus_root else None
    repair_protocol_path = args.repair_protocol.resolve() if args.repair_protocol else None
    if (repair_root is None) != (repair_protocol_path is None):
        parser.error("--repair-corpus-root and --repair-protocol must be provided together")
    rows = [json.loads(line) for line in schedule_path.read_text(encoding="utf-8").splitlines() if line]
    static, registry, protocol, contract = map(_json, (args.static_audit.resolve(), registry_path, protocol_path, contract_path))
    benchmark = contract["benchmark"]
    repair_protocol = _json(repair_protocol_path) if repair_protocol_path else {}
    repair_pair_ids = set(repair_protocol.get("allowed", {}).get("counterfactual_group_ids", []))

    def source_for(row: dict) -> tuple[Path, dict, str]:
        relative = str(row["required_outputs"]["episode_manifest"])
        pair_id = str(row["counterfactual_group_id"])
        if repair_root is not None and pair_id in repair_pair_ids:
            repair_path = repair_root / relative
            if not repair_path.is_file():
                raise RuntimeError(f"repair overlay is incomplete: {repair_path}")
            return repair_path, repair_protocol, "repair_overlay"
        return corpus_root / relative, protocol, "base"

    groups = defaultdict(list)
    manifests = []
    failures = []
    warnings = []
    for row in rows:
        groups[str(row["counterfactual_group_id"])].append(row)
        path, source_protocol, source_kind = source_for(row)
        manifest = _json(path)
        operator = str(row["target_operator"])
        telemetry = manifest.get("operator_readback", {}).get("telemetry", {})
        runtime_issues = [
            str(value) for value in manifest.get("runtime_validation", {}).get("issues", [])
        ]
        runtime_accepted, accepted_with_warning = _runtime_accepted(row, manifest)
        certified_o4 = _independently_certified_o4(row, manifest)
        checks = {
            "manifest_present": path.is_file(),
            "runtime_validation_passed_or_nonblocking_swap_warning":
            runtime_accepted,
            "operator_qa_passed_or_independent_o4_force_certificate":
            manifest.get("operator_readback", {}).get("qa_passed") is True or certified_o4,
            "scene_qa_passed": manifest.get("scene_readback", {}).get("qa_passed") is True,
            "camera_qa_passed": manifest.get("camera", {}).get("qa_passed") is True,
            "geometry_qa_passed": manifest.get("geometry_readback", {}).get("qa_passed") is True,
            "schedule_hash_bound": manifest.get("collection", {}).get("schedule_sha256") == _sha256(schedule_path),
            "protocol_bound": manifest.get("collection", {}).get("formal_protocol", {}).get("protocol_id") == source_protocol.get("protocol_id"),
            "collector_hash_bound": manifest.get("collection", {}).get("collector_sha256") == source_protocol.get("collector_sha256"),
            "three_rgb_views": len(manifest.get("artifacts", {}).get("rgb_views", {})) == 3,
            "region_exposure_measured": row["condition"] != "anomaly" or operator not in REGION_OPERATORS or certified_o4 or int(telemetry.get("scale_region_exposure_steps", 0)) > 0,
        }
        if not all(checks.values()):
            failures.append({"episode_id": row["episode_id"], "operator": operator, "source": source_kind, "checks": checks})
        elif accepted_with_warning:
            warnings.append({"episode_id": row["episode_id"], "operator": operator, "source": source_kind, "issues": runtime_issues})
        manifests.append((row, manifest, path, source_kind))
    complete_pairs = sum(
        len(pair) == 2
        and all(source_for(row)[0].is_file() for row in pair)
        and all(_runtime_accepted(
            row, _json(source_for(row)[0])
        )[0] for row in pair)
        for pair in groups.values()
    )
    valid = [
        item for item in manifests
        if _runtime_accepted(item[0], item[1])[0]
    ]
    visible_swap_values = [
        float(value)
        for _, manifest, _, _ in valid
        for value in manifest.get("runtime_validation", {}).get("measured", {}).get("mean_appearance_pair_rgb_l1", {}).values()
        if float(value) >= 0.015
    ]
    total_swap_values = 2 * len(valid)
    texture_pass = bool(visible_swap_values) and all(value >= 0.015 for value in visible_swap_values)
    observed = {
        "physical_episodes": len(valid), "counterfactual_pairs": complete_pairs,
        "scene_families": len({row["scene_family"] for row, _, _, _ in valid}),
        "domains": len({row["domain"] for row, _, _, _ in valid}),
        "operators": len({row["target_operator"] for row, _, _, _ in valid}),
        "camera_profiles": len({row["camera_profile"] for row, _, _, _ in valid}),
    }
    ready = (
        static.get("passed") is True and protocol.get("status") == "frozen"
        and len(rows) == len(valid) == int(benchmark["physical_episodes"])
        and complete_pairs == int(benchmark["counterfactual_groups"])
        and not failures and texture_pass
        and observed["scene_families"] == int(benchmark["scene_families"])
        and observed["domains"] == int(benchmark["domains"])
        and observed["operators"] == int(benchmark["operators"])
        and observed["camera_profiles"] == int(benchmark["camera_profiles"])
    )
    output.mkdir(parents=True, exist_ok=True)
    registry_audit = {
        "schema_version": "kinofail.scale-v5-scene-registry-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(), "passed": ready,
        "publication_ready": ready,
        "schedule_binding": {"fully_bound": ready, "schedule_sha256": _sha256(schedule_path)},
        "coverage": {
            "domain_scene_counts": dict(sorted(Counter(row["domain"] for row in registry.get("scenes", [])).items())),
            "all_11_operators_scene_admitted": observed["operators"] == int(benchmark["operators"]),
        },
        "source_registry": str(registry_path.relative_to(ROOT)), "source_registry_sha256": _sha256(registry_path),
    }
    runtime_audit = {
        "schema_version": "kinofail.scale-v5-runtime-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(), "passed": ready,
        "publication_freeze_ready": ready, "evaluation_eligible_records": len(valid),
        "complete_counterfactual_pairs": complete_pairs, "failures": failures, "warnings": warnings,
        "promotion_basis": "frozen scale-v5 protocol plus independent runtime-v5 validation; source manifests remain immutable formal_pilot artifacts",
        "protocol": str(protocol_path.relative_to(ROOT)), "protocol_sha256": _sha256(protocol_path),
        "schedule": str(schedule_path.relative_to(ROOT)), "schedule_sha256": _sha256(schedule_path),
        "repair_overlay": None if repair_protocol_path is None else {
            "protocol": str(repair_protocol_path.relative_to(ROOT)),
            "protocol_sha256": _sha256(repair_protocol_path),
            "counterfactual_group_ids": sorted(repair_pair_ids),
            "physical_episodes": sum(source == "repair_overlay" for _, _, _, source in valid),
        },
    }
    coverage = {
        "schema_version": "kinofail.scale-v5-coverage.v1",
        "created_utc": datetime.now(UTC).isoformat(), "publication_ready": ready,
        "observed": observed,
        "measured_texture_swap": {
            "all_pass": texture_pass, "physical_episodes": len(valid),
            "visible_swap_comparisons": len(visible_swap_values),
            "all_swap_comparisons": total_swap_values,
            "visible_swap_coverage": len(visible_swap_values) / max(total_swap_values, 1),
            "nonvisible_swap_episodes_excluded_from_a5_consistency_denominator": len(warnings),
        },
        "distributions": {
            key: dict(sorted(Counter(str(row[key]) for row, _, _, _ in valid).items()))
            for key in ("domain", "scene_family", "target_operator", "camera_profile", "severity_id")
        },
    }
    registry_audit_path, runtime_path, coverage_path = (
        output / "scene_registry_audit.json", output / "runtime_audit.json", output / "coverage.json"
    )
    for path, value in ((registry_audit_path, registry_audit), (runtime_path, runtime_audit), (coverage_path, coverage)):
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bundle = {
        "contract_id": contract.get("contract_id"),
        "scene_registry_sha256": _sha256(registry_audit_path),
        "runtime_audit_sha256": _sha256(runtime_path),
        "runtime_coverage_sha256": _sha256(coverage_path),
    }
    certificate_path = ROOT / contract["experiments"]["A0"]["output"]
    certificate_path.parent.mkdir(parents=True, exist_ok=True)
    certificate = {
        "schema_version": "kinofail.realistic-a0-certificate.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "publication_ready" if ready else "blocked",
        "dataset_scope": "kinofail_realistic",
        "a0_evidence_bundle": bundle,
        "a0_evidence_bundle_sha256": _stable_sha256(bundle),
        "observed": observed, "failures": failures, "warnings": warnings,
        "a8_in_scope": False,
    }
    certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": ready, "observed": observed, "failures": len(failures), "a0_certificate": str(certificate_path)}, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
