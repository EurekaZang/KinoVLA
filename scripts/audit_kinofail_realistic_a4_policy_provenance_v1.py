#!/usr/bin/env python3
"""Audit which locomotion policy the frozen realistic A4 collector resolved.

The A4 v5 protocol froze the collector and launcher hashes, but it did not freeze
the simulator config or policy hash.  This audit therefore distinguishes:

1. a source-resolved conclusion (what the hash-matched collector and config
   resolve to); and
2. a runtime-attested conclusion (which would require a policy hash in every
   collected result).

It never rewrites the historical protocol or results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LOAD_EXPRESSION = 'load_config("sim/go2_skeleton.yaml")'


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _artifact(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _rel(path),
        "sha256": _sha(path),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/data/kinofail_realistic_a4_actual_action_formal_v5.json",
    )
    parser.add_argument("--config", default="configs/sim/go2_skeleton.yaml")
    parser.add_argument(
        "--realistic-policy",
        default="outputs/locomotion/realistic_route_v1/policy.pt",
    )
    parser.add_argument(
        "--realistic-training-manifest",
        default="outputs/locomotion/realistic_route_v1/training_manifest.json",
    )
    parser.add_argument(
        "--realistic-policy-confirmation",
        default=(
            "outputs/kinofail_realistic/operator_confirmation/"
            "embodiedgen_realistic_route_policy_v16_unseen_batch_v1/postrun_audit.json"
        ),
    )
    parser.add_argument(
        "--results-root",
        default="outputs/kinofail_realistic/corpus_a4_actual_action_v5",
    )
    parser.add_argument(
        "--out",
        default="outputs/eval/realistic_a0_a7_v6/a4_policy_provenance_audit.json",
    )
    args = parser.parse_args()

    protocol_path = (ROOT / args.protocol).resolve()
    config_path = (ROOT / args.config).resolve()
    realistic_policy_path = (ROOT / args.realistic_policy).resolve()
    training_manifest_path = (ROOT / args.realistic_training_manifest).resolve()
    confirmation_path = (ROOT / args.realistic_policy_confirmation).resolve()
    results_root = (ROOT / args.results_root).resolve()
    out_path = (ROOT / args.out).resolve()

    protocol = json.loads(protocol_path.read_text())
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("policy_path"), str):
        raise RuntimeError("sim config must contain a string policy_path")
    collector_path = (ROOT / protocol["collector"]).resolve()
    launcher_path = (ROOT / protocol["launcher"]).resolve()
    resolved_policy_path = (ROOT / config["policy_path"]).resolve()

    collector_text = collector_path.read_text()
    result_paths = sorted(results_root.glob("*/results.jsonl"))
    if not result_paths:
        raise RuntimeError(f"no A4 results found under {results_root}")
    result_mtimes = [path.stat().st_mtime for path in result_paths]
    runtime_policy_hash_fields: list[str] = []
    result_row_count = 0
    for path in result_paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                result_row_count += 1
                row = json.loads(line)
                runtime_policy_hash_fields.extend(
                    key
                    for key in row
                    if "policy" in key.lower() and "sha" in key.lower()
                )

    training_manifest = json.loads(training_manifest_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    resolved_hash = _sha(resolved_policy_path)
    realistic_hash = _sha(realistic_policy_path)
    config_mtime = config_path.stat().st_mtime
    collector_hash_matches = _sha(collector_path) == protocol["collector_sha256"]
    launcher_hash_matches = _sha(launcher_path) == protocol["launcher_sha256"]
    source_load_is_literal = EXPECTED_LOAD_EXPRESSION in collector_text
    config_predates_all_results = config_mtime <= min(result_mtimes)
    policies_are_distinct = resolved_hash != realistic_hash
    training_manifest_hash_matches = (
        training_manifest["artifacts"]["stable_policy"]["sha256"] == realistic_hash
    )
    realistic_confirmation_passed = (
        confirmation.get("passed") is True
        and confirmation.get("policy_confirmation_state")
        == "confirmed_on_v16_failure_closed_unseen_scenes"
    )
    runtime_policy_hash_present = bool(runtime_policy_hash_fields)

    checks = {
        "collector_hash_matches_frozen_protocol": collector_hash_matches,
        "launcher_hash_matches_frozen_protocol": launcher_hash_matches,
        "collector_loads_literal_go2_skeleton_config": source_load_is_literal,
        "config_predates_all_a4_v5_result_files": config_predates_all_results,
        "config_resolves_to_existing_policy": resolved_policy_path.is_file(),
        "resolved_policy_is_legacy_default": _rel(resolved_policy_path)
        == "outputs/locomotion/policy.pt",
        "resolved_and_realistic_route_policies_are_distinct": policies_are_distinct,
        "realistic_route_training_manifest_hash_matches": training_manifest_hash_matches,
        "realistic_route_policy_has_development_confirmation": realistic_confirmation_passed,
        "a4_result_rows_contain_runtime_policy_hash": runtime_policy_hash_present,
        "a4_protocol_freezes_sim_config_hash": "sim_config_sha256" in protocol,
        "a4_protocol_freezes_policy_hash": "policy_sha256" in protocol,
    }
    source_resolved_legacy = all(
        checks[key]
        for key in (
            "collector_hash_matches_frozen_protocol",
            "launcher_hash_matches_frozen_protocol",
            "collector_loads_literal_go2_skeleton_config",
            "config_predates_all_a4_v5_result_files",
            "config_resolves_to_existing_policy",
            "resolved_policy_is_legacy_default",
            "resolved_and_realistic_route_policies_are_distinct",
        )
    )
    runtime_cryptographically_attested = (
        runtime_policy_hash_present
        and checks["a4_protocol_freezes_sim_config_hash"]
        and checks["a4_protocol_freezes_policy_hash"]
    )

    report = {
        "schema_version": "kinofail.realistic-a4-policy-provenance-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "passed": source_resolved_legacy and policies_are_distinct,
        "checks": checks,
        "artifacts": {
            "protocol": _artifact(protocol_path),
            "collector": _artifact(collector_path),
            "launcher": _artifact(launcher_path),
            "sim_config": _artifact(config_path),
            "source_resolved_policy": _artifact(resolved_policy_path),
            "realistic_route_policy": _artifact(realistic_policy_path),
            "realistic_route_training_manifest": _artifact(training_manifest_path),
            "realistic_route_development_confirmation": _artifact(confirmation_path),
            "a4_results": [_artifact(path) for path in result_paths],
        },
        "resolution_chain": {
            "collector_expression": EXPECTED_LOAD_EXPRESSION,
            "config_policy_path": config["policy_path"],
            "source_resolved_policy_sha256": resolved_hash,
            "realistic_route_policy_sha256": realistic_hash,
            "a4_result_file_count": len(result_paths),
            "a4_result_row_count": result_row_count,
        },
        "verdict": {
            "source_resolved_policy": (
                "legacy_default_policy" if source_resolved_legacy else "not_resolved"
            ),
            "source_resolved_conclusion_supported": source_resolved_legacy,
            "runtime_cryptographically_attested": runtime_cryptographically_attested,
            "confidence_boundary": (
                "High-confidence source/config provenance: the hash-matched frozen "
                "collector loaded the legacy default policy through a config that "
                "predates every A4 v5 result. The historical rows and protocol did "
                "not record/freeze the runtime policy hash, so this is not a complete "
                "cryptographic runtime attestation."
            ),
            "scientific_interpretation": (
                "A4 v5 cannot test whether the confirmed realistic-route locomotion "
                "policy supports action-conditioned recovery. Its unfavorable "
                "recovery-minus-continue result remains valid for the legacy policy "
                "stack and must not be silently overwritten."
            ),
        },
        "required_correction": {
            "preserve_historical_a4": True,
            "new_protocol_must_freeze": [
                "collector_sha256",
                "launcher_sha256",
                "sim_config_snapshot_sha256",
                "policy_sha256",
                "scene_registry_sha256",
                "schedule_sha256",
                "decision_model_sha256",
                "decision_adapter_sha256",
                "action_definitions",
                "paired_prefix_tolerance",
                "outcome_definition",
                "statistical_analysis",
            ],
            "new_c4_estimand": "selective_policy_minus_always_safe_policy",
            "development_before_confirmation": (
                "Use train/validation scenes to validate locomotion and action "
                "executability; freeze once; then collect a new untouched scene/"
                "appearance confirmation extension."
            ),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
