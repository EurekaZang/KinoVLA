"""Fail-closed readiness audit for the realistic A0--A7 replication.

The controlled A0--A7 artifacts are deliberately not inputs to this audit.  They provide the
reference hypotheses and effect sizes, but they cannot satisfy a realistic experiment gate.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "kinofail.realistic-a0-a7-readiness-audit.v2"


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _all_true(checks: Mapping[str, bool]) -> bool:
    return bool(checks) and all(value is True for value in checks.values())


def _stable_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _future_experiment_status(
    path: Path,
    *,
    a0_ready: bool,
    a0_evidence_bundle_sha256: str,
    require_model_evidence: bool,
    model_evidence_ready: bool,
    training_manifest_sha256: str | None,
    inference_manifest_sha256: str | None,
    predictions_sha256: str | None,
) -> dict[str, Any]:
    value = _json(path)
    present = path.is_file()
    completed = present and value.get("status") in {
        "confirmatory_complete",
        "confirmatory_passed",
        "publication_ready",
    }
    checks = {
        "realistic_a0_ready": a0_ready,
        "confirmatory_complete": completed,
        "realistic_dataset_scope": value.get("dataset_scope")
        == "kinofail_realistic",
        "binds_current_a0_evidence_bundle": value.get(
            "a0_evidence_bundle_sha256"
        )
        == a0_evidence_bundle_sha256,
    }
    if require_model_evidence:
        checks.update(
            {
                "realistic_model_pipeline_ready": model_evidence_ready,
                "binds_realistic_training_manifest": training_manifest_sha256
                is not None
                and value.get("training_manifest_sha256")
                == training_manifest_sha256,
                "binds_realistic_inference_manifest": inference_manifest_sha256
                is not None
                and value.get("inference_manifest_sha256")
                == inference_manifest_sha256,
                "binds_realistic_predictions": predictions_sha256 is not None
                and value.get("predictions_sha256") == predictions_sha256,
            }
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "present": present,
        "confirmatory_complete": completed,
        "reported_status": value.get("status"),
        "dataset_scope": value.get("dataset_scope"),
        "requires_model_evidence": require_model_evidence,
        "checks": checks,
        "ready": _all_true(checks),
    }


def audit_realistic_replication_readiness(
    contract: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    roots = contract["evidence_roots"]
    registry_path = _resolve(root, str(roots["scene_registry"]))
    runtime_path = _resolve(root, str(roots["runtime_audit"]))
    coverage_path = _resolve(root, str(roots["runtime_coverage"]))
    snapshot_path = (
        _resolve(root, str(roots["snapshot_development_audit"]))
        if roots.get("snapshot_development_audit")
        else None
    )
    predictions_path = _resolve(root, str(roots["predictions"]))
    training_manifest_path = _resolve(root, str(roots["training_manifest"]))
    inference_manifest_path = _resolve(root, str(roots["inference_manifest"]))
    a0_certificate_path = _resolve(
        root, str(contract["experiments"]["A0"]["output"])
    )
    registry = _json(registry_path)
    runtime = _json(runtime_path)
    coverage = _json(coverage_path)
    snapshot = _json(snapshot_path) if snapshot_path is not None else {}
    benchmark = contract["benchmark"]

    a0_evidence_bundle = {
        "contract_id": contract.get("contract_id"),
        "scene_registry_sha256": _sha256(registry_path),
        "runtime_audit_sha256": _sha256(runtime_path),
        "runtime_coverage_sha256": _sha256(coverage_path),
    }
    a0_evidence_bundle_sha256 = _stable_sha256(a0_evidence_bundle)
    a0_certificate = _json(a0_certificate_path)

    registry_coverage = registry.get("coverage", {})
    schedule_binding = registry.get("schedule_binding", {})
    observed = coverage.get("observed", {})
    a0_checks = {
        "registry_audit_passed": registry.get("passed") is True,
        "registry_publication_ready": registry.get("publication_ready") is True,
        "registry_schedule_fully_bound": schedule_binding.get("fully_bound") is True,
        "registry_has_all_domains": len(registry_coverage.get("domain_scene_counts", {}))
        == int(benchmark["domains"]),
        "registry_has_all_operators": registry_coverage.get(
            "all_11_operators_scene_admitted"
        )
        is True,
        "runtime_audit_passed": runtime.get("passed") is True,
        "runtime_publication_freeze_ready": runtime.get("publication_freeze_ready") is True,
        "eligible_episode_count": int(runtime.get("evaluation_eligible_records", -1))
        == int(benchmark["physical_episodes"]),
        "complete_pair_count": int(runtime.get("complete_counterfactual_pairs", -1))
        == int(benchmark["counterfactual_groups"]),
        "coverage_publication_ready": coverage.get("publication_ready") is True,
        "coverage_scene_families": int(observed.get("scene_families", -1))
        == int(benchmark["scene_families"]),
        "coverage_domains": int(observed.get("domains", -1)) == int(benchmark["domains"]),
        "coverage_operators": int(observed.get("operators", -1))
        == int(benchmark["operators"]),
        "coverage_camera_profiles": int(observed.get("camera_profiles", -1))
        == int(benchmark["camera_profiles"]),
        "texture_swap_runtime_pass": coverage.get("measured_texture_swap", {}).get("all_pass")
        is True,
        "a0_certificate_present": a0_certificate_path.is_file(),
        "a0_certificate_confirmatory_complete": a0_certificate.get("status")
        in {"confirmatory_complete", "confirmatory_passed", "publication_ready"},
        "a0_certificate_realistic_dataset_scope": a0_certificate.get("dataset_scope")
        == "kinofail_realistic",
        "a0_certificate_binds_current_evidence_bundle": a0_certificate.get(
            "a0_evidence_bundle_sha256"
        )
        == a0_evidence_bundle_sha256,
    }
    a0_ready = _all_true(a0_checks)

    experiments: dict[str, Any] = {
        "A0": {
            "ready": a0_ready,
            "checks": a0_checks,
            "certificate": {
                "path": str(a0_certificate_path),
                "sha256": _sha256(a0_certificate_path),
                "reported_status": a0_certificate.get("status"),
                "dataset_scope": a0_certificate.get("dataset_scope"),
            },
            "evidence_bundle": {
                **a0_evidence_bundle,
                "sha256": a0_evidence_bundle_sha256,
            },
            "observed": {
                "eligible_physical_episodes": runtime.get("evaluation_eligible_records", 0),
                "complete_counterfactual_pairs": runtime.get(
                    "complete_counterfactual_pairs", 0
                ),
                "scene_families": observed.get("scene_families", 0),
                "domains": observed.get("domains", 0),
                "operators": observed.get("operators", 0),
                "camera_profiles": observed.get("camera_profiles", 0),
            },
        }
    }
    training_manifest = _json(training_manifest_path)
    inference_manifest = _json(inference_manifest_path)
    predictions_sha256 = _sha256(predictions_path)
    training_manifest_sha256 = _sha256(training_manifest_path)
    inference_manifest_sha256 = _sha256(inference_manifest_path)
    training_checks = {
        "present": training_manifest_path.is_file(),
        "confirmatory_complete": training_manifest.get("status")
        in {"confirmatory_complete", "confirmatory_passed", "publication_ready"},
        "realistic_dataset_scope": training_manifest.get("dataset_scope")
        == "kinofail_realistic",
        "binds_current_a0_evidence_bundle": training_manifest.get(
            "a0_evidence_bundle_sha256"
        )
        == a0_evidence_bundle_sha256,
        "has_model_runs": isinstance(training_manifest.get("model_runs"), list)
        and len(training_manifest.get("model_runs", [])) > 0,
    }
    inference_checks = {
        "present": inference_manifest_path.is_file(),
        "confirmatory_complete": inference_manifest.get("status")
        in {"confirmatory_complete", "confirmatory_passed", "publication_ready"},
        "realistic_dataset_scope": inference_manifest.get("dataset_scope")
        == "kinofail_realistic",
        "binds_current_a0_evidence_bundle": inference_manifest.get(
            "a0_evidence_bundle_sha256"
        )
        == a0_evidence_bundle_sha256,
        "binds_training_manifest": training_manifest_sha256 is not None
        and inference_manifest.get("training_manifest_sha256")
        == training_manifest_sha256,
        "binds_predictions": predictions_sha256 is not None
        and inference_manifest.get("predictions_sha256") == predictions_sha256,
    }
    model_evidence_ready = (
        a0_ready
        and _all_true(training_checks)
        and _all_true(inference_checks)
        and predictions_path.is_file()
        and predictions_path.stat().st_size > 0
    )
    model_based_experiments = set(
        contract.get("claim_policy", {}).get(
            "model_based_experiments", ["A2", "A3", "A5", "A7"]
        )
    )
    for experiment_id in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        output = _resolve(root, str(contract["experiments"][experiment_id]["output"]))
        status = _future_experiment_status(
            output,
            a0_ready=a0_ready,
            a0_evidence_bundle_sha256=a0_evidence_bundle_sha256,
            require_model_evidence=experiment_id in model_based_experiments,
            model_evidence_ready=model_evidence_ready,
            training_manifest_sha256=training_manifest_sha256,
            inference_manifest_sha256=inference_manifest_sha256,
            predictions_sha256=predictions_sha256,
        )
        experiments[experiment_id] = status

    predictions = {
        "path": str(predictions_path),
        "sha256": _sha256(predictions_path),
        "present": predictions_path.is_file(),
        "nonempty": predictions_path.is_file() and predictions_path.stat().st_size > 0,
    }
    model_evidence = {
        "training_manifest": {
            "path": str(training_manifest_path),
            "sha256": training_manifest_sha256,
            "checks": training_checks,
            "ready": _all_true(training_checks),
        },
        "inference_manifest": {
            "path": str(inference_manifest_path),
            "sha256": inference_manifest_sha256,
            "checks": inference_checks,
            "ready": _all_true(inference_checks),
        },
        "predictions": predictions,
        "ready": model_evidence_ready,
    }
    snapshot_development = {
        "path": str(snapshot_path) if snapshot_path is not None else None,
        "sha256": _sha256(snapshot_path) if snapshot_path is not None else None,
        "present": snapshot_path is not None and snapshot_path.is_file(),
        "extraction_passed": snapshot.get("passed") is True,
        "counts": snapshot.get("counts", {}),
        "event_adapter_coverage": snapshot.get("event_adapter_coverage", {}),
        "publication_guard": snapshot.get("publication_guard", {}),
        "counts_as_experiment_ready": False,
        "interpretation": (
            "A passing development snapshot audit proves temporal/model-input wiring only; it "
            "does not satisfy A0 or any realistic A1-A7 confirmatory gate."
        ),
    }
    ready_ids = [key for key, value in experiments.items() if value["ready"]]
    missing_ids = [key for key, value in experiments.items() if not value["ready"]]
    outcome_counts = Counter()
    v7_path = root / (
        "outputs/kinofail_realistic/scene_sources/embodiedgen_v2/"
        "o4_route_surface_v7_confirmation_batch_v1_audit.json"
    )
    v7 = _json(v7_path)
    for attempt in v7.get("attempts", []):
        outcome_counts[str(attempt.get("outcome", "unknown"))] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": contract.get("contract_id"),
        "contract_status": contract.get("status"),
        "controlled_core_used_as_completion_evidence": False,
        "inputs": {
            "scene_registry": {"path": str(registry_path), "sha256": _sha256(registry_path)},
            "runtime_audit": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "runtime_coverage": {"path": str(coverage_path), "sha256": _sha256(coverage_path)},
            "snapshot_development_audit": {
                "path": str(snapshot_path) if snapshot_path is not None else None,
                "sha256": _sha256(snapshot_path) if snapshot_path is not None else None,
            },
        },
        "experiments": experiments,
        "predictions": predictions,
        "model_evidence": model_evidence,
        "development_milestones": {"event_aligned_snapshot_pipeline": snapshot_development},
        "ready_experiments": ready_ids,
        "missing_experiments": missing_ids,
        "n_ready": len(ready_ids),
        "n_required": 8,
        "realistic_a0_a7_complete": len(ready_ids) == 8
        and model_evidence["ready"],
        "completion_invariant": {
            "new_realistic_corpus_required": True,
            "new_training_and_inference_required": True,
            "new_a0_a7_outputs_required": True,
            "controlled_outputs_may_not_satisfy_any_gate": True,
            "a0_evidence_bundle_sha256": a0_evidence_bundle_sha256,
            "model_evidence_ready": model_evidence["ready"],
            "all_experiments_ready": len(ready_ids) == 8,
            "satisfied": len(ready_ids) == 8 and model_evidence["ready"],
        },
        "current_o4_v7": {
            "path": str(v7_path),
            "sha256": _sha256(v7_path),
            "sealed": v7.get("sealed") is True,
            "audit_integrity_passed": v7.get("audit_integrity_passed") is True,
            "formal_passes": int(v7.get("formal_v7_passes", 0)),
            "denominator": int(v7.get("requested_scene_instances", 0)),
            "primary_target_met": v7.get("primary_target_met") is True,
            "outcomes": dict(sorted(outcome_counts.items())),
            "counts_as_full_A0": False,
        },
        "publication_verdict": {
            "controlled_core_reference_evidence_available": True,
            "realistic_replication_ready": len(ready_ids) == 8
            and model_evidence["ready"],
            "kinofail_realistic_benchmark_ready": len(ready_ids) == 8
            and model_evidence["ready"],
            "sim2real_validated": False,
            "reason": (
                "Completion requires a frozen realistic corpus, a hash-bound realistic A0 "
                "certificate, new realistic training/inference/predictions, and hash-bound "
                "A1-A7 confirmatory outputs. Controlled-core results and operator demos cannot "
                "satisfy these gates."
            ),
        },
    }
