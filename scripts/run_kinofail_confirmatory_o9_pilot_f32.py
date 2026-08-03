#!/usr/bin/env python3
"""Run and audit the frozen F32 direct-contact O9 mechanism pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.data.o9_semantics import evaluate_o9_high_centering
from scripts.run_kinofail_t3_replenishment_f28 import (
    CONDA_PREFIX,
    EXPERIENCE,
    ISAACLAB,
    PYTHONPATH,
    _terminate_group,
    atomic_json,
    read_json,
    read_jsonl,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/kinofail_confirmatory_o9_pilot_f32"
ALLOWED_VISUAL_ISSUES = (
    "appearance_effect_too_small",
    "rgb_spatial_contrast_too_low",
)


def allowed_runtime_issues(values: list[Any]) -> bool:
    return all(any(str(value).endswith(suffix) for suffix in ALLOWED_VISUAL_ISSUES) for value in values)


def validate_seal(path: Path) -> dict[str, Any]:
    seal = read_json(path)
    sidecar = path.with_name("seal_manifest.sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text().split()[0] != sha256(path)
        or seal.get("status") != "sealed_before_collection"
        or seal.get("passed") is not True
        or seal.get("counterfactual_pairs") != 12
        or seal.get("model_prediction_feature_label_or_score_read") is not False
        or seal.get("result_dependent_retry") is not False
    ):
        raise RuntimeError("invalid F32 pilot seal")
    for key in ("schedule", "protocol", "design", "scene_registry", "material_lock", "semantic_module"):
        artifact = ROOT / str(seal[key])
        if not artifact.is_file() or sha256(artifact) != seal[f"{key}_sha256"]:
            raise RuntimeError(f"F32 sealed artifact drift: {artifact}")
    for row in seal["dependencies"]:
        artifact = Path(str(row["path"]))
        if not artifact.is_file() or sha256(artifact) != row["sha256"]:
            raise RuntimeError(f"F32 dependency drift: {artifact}")
    return seal


def audit_pair(pair_id: str, corpus: Path) -> tuple[bool, list[str], dict[str, Any]]:
    summary_path = corpus / "pair_summaries" / f"{pair_id}.json"
    issues: list[str] = []
    evidence: dict[str, Any] = {"summary": str(summary_path), "summary_present": summary_path.is_file()}
    if not summary_path.is_file():
        return False, ["summary_missing"], evidence
    summary = read_json(summary_path)
    results = list(summary.get("results", []))
    by_condition = {str(row.get("condition")): row for row in results}
    if len(results) != 2 or set(by_condition) != {"nominal_counterfactual", "anomaly"}:
        return False, ["summary_incomplete"], evidence
    manifests: dict[str, dict[str, Any]] = {}
    nuisances: list[dict[str, Any]] = []
    runtime_issues: dict[str, list[Any]] = {}
    for condition, result in by_condition.items():
        manifest_path = Path(str(result.get("manifest", "")))
        if not manifest_path.is_file():
            issues.append(f"manifest_missing:{condition}")
            continue
        manifest = read_json(manifest_path)
        manifests[condition] = manifest
        formal = manifest.get("collection", {}).get("formal_protocol", {})
        if formal.get("protocol_id") != "kinofail-confirmatory-o9-direct-pilot-f32":
            issues.append(f"protocol_mismatch:{condition}")
        nuisance = manifest.get("collection", {}).get("physical_nuisance")
        if not isinstance(nuisance, dict):
            issues.append(f"nuisance_missing:{condition}")
        else:
            nuisances.append(nuisance)
        observed = list(manifest.get("runtime_validation", {}).get("issues", []))
        runtime_issues[condition] = observed
        if not allowed_runtime_issues(observed):
            issues.extend(f"runtime:{condition}:{value}" for value in observed)
    nominal = by_condition["nominal_counterfactual"]
    if nominal.get("fallen") is not False:
        issues.append("nominal_fell")
    if float(nominal.get("max_route_deviation_m", 1.0e9)) > 0.55:
        issues.append("nominal_left_audited_route")
    semantic: dict[str, Any] = {}
    anomaly_manifest = manifests.get("anomaly", {})
    if anomaly_manifest:
        semantic = evaluate_o9_high_centering(
            anomaly_manifest.get("operator_readback", {}).get("telemetry", {})
        )
        if not semantic["passed"]:
            issues.append("direct_o9_semantic_gate_failed")
    if len(nuisances) != 2 or nuisances[0] != nuisances[1]:
        issues.append("counterfactual_nuisance_mismatch")
    evidence.update(
        {
            "summary_sha256": sha256(summary_path),
            "nominal_fallen": nominal.get("fallen"),
            "nominal_max_route_deviation_m": nominal.get("max_route_deviation_m"),
            "anomaly_semantic": semantic,
            "runtime_issues": runtime_issues,
            "only_prespecified_visual_runtime_issues_ignored": True,
            "pair_shared_nuisance": nuisances[0] if len(nuisances) == 2 else None,
        }
    )
    return not issues, sorted(set(issues)), evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", type=Path, default=OUTPUT / "seal_manifest.json")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()
    if args.max_workers != 3:
        raise ValueError("F32 pilot is frozen to three concurrent Isaac processes")
    seal_path = args.seal.resolve()
    seal = validate_seal(seal_path)
    schedule = ROOT / str(seal["schedule"])
    protocol = ROOT / str(seal["protocol"])
    registry = ROOT / str(seal["scene_registry"])
    asset_lock = ROOT / str(seal["material_lock"])
    corpus = Path(str(seal["corpus_root"]))
    corpus.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(schedule)
    pair_ids = sorted({str(row["counterfactual_group_id"]) for row in rows})
    if len(pair_ids) != 12:
        raise RuntimeError("F32 pilot schedule does not contain 12 pairs")
    attempts_root = OUTPUT / "attempts"
    logs_root = OUTPUT / "logs"
    attempts_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CONDA_PREFIX": str(CONDA_PREFIX),
            "PATH": f"{CONDA_PREFIX / 'bin'}:{environment.get('PATH', '')}",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONPATH": PYTHONPATH,
            "TERM": "xterm-256color",
            "HF_HUB_OFFLINE": "1",
        }
    )
    lock = threading.Lock()

    def run_one(pair_id: str) -> dict[str, Any]:
        attempt_path = attempts_root / f"{pair_id}.json"
        if attempt_path.exists():
            raise RuntimeError(f"F32 forbids retry or overwrite: {pair_id}")
        log_path = logs_root / f"{pair_id}.log"
        started = {
            "schema_version": "kinofail.confirmatory-o9-pilot-f32-attempt.v1",
            "state": "started",
            "started_utc": datetime.now(UTC).isoformat(),
            "pair_id": pair_id,
            "scientific_collection_attempt_index": 1,
            "retry_authorized": False,
            "result_dependent_retry": False,
            "model_prediction_feature_or_score_read": False,
            "log": str(log_path),
        }
        atomic_json(attempt_path, started)
        command = [
            str(ISAACLAB),
            "-p",
            str(ROOT / "scripts/isaac_collect_kinofail_confirmatory_o9_direct_v1.py"),
            "--schedule",
            str(schedule),
            "--scene-registry",
            str(registry),
            "--protocol",
            str(protocol),
            "--asset-lock",
            str(asset_lock),
            "--corpus-root",
            str(corpus),
            "--counterfactual-group-id",
            pair_id,
            "--headless",
            "--enable_cameras",
            "--experience",
            str(EXPERIENCE),
        ]
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started["pid"] = process.pid
            atomic_json(attempt_path, started)
            timed_out = False
            try:
                returncode = process.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_group(process.pid)
                returncode = process.wait()
        accepted, audit_issues, evidence = audit_pair(pair_id, corpus)
        terminal = {
            **started,
            "state": "terminal",
            "completed_utc": datetime.now(UTC).isoformat(),
            "returncode": int(returncode),
            "timed_out": timed_out,
            "accepted_for_mechanism_pilot": accepted,
            "audit_issues": audit_issues,
            "evidence": evidence,
        }
        atomic_json(attempt_path, terminal)
        with lock:
            print(json.dumps({"pair": pair_id, "returncode": returncode, "accepted": accepted, "issues": audit_issues}, sort_keys=True), flush=True)
        return terminal

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        for index, pair_id in enumerate(pair_ids):
            futures.append(pool.submit(run_one, pair_id))
            if index < 2:
                time.sleep(8.0)
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            atomic_json(
                OUTPUT / "supervisor_state.json",
                {
                    "schema_version": "kinofail.confirmatory-o9-pilot-f32-supervisor.v1",
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "state": "running" if len(results) < len(pair_ids) else "finalizing",
                    "terminal_pairs": len(results),
                    "accepted_pairs": sum(row["accepted_for_mechanism_pilot"] for row in results),
                    "planned_pairs": len(pair_ids),
                    "maximum_concurrent_isaac_processes": 3,
                },
            )
    accepted = [row for row in results if row["accepted_for_mechanism_pilot"]]
    all_terminal = len(results) == 12 and all(row["state"] == "terminal" for row in results)
    audit = {
        "schema_version": "kinofail.confirmatory-o9-pilot-f32-final-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "final",
        "passed": all_terminal and len(accepted) == 12,
        "model_prediction_feature_label_or_score_read": False,
        "pilot_pairs_permanently_excluded_from_final_confirmation": True,
        "allowed_nonphysical_runtime_issue_suffixes": list(ALLOWED_VISUAL_ISSUES),
        "counts": {
            "scheduled_pairs": 12,
            "terminal_pairs": len(results),
            "direct_semantic_and_nominal_stability_passed_pairs": len(accepted),
            "rejected_pairs": len(results) - len(accepted),
        },
        "gates": {
            "all_frozen_pairs_terminal": all_terminal,
            "all_twelve_mechanism_and_nominal_checks_pass": len(accepted) == 12,
        },
        "seal_sha256": sha256(seal_path),
        "results": sorted(results, key=lambda row: row["pair_id"]),
    }
    atomic_json(OUTPUT / "final_audit.json", audit)
    atomic_json(
        OUTPUT / "supervisor_state.json",
        {
            "schema_version": "kinofail.confirmatory-o9-pilot-f32-supervisor.v1",
            "updated_utc": datetime.now(UTC).isoformat(),
            "state": "completed" if audit["passed"] else "gate_failed",
            "terminal_pairs": len(results),
            "accepted_pairs": len(accepted),
            "planned_pairs": 12,
            "maximum_concurrent_isaac_processes": 3,
        },
    )
    print(json.dumps({"final_audit": str(OUTPUT / "final_audit.json"), "passed": audit["passed"], "counts": audit["counts"]}, indent=2), flush=True)
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
