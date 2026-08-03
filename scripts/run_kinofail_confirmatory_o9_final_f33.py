#!/usr/bin/env python3
"""Unattended three-process runner for the frozen F33 O9 confirmation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.data.o9_pair_alignment import audit_matched_horizon
from kino_vla.data.o9_semantics import evaluate_o9_high_centering
from scripts.run_kinofail_t3_replenishment_f28 import (
    CONDA_PREFIX,
    EXPERIENCE,
    ISAACLAB,
    PYTHONPATH,
    _pid_matches,
    _terminate_group,
    atomic_json,
    read_json,
    read_jsonl,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/kinofail_confirmatory_o9_final_f33"
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
        or seal.get("counterfactual_pairs") != 960
        or seal.get("minimum_accepted_pairs") != 750
        or seal.get("model_prediction_feature_label_outcome_or_score_read") is not False
        or seal.get("result_dependent_retry") is not False
    ):
        raise RuntimeError("invalid F33 seal")
    for key in (
        "schedule",
        "protocol",
        "design",
        "scene_registry",
        "material_lock",
        "semantic_module",
        "pilot_seal",
        "pilot_audit",
    ):
        artifact = ROOT / str(seal[key])
        if not artifact.is_file() or sha256(artifact) != seal[f"{key}_sha256"]:
            raise RuntimeError(f"F33 sealed artifact drift: {artifact}")
    for row in seal["dependencies"]:
        artifact = Path(str(row["path"]))
        if not artifact.is_file() or sha256(artifact) != row["sha256"]:
            raise RuntimeError(f"F33 dependency drift: {artifact}")
    return seal


def audit_pair(
    pair_id: str,
    corpus: Path,
    source_by_pair: dict[str, dict[str, Any]],
) -> tuple[bool, list[str], dict[str, Any]]:
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
        if formal.get("protocol_id") != "kinofail-confirmatory-o9-direct-final-f33":
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
    alignment: dict[str, Any] = {}
    if set(manifests) == {"nominal_counterfactual", "anomaly"}:
        try:
            alignment = audit_matched_horizon(
                Path(str(by_condition["anomaly"]["manifest"])),
                Path(str(by_condition["nominal_counterfactual"]["manifest"])),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"matched_horizon_audit_error:{type(exc).__name__}")
        else:
            if not alignment["passed"]:
                issues.extend(
                    f"matched_horizon:{key}"
                    for key, value in alignment["checks"].items()
                    if not value
                )
    source = source_by_pair[pair_id]
    evidence.update(
        {
            "summary_sha256": sha256(summary_path),
            "scene_id": source["scene_id"],
            "domain": source["domain"],
            "severity_id": source["severity_id"],
            "source_lambda": source["parameter_interpolation"]["source_lambda"],
            "source_counterfactual_group_id": source["o9_semantic_recollection"]["source_counterfactual_group_id"],
            "nominal_fallen": nominal.get("fallen"),
            "nominal_max_route_deviation_m": nominal.get("max_route_deviation_m"),
            "nominal_full_episode_outcome_reported_not_gated": True,
            "matched_diagnostic_horizon": alignment,
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
        raise ValueError("F33 is frozen to exactly three concurrent Isaac processes")
    seal_path = args.seal.resolve()
    seal = validate_seal(seal_path)
    schedule = ROOT / str(seal["schedule"])
    protocol = ROOT / str(seal["protocol"])
    registry = ROOT / str(seal["scene_registry"])
    asset_lock = ROOT / str(seal["material_lock"])
    corpus = Path(str(seal["corpus_root"]))
    corpus.mkdir(parents=True, exist_ok=True)
    schedule_rows = read_jsonl(schedule)
    source_by_pair = {
        str(row["counterfactual_group_id"]): row
        for row in schedule_rows
        if row["condition"] == "anomaly"
    }
    pair_ids = sorted(source_by_pair)
    if len(pair_ids) != 960:
        raise RuntimeError("F33 schedule does not contain 960 pairs")
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
    started_at = time.monotonic()

    def run_one(pair_id: str) -> dict[str, Any]:
        attempt_path = attempts_root / f"{pair_id}.json"
        if attempt_path.exists():
            prior = read_json(attempt_path)
            if prior.get("state") == "terminal":
                return prior
            pid = int(prior.get("pid", -1))
            if pid > 0 and _pid_matches(pid, pair_id):
                while _pid_matches(pid, pair_id):
                    time.sleep(5.0)
            accepted, audit_issues, evidence = audit_pair(pair_id, corpus, source_by_pair)
            terminal = {
                **prior,
                "state": "terminal",
                "completed_utc": datetime.now(UTC).isoformat(),
                "returncode": None,
                "timed_out": False,
                "reconciled_after_supervisor_restart": True,
                "recollection_performed": False,
                "accepted": accepted,
                "audit_issues": audit_issues,
                "evidence": evidence,
            }
            atomic_json(attempt_path, terminal)
            return terminal
        log_path = logs_root / f"{pair_id}.log"
        started = {
            "schema_version": "kinofail.confirmatory-o9-final-f33-attempt.v1",
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
            str(ROOT / "scripts/isaac_collect_kinofail_confirmatory_o9_direct_v2.py"),
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
        accepted, audit_issues, evidence = audit_pair(pair_id, corpus, source_by_pair)
        terminal = {
            **started,
            "state": "terminal",
            "completed_utc": datetime.now(UTC).isoformat(),
            "returncode": int(returncode),
            "timed_out": timed_out,
            "accepted": accepted,
            "audit_issues": audit_issues,
            "evidence": evidence,
        }
        atomic_json(attempt_path, terminal)
        with lock:
            print(json.dumps({"pair": pair_id, "returncode": returncode, "accepted": accepted, "issues": audit_issues}, sort_keys=True), flush=True)
        return terminal

    def update_state(state: str) -> None:
        attempts = [read_json(path) for path in attempts_root.glob("*.json")]
        atomic_json(
            OUTPUT / "supervisor_state.json",
            {
                "schema_version": "kinofail.confirmatory-o9-final-f33-supervisor.v1",
                "updated_utc": datetime.now(UTC).isoformat(),
                "state": state,
                "planned_pairs": 960,
                "attempt_records": len(attempts),
                "terminal_pairs": sum(row.get("state") == "terminal" for row in attempts),
                "accepted_pairs": sum(row.get("accepted") is True for row in attempts),
                "active_pairs": sum(row.get("state") == "started" for row in attempts),
                "maximum_concurrent_isaac_processes": 3,
                "elapsed_s_this_run": time.monotonic() - started_at,
            },
        )

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        for index, pair_id in enumerate(pair_ids):
            futures.append(pool.submit(run_one, pair_id))
            if index < 2:
                time.sleep(8.0)
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            update_state("running" if completed < len(pair_ids) else "finalizing")

    attempts = {path.stem: read_json(path) for path in attempts_root.glob("*.json")}
    accepted = [row for row in attempts.values() if row.get("accepted") is True]
    all_terminal = len(attempts) == 960 and all(row.get("state") == "terminal" for row in attempts.values())
    accepted_by_scene = Counter(str(row["evidence"].get("scene_id")) for row in accepted)
    accepted_by_domain = Counter(str(row["evidence"].get("domain")) for row in accepted)
    accepted_by_lambda = Counter(f"{float(row['evidence'].get('source_lambda')):.8f}" for row in accepted)
    effective_valid_scale = 9283 + len(accepted)
    remaining_attrition = 10560 - effective_valid_scale
    attrition_rate = remaining_attrition / 10560
    gates = {
        "all_frozen_pairs_terminal": all_terminal,
        "at_least_750_strict_o9_pairs": len(accepted) >= 750,
        "effective_scale_attrition_strictly_below_five_percent": attrition_rate < 0.05,
        "each_scene_has_at_least_24_of_32_pairs": len(accepted_by_scene) == 30 and min(accepted_by_scene.values(), default=0) >= 24,
        "each_domain_has_at_least_250_of_320_pairs": len(accepted_by_domain) == 3 and min(accepted_by_domain.values(), default=0) >= 250,
        "each_of_16_parameter_points_has_at_least_45_of_60_pairs": len(accepted_by_lambda) == 16 and min(accepted_by_lambda.values(), default=0) >= 45,
    }
    audit = {
        "schema_version": "kinofail.confirmatory-o9-final-f33-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "final",
        "passed": all(gates.values()),
        "model_prediction_feature_label_outcome_or_score_read": False,
        "result_dependent_retry_or_selection": False,
        "allowed_nonphysical_runtime_issue_suffixes": list(ALLOWED_VISUAL_ISSUES),
        "counts": {
            "scheduled_pairs": 960,
            "terminal_pairs": sum(row.get("state") == "terminal" for row in attempts.values()),
            "strictly_accepted_o9_pairs": len(accepted),
            "rejected_o9_pairs": 960 - len(accepted),
            "effective_non_o9_scale_pairs": 9283,
            "effective_total_scale_pairs": effective_valid_scale,
            "remaining_scale_attrition_pairs": remaining_attrition,
        },
        "effective_scale_attrition_rate": attrition_rate,
        "gates": gates,
        "accepted_by_scene": dict(sorted(accepted_by_scene.items())),
        "accepted_by_domain": dict(sorted(accepted_by_domain.items())),
        "accepted_by_source_lambda": dict(sorted(accepted_by_lambda.items())),
        "accepted": sorted(
            [
                {
                    "pair_id": row["pair_id"],
                    "source_counterfactual_group_id": row["evidence"]["source_counterfactual_group_id"],
                    "scene_id": row["evidence"]["scene_id"],
                    "domain": row["evidence"]["domain"],
                    "severity_id": row["evidence"]["severity_id"],
                    "source_lambda": row["evidence"]["source_lambda"],
                    "attempt_sha256": sha256(attempts_root / f"{row['pair_id']}.json"),
                }
                for row in accepted
            ],
            key=lambda row: row["pair_id"],
        ),
        "rejected": sorted(
            [
                {
                    "pair_id": row["pair_id"],
                    "source_counterfactual_group_id": row.get("evidence", {}).get("source_counterfactual_group_id"),
                    "audit_issues": row.get("audit_issues", []),
                }
                for row in attempts.values()
                if row.get("accepted") is not True
            ],
            key=lambda row: row["pair_id"],
        ),
        "seal_sha256": sha256(seal_path),
    }
    atomic_json(OUTPUT / "final_audit.json", audit)
    update_state("completed" if audit["passed"] else "gate_failed")
    print(json.dumps({"final_audit": str(OUTPUT / "final_audit.json"), "passed": audit["passed"], "counts": audit["counts"], "gates": gates}, indent=2), flush=True)
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
