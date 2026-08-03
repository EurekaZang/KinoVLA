#!/usr/bin/env python3
"""Run one deterministic partition of a frozen reconfirmation scene battery.

The scientific collector is executed in a fresh Isaac Sim process for every
counterfactual pair.  This operational launcher only supplies sealed inputs,
records an append-only attempt state before process creation, and prevents
any result-dependent retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ISAACLAB = Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh")
EXPERIENCE = Path(
    "/home/eureka/IsaacLab-v2.3.0/apps/"
    "isaaclab.python.headless.rendering.kit"
)
CONDA_PREFIX = Path("/home/eureka/miniconda3/envs/kinovla")
DEFAULT_COLLECTOR = (
    ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v9.py"
)
DEFAULT_OPERATIONAL_AMENDMENT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f12_process_local_rtx_texture_amendment1/"
    "amendment_manifest.json"
)
F10_SCHEMA = (
    "kinofail.reconfirmation-f10-t3-scene-source-provenance-amendment.v1"
)
F10_STATUS = "sealed_before_resumption_of_t3_acquisition"
F10_PREDECESSOR_COLLECTOR_SHA256 = (
    "de790876ff90331ab40fdbc85f90ee691594cda07dad4ec9222af285999aa726"
)
F11_SCHEMA = (
    "kinofail.reconfirmation-f11-global-capsule-attrition-amendment.v1"
)
F11_STATUS = "sealed_before_resumption_of_physical_acquisition"
F12_SCHEMA = (
    "kinofail.reconfirmation-f12-process-local-rtx-texture-amendment.v1"
)
F12_STATUS = "sealed_before_resumption_after_rtx_texture_race"
F13 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f13_launcher_eligibility_amendment1/"
    "amendment_manifest.json"
)
F14 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f14_t2_zero_observation_liveness_amendment1/"
    "amendment_manifest.json"
)
F15 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f15_runner_supersession_amendment1/"
    "amendment_manifest.json"
)
ISAACLAB_PYTHONPATH = (
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_tasks:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_assets:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_rl:"
    "/home/eureka/KinoVLA"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _attempt_audit(
    *,
    scene_id: str,
    battery: str,
    partition_index: int,
    partition_count: int,
    schedule: Path,
    protocol: Path,
    registry: Path,
    asset_lock: Path,
    collector: Path,
    operational_amendment: Path | None,
    selected_pair_ids: list[str],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "kinofail.reconfirmation-launcher-audit.v2",
        "updated_utc": datetime.now(UTC).isoformat(),
        "scientific_content_changed": False,
        "result_dependent_retry_permitted": False,
        "fresh_isaac_process_per_counterfactual_pair": True,
        "scene_id": scene_id,
        "battery": battery,
        "partition_index": partition_index,
        "partition_count": partition_count,
        "schedule": str(schedule),
        "schedule_sha256": _sha256(schedule),
        "protocol": str(protocol),
        "protocol_sha256": _sha256(protocol),
        "scene_registry": str(registry),
        "scene_registry_sha256": _sha256(registry),
        "asset_lock": str(asset_lock),
        "asset_lock_sha256": _sha256(asset_lock),
        "collector": str(collector),
        "collector_sha256": _sha256(collector),
        "operational_amendment": (
            str(operational_amendment)
            if operational_amendment is not None
            else None
        ),
        "operational_amendment_sha256": (
            _sha256(operational_amendment)
            if operational_amendment is not None
            else None
        ),
        "selected_pair_ids": selected_pair_ids,
        "attempts": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--asset-lock", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--collector", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument(
        "--operational-amendment",
        type=Path,
        default=DEFAULT_OPERATIONAL_AMENDMENT,
    )
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--partition-count", type=int, default=4)
    args = parser.parse_args()

    if args.partition_count < 1 or args.partition_count > 8:
        raise ValueError("partition count must be in [1, 8]")
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index is outside the partition count")

    schedule_path = args.schedule.resolve()
    registry_path = args.scene_registry.resolve()
    protocol_path = args.protocol.resolve()
    asset_lock_path = args.asset_lock.resolve()
    collector_path = args.collector.resolve()
    amendment_path = args.operational_amendment.resolve()
    corpus_root = args.corpus_root.resolve()
    for path in (
        schedule_path,
        registry_path,
        protocol_path,
        asset_lock_path,
        collector_path,
        amendment_path,
        ISAACLAB,
        EXPERIENCE,
        CONDA_PREFIX / "bin/python",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    protocol = _json(protocol_path)
    if (
        protocol.get("status") != "frozen"
        or protocol.get("schema_version")
        != "kinofail.formal-collection-protocol.v1"
    ):
        raise RuntimeError("collection protocol is not a frozen formal protocol")
    for key, actual in (
        ("schedule_sha256", _sha256(schedule_path)),
        ("scene_registry_sha256", _sha256(registry_path)),
        ("material_lock_sha256", _sha256(asset_lock_path)),
    ):
        if protocol.get(key) != actual:
            raise RuntimeError(f"frozen protocol {key} mismatch")
    protocol_collector_sha256 = str(protocol.get("collector_sha256"))
    if _sha256(collector_path) == protocol_collector_sha256:
        active_amendment: Path | None = None
    else:
        amendment = _json(amendment_path)
        correction = amendment.get("correction", {})
        schema = amendment.get("schema_version")
        allowed_status = {
            "kinofail.reconfirmation-collector-provenance-amendment.v1": (
                "sealed_before_scale_or_t3_episode_acquisition"
            ),
            "kinofail.reconfirmation-f0-nuisance-contract-amendment.v1": (
                "sealed_before_model_blind_feature_extraction"
            ),
        }
        if schema in allowed_status:
            predecessor_sha256 = (
                correction.get("exact_hash_predecessor_sha256")
                if schema
                == (
                    "kinofail.reconfirmation-collector-provenance-"
                    "amendment.v1"
                )
                else correction.get("formal_protocol_predecessor_sha256")
            )
            if (
                amendment.get("passed") is not True
                or amendment.get("status") != allowed_status[schema]
                or correction.get("wrapper_sha256")
                != _sha256(collector_path)
                or predecessor_sha256 != protocol_collector_sha256
                or correction.get("simulation_logic_changed") is not False
                or correction.get("sensor_or_feature_logic_changed") is not False
                or correction.get("threshold_or_analysis_changed") is not False
            ):
                raise RuntimeError("collector operational amendment is invalid")
        elif (
            schema
            == "kinofail.reconfirmation-f8-ext4-recovery-amendment.v1"
        ):
            recovery = amendment.get("recovery", {})
            scientific_contract = amendment.get("scientific_contract", {})
            predecessor_f3 = Path(str(amendment.get("predecessor_f3", "")))
            if not predecessor_f3.is_absolute():
                predecessor_f3 = ROOT / predecessor_f3
            if (
                amendment.get("passed") is not True
                or amendment.get("status")
                != "sealed_before_f8_recovery_execution"
                or not predecessor_f3.is_file()
                or amendment.get("predecessor_f3_sha256")
                != _sha256(predecessor_f3)
                or recovery.get("collector_sha256")
                != _sha256(collector_path)
                or recovery.get("runner_sha256")
                != _sha256(Path(__file__).resolve())
                or scientific_contract.get("simulation_logic_changed")
                is not False
                or scientific_contract.get("sensor_or_feature_logic_changed")
                is not False
                or scientific_contract.get("model_or_route_changed")
                is not False
                or scientific_contract.get("threshold_or_analysis_changed")
                is not False
                or scientific_contract.get("result_dependent_retry_enabled")
                is not False
            ):
                raise RuntimeError("F8 recovery amendment is invalid")
        elif schema == F10_SCHEMA:
            correction = amendment.get("correction", {})
            incident = amendment.get("incident", {})
            scientific_contract = amendment.get(
                "scientific_contract", {}
            )
            predecessor_f8 = Path(
                str(amendment.get("predecessor_f8", ""))
            )
            predecessor_f9 = Path(
                str(amendment.get("predecessor_f9", ""))
            )
            if not predecessor_f8.is_absolute():
                predecessor_f8 = ROOT / predecessor_f8
            if not predecessor_f9.is_absolute():
                predecessor_f9 = ROOT / predecessor_f9
            scene_pipeline = ROOT / str(
                correction.get("scene_pipeline", "")
            )
            finalizer = ROOT / str(
                correction.get("finalizer", "")
            )
            affected = incident.get("affected_pair_ids")
            if (
                amendment.get("passed") is not True
                or amendment.get("status") != F10_STATUS
                or not predecessor_f8.is_file()
                or not predecessor_f9.is_file()
                or amendment.get("predecessor_f8_sha256")
                != _sha256(predecessor_f8)
                or amendment.get("predecessor_f9_sha256")
                != _sha256(predecessor_f9)
                or correction.get("collector_sha256")
                != _sha256(collector_path)
                or correction.get("predecessor_collector_sha256")
                != F10_PREDECESSOR_COLLECTOR_SHA256
                or correction.get("runner_sha256")
                != _sha256(Path(__file__).resolve())
                or not scene_pipeline.is_file()
                or correction.get("scene_pipeline_sha256")
                != _sha256(scene_pipeline)
                or not finalizer.is_file()
                or correction.get("finalizer_sha256")
                != _sha256(finalizer)
                or correction.get("registry_readback_only") is not True
                or correction.get("frozen_record_hash_preserved")
                is not True
                or not isinstance(affected, list)
                or len(affected) != 15
                or len(set(affected)) != 15
                or incident.get("authorized_reexecution_pair_ids") != []
                or incident.get("identified_without_model_outputs")
                is not True
                or scientific_contract.get("simulation_logic_changed")
                is not False
                or scientific_contract.get(
                    "sensor_or_feature_logic_changed"
                )
                is not False
                or scientific_contract.get("model_or_route_changed")
                is not False
                or scientific_contract.get(
                    "threshold_or_analysis_changed"
                )
                is not False
                or scientific_contract.get(
                    "schedule_or_protocol_changed"
                )
                is not False
                or scientific_contract.get(
                    "result_dependent_retry_enabled"
                )
                is not False
            ):
                raise RuntimeError("F10 provenance amendment is invalid")
        elif schema == F11_SCHEMA:
            correction = amendment.get("correction", {})
            scientific_contract = amendment.get(
                "scientific_contract", {}
            )
            predecessor_f10 = Path(
                str(amendment.get("predecessor_f10", ""))
            )
            if not predecessor_f10.is_absolute():
                predecessor_f10 = ROOT / predecessor_f10
            scene_pipeline = ROOT / str(
                correction.get("scene_pipeline", "")
            )
            finalizer = ROOT / str(
                correction.get("finalizer", "")
            )
            capsule_builder = ROOT / str(
                correction.get("capsule_builder", "")
            )
            if (
                amendment.get("passed") is not True
                or amendment.get("status") != F11_STATUS
                or not predecessor_f10.is_file()
                or amendment.get("predecessor_f10_sha256")
                != _sha256(predecessor_f10)
                or _json(predecessor_f10).get("incident", {}).get(
                    "authorized_reexecution_pair_ids"
                )
                != []
                or correction.get("collector_sha256")
                != _sha256(collector_path)
                or correction.get("runner_sha256")
                != _sha256(Path(__file__).resolve())
                or not scene_pipeline.is_file()
                or correction.get("scene_pipeline_sha256")
                != _sha256(scene_pipeline)
                or not finalizer.is_file()
                or correction.get("finalizer_sha256")
                != _sha256(finalizer)
                or not capsule_builder.is_file()
                or correction.get("capsule_builder_sha256")
                != _sha256(capsule_builder)
                or correction.get("per_scene_gate_removed") is not True
                or correction.get("global_gate_preserved") is not True
                or scientific_contract.get(
                    "attrition_threshold_changed"
                )
                is not False
                or scientific_contract.get("simulation_logic_changed")
                is not False
                or scientific_contract.get(
                    "sensor_or_feature_logic_changed"
                )
                is not False
                or scientific_contract.get("model_or_route_changed")
                is not False
                or scientific_contract.get(
                    "schedule_or_protocol_changed"
                )
                is not False
                or scientific_contract.get(
                    "result_dependent_retry_enabled"
                )
                is not False
            ):
                raise RuntimeError("F11 global attrition amendment is invalid")
        elif schema == F12_SCHEMA:
            correction = amendment.get("correction", {})
            incident = amendment.get("incident", {})
            scientific_contract = amendment.get(
                "scientific_contract", {}
            )
            predecessor_f11 = Path(
                str(amendment.get("predecessor_f11", ""))
            )
            if not predecessor_f11.is_absolute():
                predecessor_f11 = ROOT / predecessor_f11
            scene_pipeline = ROOT / str(
                correction.get("scene_pipeline", "")
            )
            finalizer = ROOT / str(
                correction.get("finalizer", "")
            )
            backend = ROOT / str(
                correction.get("backend", "")
            )
            f13 = _json(F13)
            f14 = _json(F14)
            f15 = _json(F15)
            f13_correction = f13.get("correction", {})
            f15_correction = f15.get("correction", {})
            f15_scientific = f15.get("scientific_contract", {})
            if (
                amendment.get("passed") is not True
                or amendment.get("status") != F12_STATUS
                or not predecessor_f11.is_file()
                or amendment.get("predecessor_f11_sha256")
                != _sha256(predecessor_f11)
                or correction.get("collector_sha256")
                != _sha256(collector_path)
                or correction.get("predecessor_collector_sha256")
                != _sha256(
                    ROOT
                    / "scripts/"
                    "isaac_collect_kinofail_confirmatory_pair_v8.py"
                )
                or correction.get("runner_sha256")
                != f15_correction.get("predecessor_runner_sha256")
                or not scene_pipeline.is_file()
                or correction.get("scene_pipeline_sha256")
                != f13_correction.get(
                    "predecessor_scene_pipeline_sha256"
                )
                or f13.get("predecessor_f12_sha256")
                != _sha256(amendment_path)
                or f13_correction.get("scene_pipeline_sha256")
                != _sha256(scene_pipeline)
                or f14.get("predecessor_f13_sha256") != _sha256(F13)
                or f15.get("passed") is not True
                or f15.get("schema_version")
                != (
                    "kinofail.reconfirmation-f15-runner-"
                    "supersession-amendment.v1"
                )
                or f15.get("status")
                != "sealed_before_scene01_unstarted_partition_launch"
                or f15.get("predecessor_f14_sha256") != _sha256(F14)
                or f15_correction.get("runner_sha256")
                != _sha256(Path(__file__).resolve())
                or f15_correction.get("scene_pipeline_sha256")
                != _sha256(scene_pipeline)
                or not finalizer.is_file()
                or not backend.is_file()
                or correction.get("backend_sha256")
                != _sha256(backend)
                or correction.get("texture_pixels_changed") is not False
                or correction.get("process_local_namespace") is not True
                or incident.get("authorized_reexecution_pair_ids") != []
                or incident.get("affected_pair_count") != 4
                or scientific_contract.get("simulation_logic_changed")
                is not False
                or scientific_contract.get(
                    "sensor_or_feature_logic_changed"
                )
                is not False
                or scientific_contract.get("model_or_route_changed")
                is not False
                or scientific_contract.get(
                    "schedule_or_protocol_changed"
                )
                is not False
                or scientific_contract.get(
                    "threshold_or_analysis_changed"
                )
                is not False
                or scientific_contract.get(
                    "result_dependent_retry_enabled"
                )
                is not False
                or f15_scientific.get(
                    "simulation_or_sensor_logic_changed"
                )
                is not False
                or f15_scientific.get(
                    "schedule_feature_model_threshold_or_analysis_changed"
                )
                is not False
                or f15_scientific.get(
                    "result_dependent_retry_enabled"
                )
                is not False
            ):
                raise RuntimeError(
                    "F12 process-local RTX texture amendment is invalid"
                )
        else:
            raise RuntimeError("collector operational amendment is invalid")
        active_amendment = amendment_path

    records = _jsonl(schedule_path)
    scenes = {str(row["scene_family"]) for row in records}
    batteries = {str(row["battery"]) for row in records}
    if len(scenes) != 1 or len(batteries) != 1:
        raise RuntimeError("a partition must contain one scene and one battery")
    scene_id = next(iter(scenes))
    battery = next(iter(batteries))

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(
            str(record["counterfactual_group_id"]), []
        ).append(record)
    invalid = {
        pair_id
        for pair_id, rows in groups.items()
        if len(rows) != 2
        or {str(row["condition"]) for row in rows}
        != {"nominal_counterfactual", "anomaly"}
    }
    if invalid:
        raise RuntimeError(
            f"incomplete frozen counterfactual pairs: {sorted(invalid)[:3]}"
        )
    selected_pair_ids = [
        pair_id
        for index, pair_id in enumerate(sorted(groups))
        if index % args.partition_count == args.partition_index
    ]

    corpus_root.mkdir(parents=True, exist_ok=True)
    if active_amendment is None:
        amendment_schema = None
        battery_namespace = battery
        newly_authorized_reexecution: set[str] = set()
        inherited_authorized_reexecution: set[str] = set()
    else:
        active_value = _json(active_amendment)
        amendment_schema = str(active_value["schema_version"])
        if (
            amendment_schema
            == "kinofail.reconfirmation-f0-nuisance-contract-amendment.v1"
        ):
            battery_namespace = f"{battery}_f3v7"
            newly_authorized_reexecution = set(
                active_value["incident_disposition"][
                    "authorized_reexecution_pair_ids"
                ]
            )
            predecessor_f2 = Path(str(active_value["predecessor_f2"]))
            if not predecessor_f2.is_absolute():
                predecessor_f2 = ROOT / predecessor_f2
            inherited_authorized_reexecution = set(
                _json(predecessor_f2)["incident"]["affected_pair_ids"]
            )
        elif (
            amendment_schema
            == "kinofail.reconfirmation-f8-ext4-recovery-amendment.v1"
        ):
            battery_namespace = f"{battery}_f8r1"
            newly_authorized_reexecution = set(
                active_value["recovery"][
                    "authorized_reexecution_pair_ids"
                ]
            )
            inherited_authorized_reexecution = set()
        elif amendment_schema == F10_SCHEMA:
            battery_namespace = f"{battery}_f10v8"
            newly_authorized_reexecution = set()
            inherited_authorized_reexecution = set()
        elif amendment_schema == F11_SCHEMA:
            battery_namespace = f"{battery}_f11v8"
            newly_authorized_reexecution = set()
            inherited_authorized_reexecution = set()
        elif amendment_schema == F12_SCHEMA:
            battery_namespace = f"{battery}_f12v9"
            newly_authorized_reexecution = set()
            inherited_authorized_reexecution = set()
        else:
            battery_namespace = f"{battery}_f2v6"
            newly_authorized_reexecution = set(
                active_value["incident"]["affected_pair_ids"]
            )
            inherited_authorized_reexecution = set()
    audit_dir = corpus_root / "launcher_audits"
    log_dir = corpus_root / "launcher_logs" / battery_namespace
    audit_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    audit_path = (
        audit_dir
        / (
            f"{battery_namespace}_partition_"
            f"{args.partition_index}_of_{args.partition_count}.json"
        )
    )
    prior = _json(audit_path) if audit_path.is_file() else {"attempts": []}
    if audit_path.is_file() and (
        prior.get("schedule_sha256") != _sha256(schedule_path)
        or prior.get("selected_pair_ids") != selected_pair_ids
    ):
        raise RuntimeError("existing launcher audit belongs to another partition")
    attempts = list(prior.get("attempts", []))
    attempted = {
        str(row["counterfactual_group_id"])
        for row in attempts
        if row.get("state") in {"started", "terminal"}
    }
    historical_latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for historical_path in sorted(
        audit_dir.glob(f"{battery}*_partition_*_of_*.json")
    ):
        if historical_path == audit_path:
            continue
        historical = _json(historical_path)
        for row in historical.get("attempts", []):
            pair_id = str(row["counterfactual_group_id"])
            previous = historical_latest.get(pair_id)
            if previous is None or str(row.get("started_utc", "")) > str(
                previous[1].get("started_utc", "")
            ):
                historical_latest[pair_id] = (historical_path.name, row)
    blocked_historical = set()
    for pair_id, (audit_name, row) in historical_latest.items():
        if pair_id in newly_authorized_reexecution:
            continue
        if (
            pair_id in inherited_authorized_reexecution
            and "_f2v6_" not in audit_name
        ):
            continue
        if row.get("state") in {"started", "terminal"}:
            blocked_historical.add(pair_id)

    environment = os.environ.copy()
    environment["CONDA_PREFIX"] = str(CONDA_PREFIX)
    environment["PATH"] = (
        f"{CONDA_PREFIX / 'bin'}:{environment.get('PATH', '')}"
    )
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["PYTHONPATH"] = ISAACLAB_PYTHONPATH

    for progress, pair_id in enumerate(selected_pair_ids, start=1):
        summary_path = corpus_root / "pair_summaries" / f"{pair_id}.json"
        if summary_path.is_file() and _json(summary_path).get("passed") is True:
            status = "skip_sealed_pass"
            print(
                json.dumps(
                    {
                        "battery": battery,
                        "progress": f"{progress}/{len(selected_pair_ids)}",
                        "pair": pair_id,
                        "status": status,
                    }
                ),
                flush=True,
            )
            continue
        if pair_id in attempted:
            status = "skip_recorded_attempt_without_retry"
            print(
                json.dumps(
                    {
                        "battery": battery,
                        "progress": f"{progress}/{len(selected_pair_ids)}",
                        "pair": pair_id,
                        "status": status,
                    }
                ),
                flush=True,
            )
            continue
        if pair_id in blocked_historical:
            print(
                json.dumps(
                    {
                        "battery": battery,
                        "progress": f"{progress}/{len(selected_pair_ids)}",
                        "pair": pair_id,
                        "status": "skip_historical_terminal_without_retry",
                    }
                ),
                flush=True,
            )
            continue

        log_path = log_dir / f"{pair_id}.log"
        started = {
            "counterfactual_group_id": pair_id,
            "state": "started",
            "started_utc": datetime.now(UTC).isoformat(),
            "log": str(log_path),
            "retry_authorized": False,
        }
        attempts.append(started)
        attempted.add(pair_id)
        _write_json(
            audit_path,
            _attempt_audit(
                scene_id=scene_id,
                battery=battery,
                partition_index=args.partition_index,
                partition_count=args.partition_count,
                schedule=schedule_path,
                protocol=protocol_path,
                registry=registry_path,
                asset_lock=asset_lock_path,
                collector=collector_path,
                operational_amendment=active_amendment,
                selected_pair_ids=selected_pair_ids,
                attempts=attempts,
            ),
        )
        command = [
            str(ISAACLAB),
            "-p",
            str(collector_path),
            "--schedule",
            str(schedule_path),
            "--scene-registry",
            str(registry_path),
            "--protocol",
            str(protocol_path),
            "--asset-lock",
            str(asset_lock_path),
            "--corpus-root",
            str(corpus_root),
            "--counterfactual-group-id",
            pair_id,
            "--headless",
            "--enable_cameras",
            "--experience",
            str(EXPERIENCE),
        ]
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary = _json(summary_path) if summary_path.is_file() else {}
        terminal = {
            **started,
            "state": "terminal",
            "completed_utc": datetime.now(UTC).isoformat(),
            "returncode": int(completed.returncode),
            "summary_exists": summary_path.is_file(),
            "passed": summary.get("passed") is True,
        }
        attempts[-1] = terminal
        _write_json(
            audit_path,
            _attempt_audit(
                scene_id=scene_id,
                battery=battery,
                partition_index=args.partition_index,
                partition_count=args.partition_count,
                schedule=schedule_path,
                protocol=protocol_path,
                registry=registry_path,
                asset_lock=asset_lock_path,
                collector=collector_path,
                operational_amendment=active_amendment,
                selected_pair_ids=selected_pair_ids,
                attempts=attempts,
            ),
        )
        print(
            json.dumps(
                {
                    **terminal,
                    "battery": battery,
                    "progress": f"{progress}/{len(selected_pair_ids)}",
                }
            ),
            flush=True,
        )

    passed = {
        pair_id
        for pair_id in selected_pair_ids
        if (corpus_root / "pair_summaries" / f"{pair_id}.json").is_file()
        and _json(corpus_root / "pair_summaries" / f"{pair_id}.json").get(
            "passed"
        )
        is True
    }
    return 0 if passed == set(selected_pair_ids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
