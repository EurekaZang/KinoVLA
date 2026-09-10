#!/usr/bin/env python3
"""Run the frozen V2 all-191 Scale/T2 extension with resumable scene gates."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "outputs/kinofail_kino_v4_all191_scale_t2_extension_v2"
REGISTRY = DESIGN / "design/scene_registry.json"
SCHEDULES = DESIGN / "schedules"
SUPERSESSION = DESIGN / "design/supersession_audit.json"
AMENDMENT = ROOT / "outputs/freeze/kino_v4_all191_scale_provenance_v2/amendment_manifest.json"
LOCK = Path(
    "/data/eureka/kinofail_kino_v4_confirmation_v1/assets/terrain_pbr/"
    "terrain_assets.lock.json"
)
CORPUS = Path("/data/eureka/kinofail_kino_v4_all191_scale_t2_extension_v2/corpus")
OUT = DESIGN / "orchestration/campaign_v1"
RECOVERY_FREEZE = ROOT / (
    "outputs/freeze/kino_v4_all191_operational_supplement_v1/"
    "freeze_manifest.json"
)
RECOVERY_SCHEDULE = DESIGN / (
    "schedules/operational_supplement_v1/scenes/"
    "kino4c_production_020/scale/schedule.jsonl"
)
EVALUATION_SCALE_SCHEDULE = DESIGN / (
    "schedules/operational_supplement_v1/global/"
    "evaluation_scale_schedule.jsonl"
)
REPLACEMENT_MAP = DESIGN / (
    "schedules/operational_supplement_v1/replacement_map.json"
)
PAIR_RUNNER = ROOT / "scripts/run_kinofail_reconfirmation_pair_partition_v2.py"
T2_RUNNER = ROOT / "scripts/run_kinofail_reconfirmation_t2_scene_v2.py"
T2_OPERATIONAL_POLICY = (
    ROOT
    / "outputs/freeze/kino_v4_all191_t2_operational_timeout_policy_v1/policy_manifest.json"
)
PAIR_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_kino_v4_confirmation_pair_v1.py"
T2_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def structural_scene_audit(scene_id: str) -> dict[str, Any]:
    scene_corpus = CORPUS / scene_id
    scale_schedule = SCHEDULES / "scenes" / scene_id / "scale/schedule.jsonl"
    scale_rows = [
        value
        for value in rows(EVALUATION_SCALE_SCHEDULE)
        if str(value["scene_id"]) == scene_id
    ]
    scale_ids = {str(row["counterfactual_group_id"]) for row in scale_rows}
    replacement = load(REPLACEMENT_MAP)
    replacement_ids = set(replacement["old_to_new_pair_id"].values())
    retired_ids = set(replacement["incomplete_original_pair_ids_replaced"])
    summaries: dict[str, dict[str, Any]] = {}
    for path in (scene_corpus / "pair_summaries").glob("*.json"):
        value = load(path)
        summaries[str(value.get("counterfactual_group_id"))] = value
    complete_scale = {
        pair_id
        for pair_id, value in summaries.items()
        if pair_id in scale_ids
        and isinstance(value.get("results"), list)
        and len(value["results"]) == 2
        and {str(item.get("condition")) for item in value["results"]}
        == {"nominal_counterfactual", "anomaly"}
        and all(Path(str(item.get("manifest", ""))).is_file() for item in value["results"])
    }
    pair_operator = {
        str(row["counterfactual_group_id"]): str(row["target_operator"])
        for row in scale_rows
    }
    planned_operators = set(pair_operator.values())
    complete_operators = {pair_operator[pair_id] for pair_id in complete_scale}
    attrited_scale: set[str] = set()
    retired_operational_attrition: set[str] = set()
    invalid_attrition_markers: list[str] = []
    for path in (scene_corpus / "operational_attrition").glob("*.json"):
        value = load(path)
        pair_id = str(value.get("counterfactual_group_id", ""))
        if pair_id in retired_ids and scene_id == replacement["scene_id"]:
            retired_operational_attrition.add(pair_id)
            continue
        log_path = Path(str(value.get("log", "")))
        marker_schedule = (
            RECOVERY_SCHEDULE if pair_id in replacement_ids else scale_schedule
        )
        valid = (
            value.get("schema_version")
            == "kinofail.kino-v4-operational-attrition.v1"
            and value.get("status") == "sealed_before_model_inference"
            and value.get("scene_id") == scene_id
            and pair_id in scale_ids
            and value.get("schedule_sha256") == sha256(marker_schedule)
            and value.get("result_dependent_retry_permitted") is False
            and value.get("model_predictions_read") is False
            and value.get("method_scores_read") is False
            and value.get("scientific_content_changed") is False
            and log_path.is_file()
            and value.get("log_sha256") == sha256(log_path)
        )
        if valid:
            attrited_scale.add(pair_id)
        else:
            invalid_attrition_markers.append(path.name)

    t2_schedule = SCHEDULES / "c2_t2/schedule.jsonl"
    t2_rows = [row for row in rows(t2_schedule) if str(row["scene_cluster"]) == scene_id]
    t2_ids = {str(row["case_id"]) for row in t2_rows}
    t2_root = scene_corpus / "c2_t2"
    complete_t2 = {
        case_id for case_id in t2_ids if (t2_root / case_id / "manifest.json").is_file()
    }
    t2_summary = t2_root / "scene_summaries" / f"{scene_id}.json"
    passed = (
        len(scale_ids) == 22
        and complete_scale.isdisjoint(attrited_scale)
        and complete_scale | attrited_scale == scale_ids
        # One device-loss event can invalidate the active pair in each of the
        # three workers plus the next process launched before CUDA recovers.
        # Stop rather than silently dilute a scene beyond that incident bound.
        and len(attrited_scale) <= 4
        and complete_operators == planned_operators
        and not invalid_attrition_markers
        and len(t2_ids) == 4
        and complete_t2 == t2_ids
        and t2_summary.is_file()
    )
    return {
        "passed": passed,
        "scene_id": scene_id,
        "scale": {
            "planned_pairs": len(scale_ids),
            "structurally_complete_pairs": len(complete_scale),
            "missing_pair_ids": sorted(scale_ids - complete_scale),
            "sealed_operational_attrition_pair_ids": sorted(attrited_scale),
            "retired_operational_attrition_pair_ids": sorted(
                retired_operational_attrition
            ),
            "invalid_operational_attrition_markers": sorted(
                invalid_attrition_markers
            ),
            "planned_operator_count": len(planned_operators),
            "complete_operator_count": len(complete_operators),
        },
        "t2": {
            "planned_cases": len(t2_ids),
            "structurally_complete_cases": len(complete_t2),
            "missing_case_ids": sorted(t2_ids - complete_t2),
            "scene_summary_exists": t2_summary.is_file(),
        },
    }


def command_for(scene_id: str, name: str, partitions: int) -> list[str]:
    scene_corpus = CORPUS / scene_id
    if name == "t2":
        return [
            sys.executable,
            str(T2_RUNNER),
            "--schedule",
            str(SCHEDULES / "c2_t2/schedule.jsonl"),
            "--scene-registry",
            str(REGISTRY),
            "--asset-lock",
            str(LOCK),
            "--protocol",
            str(SCHEDULES / "c2_t2/collection_protocol.json"),
            "--out",
            str(scene_corpus / "c2_t2"),
            "--scene",
            scene_id,
            "--collector",
            str(T2_COLLECTOR),
            "--operational-policy",
            str(T2_OPERATIONAL_POLICY),
        ]
    partition = int(name.rsplit("p", 1)[1])
    scale_root = SCHEDULES / "scenes" / scene_id / "scale"
    return [
        sys.executable,
        str(PAIR_RUNNER),
        "--schedule",
        str(scale_root / "schedule.jsonl"),
        "--scene-registry",
        str(REGISTRY),
        "--protocol",
        str(scale_root / "collection_protocol.json"),
        "--asset-lock",
        str(LOCK),
        "--corpus-root",
        str(scene_corpus),
        "--collector",
        str(PAIR_COLLECTOR),
        "--operational-amendment",
        str(AMENDMENT),
        "--partition-index",
        str(partition),
        "--partition-count",
        str(partitions),
    ]


def run_job(scene_id: str, name: str, command: list[str]) -> dict[str, Any]:
    log = OUT / "logs" / scene_id / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise FileExistsError(log)
    started = datetime.now(UTC).isoformat()
    with log.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "name": name,
        "returncode": int(completed.returncode),
        "started_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "log": str(log),
        "log_sha256": sha256(log),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.partitions != 3 or not 1 <= args.workers <= 3:
        raise ValueError("frozen campaign uses exactly 3 partitions and at most 3 workers")
    for path in (
        REGISTRY,
        SUPERSESSION,
        AMENDMENT,
        LOCK,
        RECOVERY_FREEZE,
        RECOVERY_SCHEDULE,
        EVALUATION_SCALE_SCHEDULE,
        REPLACEMENT_MAP,
        PAIR_RUNNER,
        T2_RUNNER,
        T2_OPERATIONAL_POLICY,
        PAIR_COLLECTOR,
        T2_COLLECTOR,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if load(SUPERSESSION).get("passed") is not True or load(AMENDMENT).get("passed") is not True:
        raise RuntimeError("V2 precollection freeze is invalid")
    recovery = load(RECOVERY_FREEZE)
    if (
        recovery.get("passed") is not True
        or recovery.get("status")
        != "frozen_before_supplemental_acquisition"
        or recovery.get("artifacts", {}).get(
            "evaluation_scale_schedule_sha256"
        )
        != sha256(EVALUATION_SCALE_SCHEDULE)
        or recovery.get("artifacts", {}).get("supplemental_schedule_sha256")
        != sha256(RECOVERY_SCHEDULE)
        or recovery.get("artifacts", {}).get("replacement_map_sha256")
        != sha256(REPLACEMENT_MAP)
    ):
        raise RuntimeError("operational supplement freeze is invalid")

    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "campaign.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another all191 campaign process is active") from exc

    registry = load(REGISTRY)
    scene_ids = [str(row["scene_id"]) for row in registry["scenes"]]
    if len(scene_ids) != 179:
        raise RuntimeError("extension registry must contain 179 scenes")
    campaign_state = OUT / "campaign_state.json"
    atomic(
        campaign_state,
        {
            "schema_version": "kinofail.kino-v4-all191-scale-t2-campaign.v1",
            "status": "running",
            "updated_utc": datetime.now(UTC).isoformat(),
            "scene_count": len(scene_ids),
            "partitions": args.partitions,
            "workers": args.workers,
            "result_dependent_retry_permitted": False,
            "registry_sha256": sha256(REGISTRY),
            "supersession_sha256": sha256(SUPERSESSION),
            "amendment_sha256": sha256(AMENDMENT),
        },
    )

    completed_scenes: list[str] = []
    for position, scene_id in enumerate(scene_ids, start=1):
        scene_state = OUT / "scenes" / f"{scene_id}.json"
        audit = structural_scene_audit(scene_id)
        if audit["passed"]:
            atomic(
                scene_state,
                {
                    "status": "complete",
                    "completion_source": "preexisting_structural_audit",
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "audit": audit,
                },
            )
            completed_scenes.append(scene_id)
            continue
        if scene_state.is_file():
            prior = load(scene_state)
            if prior.get("status") in {"started", "operational_failure"}:
                atomic(
                    campaign_state,
                    {
                        **load(campaign_state),
                        "status": "operational_failure",
                        "updated_utc": datetime.now(UTC).isoformat(),
                        "scene_id": scene_id,
                        "scene_position": position,
                        "audit": audit,
                        "reason": "nonterminal prior scene requires explicit adjudication",
                    },
                )
                return 2

        started = {
            "schema_version": "kinofail.kino-v4-all191-scene-run.v1",
            "status": "started",
            "scene_id": scene_id,
            "scene_position": position,
            "scene_total": len(scene_ids),
            "started_utc": datetime.now(UTC).isoformat(),
            "result_dependent_retry_permitted": False,
        }
        atomic(scene_state, started)
        atomic(
            campaign_state,
            {
                **load(campaign_state),
                "updated_utc": datetime.now(UTC).isoformat(),
                "active_scene": scene_id,
                "active_scene_position": position,
                "completed_scenes": len(completed_scenes),
            },
        )
        names = ["t2"] + [f"scale_p{part}" for part in range(args.partitions)]
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_job,
                    scene_id,
                    name,
                    command_for(scene_id, name, args.partitions),
                ): name
                for name in names
            }
            for future in as_completed(futures):
                value = future.result()
                results.append(value)
                print(json.dumps({"scene": scene_id, **value}), flush=True)
        audit = structural_scene_audit(scene_id)
        passed = all(item["returncode"] == 0 for item in results) and audit["passed"]
        atomic(
            scene_state,
            {
                **started,
                "status": "complete" if passed else "operational_failure",
                "completed_utc": datetime.now(UTC).isoformat(),
                "jobs": sorted(results, key=lambda item: item["name"]),
                "audit": audit,
            },
        )
        if not passed:
            atomic(
                campaign_state,
                {
                    **load(campaign_state),
                    "status": "operational_failure",
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "scene_id": scene_id,
                    "scene_position": position,
                    "audit": audit,
                },
            )
            return 2
        completed_scenes.append(scene_id)

    final = {
        **load(campaign_state),
        "status": "complete",
        "passed": len(completed_scenes) == len(scene_ids),
        "completed_utc": datetime.now(UTC).isoformat(),
        "completed_scenes": len(completed_scenes),
        "active_scene": None,
    }
    atomic(campaign_state, final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    return 0 if final["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
