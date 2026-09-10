#!/usr/bin/env python3
"""Collect the sealed supplement, then resume the all-191 campaign at 2 workers."""

from __future__ import annotations

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
SCENE = "kino4c_production_020"
SUPPLEMENT = DESIGN / "schedules/operational_supplement_v1"
SCHEDULE = SUPPLEMENT / "scenes" / SCENE / "scale/schedule.jsonl"
PROTOCOL = SUPPLEMENT / "scenes" / SCENE / "scale/collection_protocol.json"
REGISTRY = DESIGN / "design/scene_registry.json"
LOCK = Path(
    "/data/eureka/kinofail_kino_v4_confirmation_v1/assets/terrain_pbr/"
    "terrain_assets.lock.json"
)
CORPUS = Path(
    "/data/eureka/kinofail_kino_v4_all191_scale_t2_extension_v2/corpus"
) / SCENE
AMENDMENT = ROOT / (
    "outputs/freeze/kino_v4_all191_scale_provenance_v2/"
    "amendment_manifest.json"
)
FREEZE = ROOT / (
    "outputs/freeze/kino_v4_all191_operational_supplement_v1/"
    "freeze_manifest.json"
)
PAIR_RUNNER = ROOT / "scripts/run_kinofail_reconfirmation_pair_partition_v2.py"
PAIR_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_kino_v4_confirmation_pair_v1.py"
CAMPAIGN = ROOT / "scripts/run_kino_v4_all191_scale_t2_campaign_v1.py"
OUT = DESIGN / "orchestration/operational_recovery_v1"
STATE = OUT / "state.json"


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


def atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def command(partition: int) -> list[str]:
    return [
        sys.executable,
        str(PAIR_RUNNER),
        "--schedule",
        str(SCHEDULE),
        "--scene-registry",
        str(REGISTRY),
        "--protocol",
        str(PROTOCOL),
        "--asset-lock",
        str(LOCK),
        "--corpus-root",
        str(CORPUS),
        "--collector",
        str(PAIR_COLLECTOR),
        "--operational-amendment",
        str(AMENDMENT),
        "--partition-index",
        str(partition),
        "--partition-count",
        "2",
    ]


def run_partition(partition: int) -> dict[str, Any]:
    log = OUT / f"supplement_partition_{partition}_of_2.log"
    if log.exists():
        raise FileExistsError(log)
    started = datetime.now(UTC).isoformat()
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command(partition),
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "partition": partition,
        "returncode": result.returncode,
        "started_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "log": str(log),
        "log_sha256": sha256(log),
    }


def supplement_audit() -> dict[str, Any]:
    schedule_rows = rows(SCHEDULE)
    pair_ids = {str(value["counterfactual_group_id"]) for value in schedule_rows}
    complete = set()
    for pair_id in pair_ids:
        path = CORPUS / "pair_summaries" / f"{pair_id}.json"
        if not path.is_file():
            continue
        value = load(path)
        results = value.get("results")
        if (
            value.get("counterfactual_group_id") == pair_id
            and isinstance(results, list)
            and len(results) == 2
            and {str(item.get("condition")) for item in results}
            == {"nominal_counterfactual", "anomaly"}
            and all(
                Path(str(item.get("manifest", ""))).is_file()
                for item in results
            )
        ):
            complete.add(pair_id)
    attrition = {
        pair_id
        for pair_id in pair_ids
        if (CORPUS / "operational_attrition" / f"{pair_id}.json").is_file()
    }
    operators = {
        str(value["target_operator"])
        for value in schedule_rows
        if str(value["counterfactual_group_id"]) in complete
    }
    planned_operators = {
        str(value["target_operator"]) for value in schedule_rows
    }
    passed = (
        len(pair_ids) == 19
        and complete == pair_ids
        and not attrition
        and operators == planned_operators
    )
    return {
        "passed": passed,
        "planned_pairs": len(pair_ids),
        "complete_pairs": len(complete),
        "missing_pair_ids": sorted(pair_ids - complete),
        "attrited_pair_ids": sorted(attrition),
        "complete_operator_count": len(operators),
        "planned_operator_count": len(planned_operators),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lock_handle = (OUT / "recovery.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another operational recovery is active") from exc
    for path in (
        SCHEDULE,
        PROTOCOL,
        REGISTRY,
        LOCK,
        AMENDMENT,
        FREEZE,
        PAIR_RUNNER,
        PAIR_COLLECTOR,
        CAMPAIGN,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze = load(FREEZE)
    if (
        freeze.get("passed") is not True
        or freeze.get("status") != "frozen_before_supplemental_acquisition"
        or freeze.get("artifacts", {}).get("supplemental_schedule_sha256")
        != sha256(SCHEDULE)
        or freeze.get("artifacts", {}).get("supplemental_protocol_sha256")
        != sha256(PROTOCOL)
        or freeze.get("artifacts", {}).get("pair_runner_sha256")
        != sha256(PAIR_RUNNER)
    ):
        raise RuntimeError("supplement freeze is invalid")

    state = {
        "schema_version": "kinofail.kino-v4-operational-recovery-run.v1",
        "status": "collecting_supplement",
        "started_utc": datetime.now(UTC).isoformat(),
        "scene_id": SCENE,
        "workers": 2,
        "model_predictions_read": False,
        "method_scores_read": False,
        "result_dependent_retry_permitted": False,
        "freeze": str(FREEZE),
        "freeze_sha256": sha256(FREEZE),
    }
    atomic(STATE, state)
    jobs = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_partition, value) for value in range(2)]
        for future in as_completed(futures):
            jobs.append(future.result())
            atomic(
                STATE,
                {
                    **state,
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "completed_partition_jobs": sorted(
                        jobs, key=lambda value: value["partition"]
                    ),
                },
            )
    audit = supplement_audit()
    if not all(job["returncode"] == 0 for job in jobs) or not audit["passed"]:
        atomic(
            STATE,
            {
                **state,
                "status": "supplement_operational_failure",
                "completed_utc": datetime.now(UTC).isoformat(),
                "jobs": sorted(jobs, key=lambda value: value["partition"]),
                "supplement_audit": audit,
            },
        )
        return 0

    main_log = OUT / "main_campaign_workers2.log"
    if main_log.exists():
        raise FileExistsError(main_log)
    atomic(
        STATE,
        {
            **state,
            "status": "resuming_main_campaign",
            "updated_utc": datetime.now(UTC).isoformat(),
            "jobs": sorted(jobs, key=lambda value: value["partition"]),
            "supplement_audit": audit,
            "main_campaign_workers": 2,
        },
    )
    with main_log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            [
                sys.executable,
                str(CAMPAIGN),
                "--partitions",
                "3",
                "--workers",
                "2",
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    campaign_state_path = DESIGN / "orchestration/campaign_v1/campaign_state.json"
    campaign_state = load(campaign_state_path)
    passed = result.returncode == 0 and campaign_state.get("passed") is True
    atomic(
        STATE,
        {
            **load(STATE),
            "status": "complete" if passed else "main_campaign_operational_failure",
            "completed_utc": datetime.now(UTC).isoformat(),
            "main_campaign_returncode": result.returncode,
            "main_campaign_log": str(main_log),
            "main_campaign_log_sha256": sha256(main_log),
            "main_campaign_state": str(campaign_state_path),
            "main_campaign_state_sha256": sha256(campaign_state_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
