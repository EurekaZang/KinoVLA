#!/usr/bin/env python3
"""Preflight one formal pair, then run/resume the complete scale-v8 collection."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
LAUNCHER = "scripts/run_kinofail_realistic_scale_v2.py"
COMMON = [
    "--schedule",
    "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl",
    "--scene-registry",
    "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
    "--protocol",
    "configs/data/kinofail_realistic_scale_v8_replication_formal_v3.json",
    "--corpus-root",
    "outputs/kinofail_realistic/corpus_scale_v8_replication_v1",
    "--collector",
    "scripts/isaac_collect_kinofail_realistic_pair_v8.py",
]
AUDIT = (
    ROOT
    / "outputs/kinofail_realistic/corpus_scale_v8_replication_v1/"
    "launcher_audits/preflight_then_full.json"
)
SCHEDULE = (
    ROOT
    / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
)
CORPUS = ROOT / "outputs/kinofail_realistic/corpus_scale_v8_replication_v1"
PROTOCOL_ID = "kinofail_realistic_scale_1980_v8_replication_v3"


def _command(extra: list[str]) -> list[str]:
    return [str(PYTHON), LAUNCHER, *COMMON, *extra]


def _run(extra: list[str]) -> dict:
    command = _command(extra)
    print(json.dumps({"running": command}), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return {"command": command, "returncode": completed.returncode}


def _pair_ids() -> list[str]:
    rows = [
        json.loads(line)
        for line in SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return sorted({str(row["counterfactual_group_id"]) for row in rows})


def _runtime_complete(pair_id: str) -> bool:
    summary_path = CORPUS / "pair_summaries" / f"{pair_id}.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary.get("results", [])
    if len(results) != 2:
        return False
    nuisances = []
    conditions = set()
    for result in results:
        manifest_path = Path(str(result.get("manifest", "")))
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        conditions.add(str(result.get("condition")))
        formal = manifest.get("collection", {}).get("formal_protocol", {})
        nuisance = manifest.get("collection", {}).get("physical_nuisance")
        if formal.get("protocol_id") != PROTOCOL_ID or not isinstance(nuisance, dict):
            return False
        nuisances.append(nuisance)
    return (
        conditions == {"anomaly", "nominal_counterfactual"}
        and nuisances[0] == nuisances[1]
        and nuisances[0].get("pair_shared") is True
    )


def _run_parallel(shards: list[list[str]], *, audit_prefix: str) -> dict:
    processes = []
    commands = []
    for index, pair_ids in enumerate(shards):
        command = _command(
            [
                "--pair-ids",
                *pair_ids,
                "--audit-name",
                f"{audit_prefix}_shard{index}.json",
            ]
        )
        commands.append(command)
        print(
            json.dumps(
                {
                    "running_shard": index,
                    "pairs": len(pair_ids),
                    "first_pair": pair_ids[0] if pair_ids else None,
                    "last_pair": pair_ids[-1] if pair_ids else None,
                }
            ),
            flush=True,
        )
        processes.append(subprocess.Popen(command, cwd=ROOT))
    returncodes = [process.wait() for process in processes]
    return {
        "commands": commands,
        "pair_counts": [len(value) for value in shards],
        "returncodes": returncodes,
    }


def _write(stages: list[dict], status: str) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(
        json.dumps(
            {
                "schema_version": "kinofail.scale-v8-preflight-then-full.v1",
                "updated_utc": datetime.now(UTC).isoformat(),
                "status": status,
                "stages": stages,
                "policy": (
                    "One single-process pair and two concurrent pairs must write complete, "
                    "pair-shared, protocol-bound manifests. QA-negative but structurally complete "
                    "preflight pairs do not block. Full collection uses two disjoint shards."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    pair_ids = _pair_ids()
    if len(pair_ids) != 990:
        raise RuntimeError(f"expected 990 pairs, found {len(pair_ids)}")
    stages = [
        _run(
            [
                "--pair-ids",
                pair_ids[0],
                "--stop-on-failure",
                "--audit-name",
                "physical_nuisance_preflight.json",
            ]
        )
    ]
    if not _runtime_complete(pair_ids[0]):
        _write(stages, "blocked_by_preflight")
        return 3
    concurrent_ids = pair_ids[1:3]
    concurrency = _run_parallel(
        [[concurrent_ids[0]], [concurrent_ids[1]]],
        audit_prefix="physical_nuisance_concurrency_preflight",
    )
    stages.append({"stage": "concurrency_preflight", **concurrency})
    if not all(_runtime_complete(pair_id) for pair_id in concurrent_ids):
        _write(stages, "blocked_by_concurrency_preflight")
        return 4
    _write(stages, "preflight_passed")
    remaining = pair_ids[3:]
    shards = [remaining[0::2], remaining[1::2]]
    full = _run_parallel(shards, audit_prefix="full_collection")
    stages.append({"stage": "parallel_full_collection", **full})
    status = (
        "complete"
        if all(code == 0 for code in full["returncodes"])
        else "complete_with_excluded_pairs"
    )
    _write(stages, status)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
