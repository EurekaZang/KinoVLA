#!/usr/bin/env python3
"""Collect the frozen A5 physical-realization arm in two disjoint GPU-safe shards."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
LAUNCHER = ROOT / "scripts/run_kinofail_realistic_scale_v2.py"
SCHEDULE = ROOT / "outputs/kinofail_realistic/design_a5_realization_v1/schedule.jsonl"
REGISTRY = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
PROTOCOL = ROOT / "configs/data/kinofail_realistic_a5_realization_formal_v2.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_a5_realization_v2.py"
CORPUS = ROOT / "outputs/kinofail_realistic/corpus_a5_realization_v1"
PROTOCOL_ID = "kinofail_realistic_a5_realization_360_v2"


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _pair_ids() -> list[str]:
    rows = [
        json.loads(line)
        for line in SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    pair_ids = sorted({str(row["counterfactual_group_id"]) for row in rows})
    if len(rows) != 360 or len(pair_ids) != 180:
        raise RuntimeError(
            f"expected frozen 360-row/180-pair realization schedule, got "
            f"{len(rows)}/{len(pair_ids)}"
        )
    return pair_ids


def _structurally_complete(pair_id: str) -> bool:
    summary = _json(CORPUS / "pair_summaries" / f"{pair_id}.json")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 2:
        return False
    conditions: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            return False
        conditions.add(str(result.get("condition")))
        manifest = _json(Path(str(result.get("manifest", ""))))
        if (
            not manifest
            or manifest.get("collection", {})
            .get("formal_protocol", {})
            .get("protocol_id")
            != PROTOCOL_ID
        ):
            return False
    return conditions == {"anomaly", "nominal_counterfactual"}


def _command(pair_ids: list[str], audit_name: str) -> list[str]:
    return [
        str(PYTHON),
        str(LAUNCHER),
        "--schedule",
        str(SCHEDULE),
        "--scene-registry",
        str(REGISTRY),
        "--protocol",
        str(PROTOCOL),
        "--corpus-root",
        str(CORPUS),
        "--collector",
        str(COLLECTOR),
        "--pair-ids",
        *pair_ids,
        "--audit-name",
        audit_name,
    ]


def _parallel(shards: list[list[str]], prefix: str) -> list[int]:
    processes: list[subprocess.Popen[bytes]] = []
    for index, pair_ids in enumerate(shards):
        if not pair_ids:
            continue
        command = _command(pair_ids, f"{prefix}_shard{index}.json")
        print(
            json.dumps(
                {
                    "stage": prefix,
                    "shard": index,
                    "pairs": len(pair_ids),
                    "first_pair": pair_ids[0],
                    "last_pair": pair_ids[-1],
                }
            ),
            flush=True,
        )
        processes.append(subprocess.Popen(command, cwd=ROOT))
    return [process.wait() for process in processes]


def _consolidate(pair_ids: list[str], sources: list[Path]) -> int:
    attempt_by_pair: dict[str, dict[str, Any]] = {}
    for path in sources:
        for row in _json(path).get("attempts", []):
            pair_id = str(row.get("counterfactual_group_id"))
            if pair_id in pair_ids and _structurally_complete(pair_id):
                attempt_by_pair[pair_id] = row
    audit = {
        "schema_version": "kinofail.realistic-scale-v2-launcher.v1",
        "updated_utc": datetime.now(UTC).isoformat(),
        "mode": "full",
        "protocol": str(PROTOCOL),
        "selected_pairs": len(pair_ids),
        "attempts": [attempt_by_pair[pair_id] for pair_id in pair_ids if pair_id in attempt_by_pair],
        "parallel_collection": {
            "shards": 2,
            "policy": (
                "Only structurally complete attempts are consolidated. The downstream serial "
                "resume therefore retries process/write interruptions but not complete negative QA."
            ),
        },
    }
    path = CORPUS / "launcher_audits/full_collection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(attempt_by_pair)


def main() -> int:
    pair_ids = _pair_ids()
    CORPUS.mkdir(parents=True, exist_ok=True)
    preflight_ids = pair_ids[:2]
    preflight_codes = _parallel(
        [[preflight_ids[0]], [preflight_ids[1]]],
        "parallel_preflight",
    )
    if not all(_structurally_complete(pair_id) for pair_id in preflight_ids):
        print(
            json.dumps(
                {
                    "status": "blocked_by_concurrency_preflight",
                    "returncodes": preflight_codes,
                }
            )
        )
        return 3
    remaining = pair_ids[2:]
    full_codes = _parallel([remaining[0::2], remaining[1::2]], "parallel_full")
    audit_dir = CORPUS / "launcher_audits"
    sources = [
        audit_dir / "parallel_preflight_shard0.json",
        audit_dir / "parallel_preflight_shard1.json",
        audit_dir / "parallel_full_shard0.json",
        audit_dir / "parallel_full_shard1.json",
    ]
    complete = _consolidate(pair_ids, sources)
    result = {
        "status": "complete" if complete == len(pair_ids) else "resume_required",
        "structurally_complete_pairs": complete,
        "scheduled_pairs": len(pair_ids),
        "preflight_returncodes": preflight_codes,
        "full_returncodes": full_codes,
    }
    print(json.dumps(result, indent=2))
    # A return code of two asks the already queued serial launcher to finish only missing pairs.
    return 0 if complete == len(pair_ids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
