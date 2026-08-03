#!/usr/bin/env python3
"""Retry only structurally incomplete scale-v8 pairs before A0 freezing.

Scientific QA failures with two complete, protocol-bound manifests are retained exactly as
observed.  This repair is only for process/interrupted-write failures that would otherwise leave
the frozen 990-pair schedule incomplete.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
LAUNCHER = ROOT / "scripts/run_kinofail_realistic_scale_v2.py"
DEFAULT_SCHEDULE = (
    ROOT / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
)
DEFAULT_CORPUS = ROOT / "outputs/kinofail_realistic/corpus_scale_v8_replication_v1"
DEFAULT_PROTOCOL = ROOT / "configs/data/kinofail_realistic_scale_v8_replication_formal_v3.json"
DEFAULT_REGISTRY = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
DEFAULT_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v8.py"
PROTOCOL_ID = "kinofail_realistic_scale_1980_v8_replication_v3"


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _pair_ids(schedule: Path) -> list[str]:
    rows = [
        json.loads(line)
        for line in schedule.read_text(encoding="utf-8").splitlines()
        if line
    ]
    pair_ids = sorted({str(row["counterfactual_group_id"]) for row in rows})
    if len(rows) != 2 * len(pair_ids):
        raise RuntimeError("schedule is not a one-to-one counterfactual pairing")
    return pair_ids


def _structurally_complete(corpus: Path, pair_id: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    summary_path = corpus / "pair_summaries" / f"{pair_id}.json"
    summary = _json(summary_path)
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 2:
        return False, ["missing_or_incomplete_pair_summary"]
    conditions: set[str] = set()
    nuisances: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            issues.append("invalid_result_record")
            continue
        conditions.add(str(result.get("condition")))
        manifest_path = Path(str(result.get("manifest", "")))
        if not manifest_path.is_file():
            issues.append(f"missing_manifest:{result.get('condition')}")
            continue
        manifest = _json(manifest_path)
        formal = manifest.get("collection", {}).get("formal_protocol", {})
        nuisance = manifest.get("collection", {}).get("physical_nuisance")
        if formal.get("protocol_id") != PROTOCOL_ID:
            issues.append(f"protocol_mismatch:{result.get('condition')}")
        if not isinstance(nuisance, dict):
            issues.append(f"physical_nuisance_missing:{result.get('condition')}")
        else:
            nuisances.append(nuisance)
    if conditions != {"anomaly", "nominal_counterfactual"}:
        issues.append("condition_pair_incomplete")
    if len(nuisances) == 2:
        if nuisances[0] != nuisances[1]:
            issues.append("pair_shared_physical_nuisance_mismatch")
        if nuisances[0].get("pair_shared") is not True:
            issues.append("pair_shared_flag_missing")
    return not issues, issues


def _run_shards(
    pair_ids: list[str],
    *,
    schedule: Path,
    registry: Path,
    protocol: Path,
    corpus: Path,
    collector: Path,
    round_index: int,
) -> list[int]:
    shards = [pair_ids[0::2], pair_ids[1::2]]
    processes: list[subprocess.Popen[bytes]] = []
    for shard_index, shard in enumerate(shards):
        if not shard:
            continue
        command = [
            str(PYTHON),
            str(LAUNCHER),
            "--schedule",
            str(schedule),
            "--scene-registry",
            str(registry),
            "--protocol",
            str(protocol),
            "--corpus-root",
            str(corpus),
            "--collector",
            str(collector),
            "--pair-ids",
            *shard,
            "--audit-name",
            f"structural_repair_round{round_index}_shard{shard_index}.json",
            "--collector-overwrite",
        ]
        print(
            json.dumps(
                {
                    "round": round_index,
                    "shard": shard_index,
                    "pairs": len(shard),
                    "first_pair": shard[0],
                    "last_pair": shard[-1],
                }
            ),
            flush=True,
        )
        processes.append(subprocess.Popen(command, cwd=ROOT))
    return [process.wait() for process in processes]


def _write_audit(
    path: Path,
    *,
    scheduled_pairs: int,
    rounds: list[dict[str, Any]],
    remaining: dict[str, list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "kinofail.scale-v8-structural-repair.v1",
        "updated_utc": datetime.now(UTC).isoformat(),
        "status": "complete" if not remaining else "blocked",
        "policy": (
            "Retry interrupted or structurally incomplete pairs only. Complete pairs remain "
            "immutable even when scientific/local QA is negative."
        ),
        "scheduled_pairs": scheduled_pairs,
        "rounds": rounds,
        "remaining_structurally_incomplete": remaining,
        "all_pairs_structurally_complete": not remaining,
        "a8_in_scope": False,
    }
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--scene-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--collector", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--audit",
        type=Path,
        default=(
            DEFAULT_CORPUS
            / "launcher_audits/structural_repair.json"
        ),
    )
    args = parser.parse_args()
    if args.max_rounds < 1:
        raise ValueError("max-rounds must be positive")
    schedule = args.schedule.resolve()
    registry = args.scene_registry.resolve()
    protocol = args.protocol.resolve()
    corpus = args.corpus_root.resolve()
    collector = args.collector.resolve()
    pair_ids = _pair_ids(schedule)
    if len(pair_ids) != 990:
        raise RuntimeError(f"expected 990 frozen pairs, found {len(pair_ids)}")

    rounds: list[dict[str, Any]] = []
    for round_index in range(1, args.max_rounds + 1):
        incomplete = {
            pair_id: issues
            for pair_id in pair_ids
            if not (complete := _structurally_complete(corpus, pair_id))[0]
            for issues in [complete[1]]
        }
        if not incomplete:
            _write_audit(
                args.audit.resolve(),
                scheduled_pairs=len(pair_ids),
                rounds=rounds,
                remaining={},
            )
            print(json.dumps({"status": "complete", "scheduled_pairs": len(pair_ids)}))
            return 0
        returncodes = _run_shards(
            list(incomplete),
            schedule=schedule,
            registry=registry,
            protocol=protocol,
            corpus=corpus,
            collector=collector,
            round_index=round_index,
        )
        rounds.append(
            {
                "round": round_index,
                "attempted_pairs": len(incomplete),
                "returncodes": returncodes,
                "issues_before_retry": incomplete,
            }
        )
        _write_audit(
            args.audit.resolve(),
            scheduled_pairs=len(pair_ids),
            rounds=rounds,
            remaining=incomplete,
        )

    remaining = {
        pair_id: issues
        for pair_id in pair_ids
        if not (complete := _structurally_complete(corpus, pair_id))[0]
        for issues in [complete[1]]
    }
    _write_audit(
        args.audit.resolve(),
        scheduled_pairs=len(pair_ids),
        rounds=rounds,
        remaining=remaining,
    )
    print(
        json.dumps(
            {
                "status": "complete" if not remaining else "blocked",
                "remaining_structurally_incomplete": len(remaining),
            }
        )
    )
    return 0 if not remaining else 2


if __name__ == "__main__":
    raise SystemExit(main())
