#!/usr/bin/env python3
"""Collect and verify the nine-pair O3 temporal-window repair overlay."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
LAUNCHER = ROOT / "scripts/run_kinofail_realistic_scale_v2.py"
SCHEDULE = ROOT / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
REGISTRY = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
PROTOCOL = ROOT / "configs/data/kinofail_realistic_o3_temporal_repair_formal_v1.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_o3_temporal_repair_v1.py"
CORPUS = ROOT / "outputs/kinofail_realistic/corpus_scale_v8_o3_temporal_repair_v1"


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


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


def _accepted(pair_id: str, protocol_id: str) -> tuple[bool, list[str]]:
    from kino_vla.data.realistic_snapshots import _runtime_is_accepted

    rows = [
        json.loads(line)
        for line in SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    schedule_by_episode = {str(row["episode_id"]): row for row in rows}
    summary = _json(CORPUS / "pair_summaries" / f"{pair_id}.json")
    results = summary.get("results")
    issues: list[str] = []
    if not isinstance(results, list) or len(results) != 2:
        return False, ["missing_or_incomplete_pair_summary"]
    conditions: set[str] = set()
    nuisances: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            issues.append("invalid_pair_result")
            continue
        condition = str(result.get("condition"))
        conditions.add(condition)
        record = schedule_by_episode.get(str(result.get("episode_id")))
        manifest = _json(Path(str(result.get("manifest", ""))))
        if record is None or not manifest:
            issues.append(f"missing_schedule_or_manifest:{condition}")
            continue
        if (
            manifest.get("collection", {})
            .get("formal_protocol", {})
            .get("protocol_id")
            != protocol_id
        ):
            issues.append(f"repair_protocol_mismatch:{condition}")
        repair = manifest.get("collection", {}).get("o3_temporal_repair", {})
        if (
            repair.get("pair_shared_pre_operator_steps") != 30
            or repair.get("pre_operator_duration_s") != 0.6
        ):
            issues.append(f"repair_readback_mismatch:{condition}")
        nuisance = manifest.get("collection", {}).get("physical_nuisance")
        if not isinstance(nuisance, dict):
            issues.append(f"physical_nuisance_missing:{condition}")
        else:
            nuisances.append(nuisance)
        if not _runtime_is_accepted(
            record,
            manifest,
            allowed_suffixes=[
                "appearance_effect_too_small",
                "rgb_spatial_contrast_too_low",
            ],
        ):
            issues.extend(
                str(value)
                for value in manifest.get("runtime_validation", {}).get("issues", [])
            )
    if conditions != {"anomaly", "nominal_counterfactual"}:
        issues.append("condition_pair_incomplete")
    if len(nuisances) == 2 and (
        nuisances[0] != nuisances[1]
        or nuisances[0].get("profile_index") != 1
        or nuisances[0].get("pair_shared") is not True
    ):
        issues.append("repair_pair_nuisance_mismatch")
    return not issues, sorted(set(issues))


def main() -> int:
    protocol = _json(PROTOCOL)
    pair_ids = [str(value) for value in protocol["allowed"]["counterfactual_group_ids"]]
    if len(pair_ids) != 9:
        raise RuntimeError(f"expected nine repair pairs, found {len(pair_ids)}")
    shards = [pair_ids[0::2], pair_ids[1::2]]
    processes: list[subprocess.Popen[bytes]] = []
    for index, shard in enumerate(shards):
        command = _command(shard, f"full_collection_shard{index}.json")
        print(
            json.dumps(
                {
                    "shard": index,
                    "pairs": len(shard),
                    "first_pair": shard[0],
                    "last_pair": shard[-1],
                }
            ),
            flush=True,
        )
        processes.append(subprocess.Popen(command, cwd=ROOT))
    returncodes = [process.wait() for process in processes]
    checks = {
        pair_id: _accepted(pair_id, str(protocol["protocol_id"]))
        for pair_id in pair_ids
    }
    failures = {
        pair_id: issues
        for pair_id, (accepted, issues) in checks.items()
        if not accepted
    }
    result = {
        "status": "complete" if not failures else "blocked",
        "pairs": len(pair_ids),
        "accepted_pairs": len(pair_ids) - len(failures),
        "returncodes": returncodes,
        "failures": failures,
        "claim_guard": (
            "The overlay changes activation timing for the complete frozen design cell only; "
            "mechanism strength and outcome are not tuned."
        ),
        "a8_in_scope": False,
    }
    audit = CORPUS / "repair_audit.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
