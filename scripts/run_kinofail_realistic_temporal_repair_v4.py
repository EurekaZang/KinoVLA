#!/usr/bin/env python3
"""Collect and verify the 135-pair O3-severe/O8-all repair overlay."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/run_kinofail_realistic_o3_temporal_repair_v1.py"
PROTOCOL = ROOT / "configs/data/kinofail_realistic_temporal_repair_formal_v4.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_temporal_repair_v4.py"
CORPUS = ROOT / "outputs/kinofail_realistic/corpus_scale_v8_temporal_repair_v4"


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _load_base():
    spec = importlib.util.spec_from_file_location("kinofail_temporal_repair_runner_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _in_scope(record: dict[str, Any]) -> bool:
    operator = str(record.get("target_operator"))
    return (
        operator == "O8_invisible_collider"
        or (operator == "O3_collapse" and record.get("severity_id") == "severe")
    )


def _accepted(module, pair_id: str, protocol_id: str) -> tuple[bool, list[str]]:
    from kino_vla.data.realistic_snapshots import _runtime_is_accepted

    rows = [
        json.loads(line)
        for line in module.SCHEDULE.read_text(encoding="utf-8").splitlines()
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
        if not _in_scope(record):
            issues.append(f"episode_outside_repair_scope:{result.get('episode_id')}")
        if (
            manifest.get("collection", {})
            .get("formal_protocol", {})
            .get("protocol_id")
            != protocol_id
        ):
            issues.append(f"repair_protocol_mismatch:{condition}")
        repair = manifest.get("collection", {}).get("temporal_window_repair", {})
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
        or nuisances[0].get("profile_index") not in range(5)
        or nuisances[0].get("pair_shared") is not True
    ):
        issues.append("repair_pair_nuisance_mismatch")
    return not issues, sorted(set(issues))


def main() -> int:
    module = _load_base()
    module.PROTOCOL = PROTOCOL
    module.COLLECTOR = COLLECTOR
    module.CORPUS = CORPUS
    module._accepted = lambda pair_id, protocol_id: _accepted(
        module, pair_id, protocol_id
    )
    protocol = _json(PROTOCOL)
    protocol_id = str(protocol["protocol_id"])
    pair_ids = [str(value) for value in protocol["allowed"]["counterfactual_group_ids"]]
    if len(pair_ids) != 135:
        raise RuntimeError(f"expected 135 repair pairs, found {len(pair_ids)}")
    rows = [
        json.loads(line)
        for line in module.SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    representative = {
        str(row["counterfactual_group_id"]): row
        for row in rows
        if row["condition"] == "anomaly"
    }
    preflight = [
        next(
            pair_id
            for pair_id in pair_ids
            if representative[pair_id]["target_operator"] == "O3_collapse"
        ),
        next(
            pair_id
            for pair_id in pair_ids
            if representative[pair_id]["target_operator"] == "O8_invisible_collider"
            and representative[pair_id]["severity_id"] == "moderate"
        ),
    ]

    def run_shards(shards: list[list[str]], prefix: str) -> list[int]:
        processes: list[subprocess.Popen[bytes]] = []
        for index, shard in enumerate(shards):
            if not shard:
                continue
            command = module._command(shard, f"{prefix}_shard{index}.json")
            print(
                json.dumps(
                    {
                        "stage": prefix,
                        "shard": index,
                        "pairs": len(shard),
                        "first_pair": shard[0],
                        "last_pair": shard[-1],
                    }
                ),
                flush=True,
            )
            processes.append(subprocess.Popen(command, cwd=ROOT))
        return [process.wait() for process in processes]

    preflight_returncodes = run_shards(
        [[preflight[0]], [preflight[1]]], "preflight"
    )
    preflight_failures = {
        pair_id: issues
        for pair_id in preflight
        for accepted, issues in [_accepted(module, pair_id, protocol_id)]
        if not accepted
    }
    if preflight_failures:
        result = {
            "status": "blocked_at_preflight",
            "pairs": len(pair_ids),
            "accepted_pairs": 0,
            "preflight_returncodes": preflight_returncodes,
            "failures": preflight_failures,
            "a8_in_scope": False,
        }
        audit = CORPUS / "repair_audit.json"
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 2

    remaining = [pair_id for pair_id in pair_ids if pair_id not in set(preflight)]
    returncodes = run_shards([remaining[0::2], remaining[1::2]], "full_collection")
    checks = {
        pair_id: _accepted(module, pair_id, protocol_id)
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
        "preflight_pairs": preflight,
        "preflight_returncodes": preflight_returncodes,
        "returncodes": returncodes,
        "failures": failures,
        "claim_guard": (
            "One uniform pre-outcome temporal amendment covers complete mechanism-defined "
            "blocks; operator strengths and outcomes are not tuned."
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
