#!/usr/bin/env python3
"""Resume-safe launcher for one sealed confirmatory scene shard.

Two launchers may process deterministic disjoint partitions of the same scene.
No overwrite or result-based retry option exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ISAAC_SETUP = Path("/home/eureka/nvidia/isaacsim/setup_conda_env.sh")
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
DEFAULT_COLLECTOR = (
    ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v1.py"
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
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--asset-lock", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--collector", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--partition-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--partition-count", type=int, choices=(2,), default=2)
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()

    schedule = args.schedule.resolve()
    registry = args.scene_registry.resolve()
    protocol_path = args.protocol.resolve()
    asset_lock = args.asset_lock.resolve()
    collector = args.collector.resolve()
    corpus_root = args.corpus_root.resolve()
    protocol = _json(protocol_path)
    for runtime_path in (ISAAC_SETUP, PYTHON):
        if not runtime_path.is_file():
            raise FileNotFoundError(runtime_path)
    for key, actual in (
        ("schedule_sha256", _sha256(schedule)),
        ("scene_registry_sha256", _sha256(registry)),
        ("collector_sha256", _sha256(collector)),
        ("material_lock_sha256", _sha256(asset_lock)),
    ):
        if protocol.get(key) != actual:
            raise RuntimeError(f"sealed protocol {key} mismatch")

    records = _jsonl(schedule)
    scenes = {str(row["scene_family"]) for row in records}
    if len(scenes) != 1:
        raise RuntimeError("confirmatory shard schedule must contain exactly one scene")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["counterfactual_group_id"]), []).append(
            record
        )
    invalid = {
        pair_id: rows
        for pair_id, rows in groups.items()
        if len(rows) != 2
        or {str(row["condition"]) for row in rows}
        != {"nominal_counterfactual", "anomaly"}
    }
    if invalid:
        raise RuntimeError(f"invalid confirmatory pairs: {sorted(invalid)[:3]}")
    pair_ids = sorted(groups)
    selected = [
        pair_id
        for index, pair_id in enumerate(pair_ids)
        if index % args.partition_count == args.partition_index
    ]
    if args.max_pairs is not None:
        selected = selected[: args.max_pairs]
    corpus_root.mkdir(parents=True, exist_ok=True)
    audit_dir = corpus_root / "launcher_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"partition_{args.partition_index}_of_2.json"
    if audit_path.exists():
        prior = _json(audit_path)
        if (
            prior.get("schedule_sha256") != _sha256(schedule)
            or prior.get("selected_pair_ids") != selected
        ):
            raise RuntimeError("existing launcher audit belongs to another shard")
    else:
        prior = {"attempts": []}
    attempts = list(prior.get("attempts", []))
    attempted = {str(row["counterfactual_group_id"]) for row in attempts}

    for index, pair_id in enumerate(selected, start=1):
        summary_path = corpus_root / "pair_summaries" / f"{pair_id}.json"
        if summary_path.is_file() and _json(summary_path).get("passed") is True:
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(selected)}",
                        "pair": pair_id,
                        "status": "skip_sealed_pass",
                    }
                ),
                flush=True,
            )
            continue
        if pair_id in attempted:
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(selected)}",
                        "pair": pair_id,
                        "status": "skip_recorded_terminal_failure",
                    }
                ),
                flush=True,
            )
            continue
        command = [
            "/bin/bash",
            "-lc",
            'source "$1" >/dev/null 2>&1 && exec "$2" "${@:3}"',
            "kinofail-confirmatory",
            str(ISAAC_SETUP),
            str(PYTHON),
            str(collector),
            "--schedule",
            str(schedule),
            "--scene-registry",
            str(registry),
            "--protocol",
            str(protocol_path),
            "--asset-lock",
            str(asset_lock),
            "--corpus-root",
            str(corpus_root),
            "--counterfactual-group-id",
            pair_id,
            "--headless",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        summary = _json(summary_path) if summary_path.is_file() else {}
        result = {
            "counterfactual_group_id": pair_id,
            "returncode": int(completed.returncode),
            "summary_exists": summary_path.is_file(),
            "passed": summary.get("passed") is True,
            "completed_utc": datetime.now(UTC).isoformat(),
            "retry_authorized": False,
        }
        attempts.append(result)
        attempted.add(pair_id)
        audit = {
            "schema_version": "kinofail.confirmatory-shard-launcher.v1",
            "updated_utc": datetime.now(UTC).isoformat(),
            "scene_id": next(iter(scenes)),
            "partition_index": args.partition_index,
            "partition_count": args.partition_count,
            "schedule": str(schedule),
            "schedule_sha256": _sha256(schedule),
            "protocol": str(protocol_path),
            "protocol_sha256": _sha256(protocol_path),
            "collector_sha256": _sha256(collector),
            "material_lock_sha256": _sha256(asset_lock),
            "selected_pair_ids": selected,
            "attempts": attempts,
        }
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps({**result, "progress": f"{index}/{len(selected)}"}),
            flush=True,
        )

    passed = {
        pair_id
        for pair_id in selected
        if (corpus_root / "pair_summaries" / f"{pair_id}.json").is_file()
        and _json(
            corpus_root / "pair_summaries" / f"{pair_id}.json"
        ).get("passed")
        is True
    }
    return 0 if passed == set(selected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
