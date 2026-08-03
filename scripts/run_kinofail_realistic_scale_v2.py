#!/usr/bin/env python3
"""Resume-safe launcher for the frozen Kino-Fail realistic scale-v2 collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISAAC_SETUP = Path("/home/eureka/nvidia/isaacsim/setup_conda_env.sh")
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
DEFAULT_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v2.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=ROOT / "outputs/kinofail_realistic/design_scale_v1/pilot_schedule.jsonl")
    parser.add_argument("--scene-registry", type=Path, default=ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/data/kinofail_realistic_scale_formal_pilot_v2.json")
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "outputs/kinofail_realistic/corpus_scale_v2")
    parser.add_argument("--collector", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--one-per-operator", action="store_true")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--operators", nargs="*")
    parser.add_argument("--pair-ids", nargs="*")
    parser.add_argument("--audit-name")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--collector-overwrite",
        action="store_true",
        help=(
            "Pass --overwrite to the collector. Intended only for an explicitly selected "
            "structurally incomplete pair during deterministic repair."
        ),
    )
    args = parser.parse_args()

    schedule = args.schedule.resolve()
    registry = args.scene_registry.resolve()
    protocol_path = args.protocol.resolve()
    collector = args.collector.resolve()
    corpus_root = args.corpus_root.resolve()
    protocol = _json(protocol_path)
    for key, actual in (
        ("schedule_sha256", _sha256(schedule)),
        ("scene_registry_sha256", _sha256(registry)),
        ("collector_sha256", _sha256(collector)),
    ):
        if protocol.get(key) != actual:
            raise RuntimeError(f"frozen protocol {key} mismatch")

    records = _jsonl(schedule)
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record["counterfactual_group_id"]), []).append(record)
    pairs = []
    for pair_id, rows in groups.items():
        if len(rows) != 2 or len({row["condition"] for row in rows}) != 2:
            raise RuntimeError(f"invalid scheduled pair: {pair_id}")
        pairs.append((pair_id, rows[0]["target_operator"]))
    pairs.sort()
    if args.operators:
        selected = set(args.operators)
        pairs = [item for item in pairs if item[1] in selected]
    if args.pair_ids:
        selected_pairs = set(args.pair_ids)
        pairs = [item for item in pairs if item[0] in selected_pairs]
    if args.one_per_operator:
        first = {}
        for pair_id, operator_id in pairs:
            first.setdefault(operator_id, (pair_id, operator_id))
        pairs = sorted(first.values(), key=lambda item: item[1])
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]

    corpus_root.mkdir(parents=True, exist_ok=True)
    audit_dir = corpus_root / "launcher_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / (
        args.audit_name
        if args.audit_name
        else ("smoke_11.json" if args.one_per_operator else "full_collection.json")
    )
    prior = _json(audit_path) if audit_path.is_file() else {"attempts": []}
    attempts = list(prior.get("attempts", []))
    attempted_ids = {row["counterfactual_group_id"] for row in attempts}

    for index, (pair_id, operator_id) in enumerate(pairs, 1):
        summary_path = corpus_root / "pair_summaries" / f"{pair_id}.json"
        if summary_path.is_file() and _json(summary_path).get("passed") is True:
            print(json.dumps({"progress": f"{index}/{len(pairs)}", "pair": pair_id, "operator": operator_id, "status": "skip_passed"}), flush=True)
            continue
        if pair_id in attempted_ids:
            print(json.dumps({"progress": f"{index}/{len(pairs)}", "pair": pair_id, "operator": operator_id, "status": "skip_recorded_failure"}), flush=True)
            continue
        command = [
            "/bin/bash", "-lc",
            'source "$1" >/dev/null 2>&1 && exec "$2" "${@:3}"',
            "kinofail-scale", str(ISAAC_SETUP), str(PYTHON), str(collector),
            "--schedule", str(schedule), "--scene-registry", str(registry),
            "--protocol", str(protocol_path), "--corpus-root", str(corpus_root),
            "--counterfactual-group-id", pair_id, "--headless",
        ]
        if args.collector_overwrite:
            command.append("--overwrite")
        print(json.dumps({"progress": f"{index}/{len(pairs)}", "pair": pair_id, "operator": operator_id, "status": "running"}), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        summary = _json(summary_path) if summary_path.is_file() else {}
        row = {
            "counterfactual_group_id": pair_id,
            "operator_id": operator_id,
            "returncode": completed.returncode,
            "summary_exists": summary_path.is_file(),
            "passed": summary.get("passed") is True,
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        attempts.append(row)
        attempted_ids.add(pair_id)
        audit = {
            "schema_version": "kinofail.realistic-scale-v2-launcher.v1",
            "updated_utc": datetime.now(UTC).isoformat(),
            "mode": "one_per_operator" if args.one_per_operator else "full",
            "protocol": str(protocol_path), "protocol_sha256": _sha256(protocol_path),
            "selected_pairs": len(pairs), "attempts": attempts,
        }
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({**row, "progress": f"{index}/{len(pairs)}"}), flush=True)
        if not row["passed"] and args.stop_on_failure:
            return 2

    selected_ids = {pair_id for pair_id, _ in pairs}
    passed_ids = {
        pair_id for pair_id in selected_ids
        if (corpus_root / "pair_summaries" / f"{pair_id}.json").is_file()
        and _json(corpus_root / "pair_summaries" / f"{pair_id}.json").get("passed") is True
    }
    return 0 if passed_ids == selected_ids else 2


if __name__ == "__main__":
    raise SystemExit(main())
