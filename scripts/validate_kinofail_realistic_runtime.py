#!/usr/bin/env python3
"""Validate Kino-Fail runtime artifacts and emit only evaluation-eligible records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kino_vla.data.runtime_manifest import audit_runtime_corpus  # noqa: E402
from kino_vla.utils.config import CONFIGS_DIR  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule", default="outputs/kinofail_realistic/design_v1/pilot_schedule.jsonl"
    )
    parser.add_argument("--corpus-root", default="outputs/kinofail_realistic/corpus_v1")
    parser.add_argument("--config", default="data/kinofail_realistic.yaml")
    parser.add_argument("--out", default="outputs/kinofail_realistic/runtime_audit/pilot")
    parser.add_argument("--write-validated", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    schedule_path = Path(args.schedule).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = CONFIGS_DIR / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gates = config.get("runtime_quality_gates", {})
    audit = audit_runtime_corpus(
        _read_jsonl(schedule_path),
        corpus_root=Path(args.corpus_root),
        gate_overrides=gates,
        write_validated=args.write_validated,
        require_complete=args.require_complete,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    eligible_path = output / "evaluation_eligible.jsonl"
    with eligible_path.open("w", encoding="utf-8") as stream:
        for record in audit.records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    summary = dict(audit.summary)
    summary["schedule_path"] = str(schedule_path)
    summary["schedule_sha256"] = _sha256(schedule_path)
    summary["runtime_config_path"] = str(config_path.resolve())
    summary["runtime_config_sha256"] = _sha256(config_path)
    summary["eligible_manifest"] = str(eligible_path.resolve())
    summary["eligible_manifest_sha256"] = _sha256(eligible_path)
    summary_path = output / "runtime_audit.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not audit.passed:
        raise SystemExit("Kino-Fail runtime audit failed")


if __name__ == "__main__":
    main()
