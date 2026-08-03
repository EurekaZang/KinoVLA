#!/usr/bin/env python3
"""Audit attribution predictions under same-physics terrain texture swaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kino_vla.eval.texture_swap import audit_texture_swap_predictions  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/data/kinofail_realistic.yaml"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/kinofail_realistic/texture_swap/consistency_audit.json"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gates = config["texture_swap_evaluation_gates"]
    audit = audit_texture_swap_predictions(
        _read_jsonl(args.predictions),
        min_group_hard_consistency=float(gates["min_group_hard_prediction_consistency"]),
        min_pairwise_agreement=float(gates["min_pairwise_prediction_agreement"]),
        max_mean_pairwise_jsd=float(gates["max_mean_pairwise_probability_jsd"]),
    )
    payload = {
        "schema_version": "kinofail.texture-swap-audit.v1",
        "passed": audit.passed,
        "issues": list(audit.issues),
        **audit.metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not audit.passed:
        raise SystemExit("texture-swap consistency gate failed")


if __name__ == "__main__":
    main()
