#!/usr/bin/env python3
"""Freeze the snapshot-protocol amendment that selects QA-repaired O9 pairs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> None:
    base_path = ROOT / "configs/eval/kinofail_realistic_snapshot_formal_v3.json"
    repair_path = ROOT / "configs/data/kinofail_realistic_o9_repair_formal_v2.json"
    output = ROOT / "configs/eval/kinofail_realistic_snapshot_formal_v4.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    base, repair = _json(base_path), _json(repair_path)
    pair_ids = [str(value) for value in repair["allowed"]["counterfactual_group_ids"]]
    if not pair_ids:
        raise RuntimeError("repair protocol has no admitted counterfactual pair")
    amended = deepcopy(base)
    amended["protocol_id"] = "realistic_snapshot_formal_scale_v7_repair_overlay_v4"
    amended["status"] = "frozen_after_runtime_qa_before_snapshot_extraction"
    amended["source_corpus_overlays"] = [
        {
            "corpus_root": "outputs/kinofail_realistic/corpus_scale_v7_o9_repair_v2",
            "counterfactual_group_ids": pair_ids,
            "required_collection_protocol_id": repair["protocol_id"],
            "repair_protocol": str(repair_path.relative_to(ROOT)),
            "repair_protocol_sha256": _sha(repair_path),
        }
    ]
    admitted_suffixes = amended["selection"]["allow_nonblocking_runtime_issue_suffixes"]
    if "rgb_spatial_contrast_too_low" not in admitted_suffixes:
        admitted_suffixes.append("rgb_spatial_contrast_too_low")
    amended["amendment"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "base_snapshot_protocol": str(base_path.relative_to(ROOT)),
        "base_snapshot_protocol_sha256": _sha(base_path),
        "scope": "Select only the independently QA-triggered O9 repair pairs from the immutable repair corpus; all other pairs remain in scale-v7. Retain the single synchronized O9 glare sequence as a hard camera nuisance while excluding its failed visual comparison from texture-consistency denominators.",
        "model_architecture_or_split_changed": False,
        "selection_uses_model_outcomes": False,
    }
    output.write_text(json.dumps(amended, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": _sha(output), "repair_pairs": pair_ids}, indent=2))


if __name__ == "__main__":
    main()
