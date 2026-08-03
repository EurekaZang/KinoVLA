#!/usr/bin/env python3
"""Hash the realistic model protocol before formal model inputs and predictions exist."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = ROOT / "configs/eval/kinofail_realistic_multimodal_v1.json"
    model = ROOT / "kino_vla/eval/realistic_multimodal.py"
    feature = ROOT / "scripts/extract_kinofail_realistic_features.py"
    runner = ROOT / "scripts/run_kinofail_realistic_multimodal_v1.py"
    snapshot_records = ROOT / "outputs/eval/realistic_a0_a7_v4/snapshots/snapshot_records.jsonl"
    features = ROOT / "outputs/eval/realistic_a0_a7_v4/features/features.npz"
    predictions = ROOT / "outputs/eval/realistic_a0_a7_v4/predictions.jsonl"
    launcher = ROOT / "outputs/kinofail_realistic/corpus_scale_v7/launcher_audits/full_collection.json"
    launcher_value = json.loads(launcher.read_text(encoding="utf-8"))
    value = {
        "schema_version": "kinofail.realistic-multimodal-freeze.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": json.loads(config.read_text(encoding="utf-8"))["protocol_id"],
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha(path)
            for path in (config, model, feature, runner)
        },
        "runtime_attempts_existing_at_freeze": len(launcher_value.get("attempts", [])),
        "formal_model_artifacts_absent_at_freeze": {
            str(path.relative_to(ROOT)): not path.exists()
            for path in (snapshot_records, features, predictions)
        },
        "claim": "Splits, features, model roster, A3 cells, and statistics were fixed before formal model-input extraction, training, and heldout inference.",
        "supersedes": "outputs/eval/realistic_a0_a7_v4/multimodal_protocol_freeze.json",
        "amendment_reason": "Evidence-wiring-only amendment: bind training, inference, predictions, and experiment reports to the final A0 evidence bundle. Model inputs, splits, hyperparameters, cells, and statistics are unchanged.",
        "a8_in_scope": False,
    }
    output = ROOT / "outputs/eval/realistic_a0_a7_v4/multimodal_protocol_freeze_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runtime_attempts": value["runtime_attempts_existing_at_freeze"], "sha256": _sha(output)}, indent=2))


if __name__ == "__main__":
    main()
