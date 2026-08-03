#!/usr/bin/env python3
"""Issue C2 v3 Amendment 1 for one truncated registry SHA-256."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OLD_REGISTRY_SHA = (
    "30c6ca297e790e1f17c461eede4daa2df6d7af31a9c5f20924a6e408a50e6411"
)
TRUNCATED_OFFICE_AUDIT_SHA = (
    "b255f07e35a368d1ea3fa6be071af4ba05f8296a57e8e0475568509b9c07"
)
CORRECT_OFFICE_AUDIT_SHA = (
    "b255f07e35a368d1ea3fa6be071ce79af4ba05f8296a57e8e0475568509b9c07"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    registry_path = (
        ROOT
        / "configs/data"
        / "kinofail_realistic_c2_v3_confirmation_scene_registry.json"
    )
    registry = _json(registry_path)
    offices = [
        row
        for row in registry["scenes"]
        if row["scene_id"] == "indoor_office_96"
    ]
    if len(offices) != 1:
        raise RuntimeError("C2 v3 Office registry row is not unique")
    office = offices[0]
    office_audit = ROOT / office["compiled_audit"]
    if (
        office["compiled_audit_sha256"]
        != CORRECT_OFFICE_AUDIT_SHA
        or _sha(office_audit) != CORRECT_OFFICE_AUDIT_SHA
        or len(TRUNCATED_OFFICE_AUDIT_SHA) != 60
        or len(CORRECT_OFFICE_AUDIT_SHA) != 64
    ):
        raise RuntimeError(
            "C2 v3 registry is not in the expected corrected state"
        )
    for scene in registry["scenes"]:
        if (
            _sha(ROOT / scene["episode_usd"])
            != scene["episode_sha256"]
            or _sha(ROOT / scene["compiled_audit"])
            != scene["compiled_audit_sha256"]
        ):
            raise RuntimeError(
                f"scene artifact mismatch after correction: {scene['scene_id']}"
            )

    invalid_corpus = (
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v3_t2"
    )
    invalid_manifests = list(invalid_corpus.glob("*/manifest.json"))
    invalid_scenes = {
        str(_json(path)["scene_cluster"])
        for path in invalid_manifests
    }
    if (
        len(invalid_manifests) != 10
        or invalid_scenes
        != {
            "forest_trail_whipple_metric_g04",
            "indoor_diningroom_191",
        }
        or (
            invalid_corpus
            / "scene_summaries/indoor_office_96.json"
        ).exists()
    ):
        raise RuntimeError(
            "unexpected pre-amendment outcome footprint"
        )
    t3_corpus = (
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v3_t3"
    )
    if t3_corpus.exists():
        raise RuntimeError(
            "T3 outcomes exist; registry-only amendment is no longer valid"
        )
    new_t2_corpus = (
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v3_t2_amendment1"
    )
    new_t3_corpus = (
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v3_t3_amendment1"
    )
    if new_t2_corpus.exists() or new_t3_corpus.exists():
        raise RuntimeError("amendment outcomes already exist")

    amendment_path = (
        ROOT
        / "outputs/kinofail_realistic"
        / "design_c2_bidirectional_confirmation_v3_amendment1.json"
    )
    amendment = {
        "schema_version": (
            "kinofail.realistic-c2-v3-administrative-amendment.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_any_t3_or_office_outcome",
        "reason": (
            "The Office compiled-audit SHA-256 was transcribed as a "
            "60-character string by omitting 'ce79'. Artifact bytes did "
            "not change and the hash gate blocked the first Office run."
        ),
        "change": {
            "json_pointer": (
                "/scenes/1/compiled_audit_sha256"
            ),
            "old_value": TRUNCATED_OFFICE_AUDIT_SHA,
            "old_length": len(TRUNCATED_OFFICE_AUDIT_SHA),
            "new_value": CORRECT_OFFICE_AUDIT_SHA,
            "new_length": len(CORRECT_OFFICE_AUDIT_SHA),
            "old_registry_sha256": OLD_REGISTRY_SHA,
            "new_registry_sha256": _sha(registry_path),
        },
        "unchanged": [
            "model architecture and hyperparameters",
            "development data and development report",
            "T2 and T3 schedules",
            "scene selection",
            "random seeds",
            "acceptance thresholds",
            "bootstrap seed and draws",
        ],
        "outcome_footprint_at_amendment": {
            "invalid_pre_amendment_t2_cases": len(
                invalid_manifests
            ),
            "invalid_pre_amendment_t2_scenes": sorted(
                invalid_scenes
            ),
            "office_cases": 0,
            "t3_physical_episodes": 0,
            "formal_feature_rows": 0,
            "formal_predictions": 0,
        },
        "disposition": {
            "pre_amendment_t2_corpus": (
                "retained as invalid administrative audit only"
            ),
            "publication_corpus": (
                "all 15 T2 cases and all T3 episodes must be collected "
                "under Amendment 1 protocols"
            ),
        },
        "passed": True,
    }
    _write(amendment_path, amendment)

    original_t2 = _json(
        ROOT
        / "configs/eval/kinofail_realistic_c2_v3_t2_collection.json"
    )
    original_t3 = _json(
        ROOT
        / "configs/eval/kinofail_realistic_c2_v3_t3_collection.json"
    )
    original_evaluation = _json(
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_bidirectional_formal_v3.json"
    )
    new_registry_sha = _sha(registry_path)
    t2_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_v3_t2_collection_amendment1.json"
    )
    t3_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_v3_t3_collection_amendment1.json"
    )
    evaluation_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_bidirectional_formal_v3_amendment1.json"
    )
    common_amendment = {
        "amendment_id": "C2-v3-A1-registry-hash-transcription",
        "amendment_sha256": _sha(amendment_path),
        "amendment_scope": "administrative_hash_correction_only",
    }
    amended_t2 = {
        **original_t2,
        "protocol_id": (
            "kinofail-realistic-c2-v3-t2-collection-amendment1"
        ),
        "status": "frozen_amendment1",
        "scene_registry_sha256": new_registry_sha,
        **common_amendment,
    }
    _write(t2_path, amended_t2)
    amended_t3 = {
        **original_t3,
        "protocol_id": (
            "kinofail-realistic-c2-v3-t3-collection-amendment1"
        ),
        "scene_registry_sha256": new_registry_sha,
        **common_amendment,
    }
    _write(t3_path, amended_t3)
    amended_evaluation = {
        **original_evaluation,
        "protocol_id": (
            "kinofail-realistic-c2-bidirectional-confirmation-v3-amendment1"
        ),
        "status": (
            "frozen_amendment1_before_any_t3_or_office_outcome"
        ),
        "scene_registry_sha256": new_registry_sha,
        "t2_collection_protocol_sha256": _sha(t2_path),
        "t3_collection_protocol_sha256": _sha(t3_path),
        **common_amendment,
    }
    _write(evaluation_path, amended_evaluation)
    print(
        json.dumps(
            {
                "passed": True,
                "amendment": str(amendment_path),
                "amendment_sha256": _sha(amendment_path),
                "t2_protocol": str(t2_path),
                "t2_sha256": _sha(t2_path),
                "t3_protocol": str(t3_path),
                "t3_sha256": _sha(t3_path),
                "evaluation_protocol": str(evaluation_path),
                "evaluation_sha256": _sha(evaluation_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
