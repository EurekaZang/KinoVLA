#!/usr/bin/env python3
"""Issue C2 v3 Amendment 2 for per-pair Isaac process isolation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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
    failed_corpus = (
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v3_t3_amendment1"
    )
    summaries = list(
        (failed_corpus / "pair_summaries").glob("*.json")
    )
    manifests = list(failed_corpus.rglob("manifest.json"))
    if (
        len(summaries) != 1
        or _json(summaries[0]).get("passed") is not True
        or len(manifests) != 3
    ):
        raise RuntimeError(
            "unexpected Amendment 1 T3 failure footprint"
        )
    if (
        ROOT / "outputs/eval/c2_bidirectional_v3/formal_features"
    ).exists() or (
        ROOT / "outputs/eval/c2_bidirectional_v3/formal"
    ).exists():
        raise RuntimeError(
            "formal C2 v3 features or predictions already exist"
        )
    publication_corpus = (
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v3_t3_amendment2"
    )
    if publication_corpus.exists():
        raise RuntimeError("Amendment 2 outcomes already exist")

    isolated_runner = (
        ROOT
        / "scripts"
        / "run_kinofail_realistic_c2_v3_t3_isolated.py"
    )
    pair_collector = (
        ROOT
        / "scripts/isaac_collect_kinofail_realistic_pair_v2.py"
    )
    amendment_path = (
        ROOT
        / "outputs/kinofail_realistic"
        / "design_c2_bidirectional_confirmation_v3_amendment2.json"
    )
    amendment = {
        "schema_version": (
            "kinofail.realistic-c2-v3-administrative-amendment.v2"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "frozen_after_one_complete_t3_pair_before_formal_features"
        ),
        "reason": (
            "Reusing one Isaac application across multiple RTX material "
            "pairs triggered a texture-file/mutex lifecycle assertion. "
            "The pre-existing frozen pair collector avoids this by "
            "starting one isolated Isaac process per pair."
        ),
        "change": {
            "old_collector": (
                "scripts/isaac_collect_kinofail_realistic_c2_v3_t3_scene.py"
            ),
            "new_collector": str(pair_collector.relative_to(ROOT)),
            "new_collector_sha256": _sha(pair_collector),
            "new_runner": str(isolated_runner.relative_to(ROOT)),
            "new_runner_sha256": _sha(isolated_runner),
            "process_isolation": "one Isaac application per pair",
        },
        "unchanged": [
            "model architecture and hyperparameters",
            "development data and report",
            "T2 and T3 schedules",
            "scene selection and registry",
            "operator parameters",
            "random seeds",
            "acceptance thresholds",
            "bootstrap seed and draws",
            "runtime episode validation",
        ],
        "outcome_footprint_at_amendment": {
            "complete_t3_pairs": len(summaries),
            "validated_t3_episodes": len(manifests),
            "formal_feature_rows": 0,
            "formal_predictions": 0,
        },
        "disposition": {
            "amendment1_t3_corpus": (
                "retained as invalid process-lifecycle audit only"
            ),
            "publication_t3_corpus": (
                "all 30 pairs must be collected from scratch under "
                "Amendment 2"
            ),
        },
        "passed": True,
    }
    _write(amendment_path, amendment)

    t3_a1_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_v3_t3_collection_amendment1.json"
    )
    evaluation_a1_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_bidirectional_formal_v3_amendment1.json"
    )
    t2_a1_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_v3_t2_collection_amendment1.json"
    )
    t3_a1 = _json(t3_a1_path)
    evaluation_a1 = _json(evaluation_a1_path)
    t3_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_v3_t3_collection_amendment2.json"
    )
    evaluation_path = (
        ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_bidirectional_formal_v3_amendment2.json"
    )
    common = {
        "amendment_id": "C2-v3-A2-per-pair-process-isolation",
        "amendment_sha256": _sha(amendment_path),
        "amendment_scope": "collection_process_isolation_only",
    }
    t3_a2 = {
        **t3_a1,
        "protocol_id": (
            "kinofail-realistic-c2-v3-t3-collection-amendment2"
        ),
        "collector_path": str(pair_collector.relative_to(ROOT)),
        "collector_sha256": _sha(pair_collector),
        "runner_path": str(isolated_runner.relative_to(ROOT)),
        "runner_sha256": _sha(isolated_runner),
        "collection_contract": {
            **t3_a1["collection_contract"],
            "isaac_processes": 30,
            "pairs_per_process": 1,
        },
        **common,
    }
    _write(t3_path, t3_a2)
    evaluation_a2 = {
        **evaluation_a1,
        "protocol_id": (
            "kinofail-realistic-c2-bidirectional-confirmation-v3-amendment2"
        ),
        "status": (
            "frozen_amendment2_before_formal_features"
        ),
        "t2_collection_protocol_sha256": _sha(t2_a1_path),
        "t3_collection_protocol_sha256": _sha(t3_path),
        **common,
    }
    _write(evaluation_path, evaluation_a2)
    print(
        json.dumps(
            {
                "passed": True,
                "amendment": str(amendment_path),
                "amendment_sha256": _sha(amendment_path),
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
