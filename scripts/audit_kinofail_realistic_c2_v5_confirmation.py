#!/usr/bin/env python3
"""Audit the frozen C2 v5 confirmation corpora before feature extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.data.runtime_manifest import schedule_record_sha256
from kino_vla.eval.c2_temporal_v5 import (
    MAX_END_SKEW_S,
    geometry_aligned_invariant_summary,
)


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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2-schedule", type=Path, required=True)
    parser.add_argument("--t2-corpus", type=Path, required=True)
    parser.add_argument("--t3-schedule", type=Path, required=True)
    parser.add_argument("--t3-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    t2_schedule_path = args.t2_schedule.resolve()
    t2_corpus = args.t2_corpus.resolve()
    t3_schedule_path = args.t3_schedule.resolve()
    t3_corpus = args.t3_corpus.resolve()
    t2_schedule = _jsonl(t2_schedule_path)
    t3_schedule = _jsonl(t3_schedule_path)

    t2_expected = {str(row["case_id"]): row for row in t2_schedule}
    t2_manifests = {
        path.parent.name: _json(path)
        for path in t2_corpus.glob("*/manifest.json")
    }
    t2_checks = {
        "fifteen_scheduled_cases": len(t2_expected) == 15,
        "exact_case_set": set(t2_manifests) == set(t2_expected),
        "all_manifests_pass": all(
            row.get("passed") is True for row in t2_manifests.values()
        ),
        "six_samples_per_case": all(
            len(row.get("samples", [])) == 6
            for row in t2_manifests.values()
        ),
        "three_scenes_and_domains": (
            len({row["scene_cluster"] for row in t2_schedule}) == 3
            and len({row["domain"] for row in t2_schedule}) == 3
        ),
        "schedule_metadata_matches": all(
            manifest.get("case_id") == case_id
            and manifest.get("scene_cluster")
            == t2_expected[case_id]["scene_cluster"]
            and manifest.get("domain") == t2_expected[case_id]["domain"]
            for case_id, manifest in t2_manifests.items()
        ),
        "all_embedded_checks_pass": all(
            all(bool(value) for value in manifest.get("checks", {}).values())
            for manifest in t2_manifests.values()
        ),
    }

    expected_episodes = {
        str(row["episode_id"]): row for row in t3_schedule
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    temporal_alignment: dict[str, dict[str, Any]] = {}
    minimum_swap_l1 = float("inf")
    artifact_hashes_checked = 0
    t3_manifest_checks: list[bool] = []
    for episode_id, row in expected_episodes.items():
        group_id = str(row["counterfactual_group_id"])
        groups.setdefault(group_id, []).append(row)
        episode_dir = t3_corpus / Path(
            row["required_outputs"]["episode_manifest"]
        ).parent
        manifest_path = episode_dir / "manifest.json"
        if not manifest_path.is_file():
            t3_manifest_checks.append(False)
            continue
        manifest = _json(manifest_path)
        runtime = manifest.get("runtime_validation", {})
        measured = runtime.get("measured", {})
        threshold = float(
            runtime.get("thresholds", {}).get(
                "min_mean_appearance_pair_rgb_l1", float("inf")
            )
        )
        swaps = measured.get("mean_appearance_pair_rgb_l1", {})
        if swaps:
            minimum_swap_l1 = min(
                minimum_swap_l1,
                *(float(value) for value in swaps.values()),
            )
        artifacts = manifest.get("artifacts", {})
        artifact_entries = [
            artifacts.get("proprio", {}),
            artifacts.get("telemetry", {}),
            *[
                entry
                for entries in artifacts.get("rgb_views", {}).values()
                for entry in entries
            ],
        ]
        hashes_match = True
        for entry in artifact_entries:
            path = episode_dir / str(entry.get("path", ""))
            if not path.is_file() or _sha(path) != entry.get("sha256"):
                hashes_match = False
                break
            artifact_hashes_checked += 1
        t3_manifest_checks.append(
            manifest.get("episode_id") == episode_id
            and manifest.get("counterfactual_group_id") == group_id
            and manifest.get("schedule_record_sha256")
            == schedule_record_sha256(row)
            and manifest.get("artifact_state") == "validated"
            and manifest.get("evaluation_eligible") is True
            and manifest.get("operator_readback", {}).get("qa_passed")
            is True
            and runtime.get("passed") is True
            and len(swaps) == 2
            and all(float(value) >= threshold for value in swaps.values())
            and hashes_match
        )
        if row["condition"] == "anomaly":
            _, alignment = geometry_aligned_invariant_summary(episode_dir)
            temporal_alignment[episode_id] = alignment

    pair_summaries = {
        path.stem: _json(path)
        for path in (t3_corpus / "pair_summaries").glob("*.json")
    }
    t3_checks = {
        "sixty_scheduled_episodes": len(expected_episodes) == 60,
        "thirty_counterfactual_pairs": (
            len(groups) == 30
            and all(
                {row["condition"] for row in rows}
                == {"nominal_counterfactual", "anomaly"}
                for rows in groups.values()
            )
        ),
        "all_runtime_manifests_bound_and_pass": (
            len(t3_manifest_checks) == 60
            and all(t3_manifest_checks)
        ),
        "all_pair_summaries_pass": (
            set(pair_summaries) == set(groups)
            and all(
                row.get("passed") is True
                for row in pair_summaries.values()
            )
        ),
        "three_scenes_and_domains": (
            len({row["scene_family"] for row in t3_schedule}) == 3
            and len({row["domain"] for row in t3_schedule}) == 3
        ),
        "all_thirty_anomalies_temporally_aligned": (
            len(temporal_alignment) == 30
            and max(
                item["end_skew_s"]
                for item in temporal_alignment.values()
            )
            <= MAX_END_SKEW_S
        ),
    }
    checks = {
        **{f"t2_{key}": value for key, value in t2_checks.items()},
        **{f"t3_{key}": value for key, value in t3_checks.items()},
    }
    report = {
        "schema_version": "kinofail.realistic-c2-v5-confirmation-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "t2_cases": len(t2_manifests),
            "t2_samples": sum(
                len(row.get("samples", []))
                for row in t2_manifests.values()
            ),
            "t3_pairs": len(groups),
            "t3_episodes": len(t3_manifest_checks),
            "t3_anomaly_temporal_windows": len(temporal_alignment),
            "artifact_hashes_checked": artifact_hashes_checked,
            "scene_clusters": len(
                {row["scene_cluster"] for row in t2_schedule}
            ),
            "domains": len({row["domain"] for row in t2_schedule}),
        },
        "appearance_audit": {
            "minimum_measured_swap_rgb_l1": (
                minimum_swap_l1
                if minimum_swap_l1 != float("inf")
                else None
            ),
        },
        "temporal_audit": {
            "maximum_end_skew_s": max(
                (
                    item["end_skew_s"]
                    for item in temporal_alignment.values()
                ),
                default=None,
            ),
            "episode_alignment": temporal_alignment,
        },
        "source_sha256": {
            "t2_schedule": _sha(t2_schedule_path),
            "t3_schedule": _sha(t3_schedule_path),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "counts": report["counts"],
                "appearance_audit": report["appearance_audit"],
                "maximum_end_skew_s": report["temporal_audit"][
                    "maximum_end_skew_s"
                ],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
