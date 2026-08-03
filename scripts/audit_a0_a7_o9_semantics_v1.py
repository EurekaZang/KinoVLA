#!/usr/bin/env python3
"""Audit every O9 artifact contributing to the frozen A0--A7 evidence.

The legacy O9 collector admitted an episode after region exposure.  That is
not sufficient to establish high-centering, which requires sustained
base/belly loading on a transverse obstacle with partial foot unloading.  This
script separates:

* direct evidence: privileged operator telemetry is still available;
* proxy evidence: only the frozen proprioceptive snapshot survives; and
* experiment scope: A1 and A6 do not contain an O9 primary cell.

The audit is fail-closed.  Missing privileged telemetry never counts as proof.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kino_vla.data.o9_semantics import evaluate_o9_high_centering


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = ROOT / "outputs/eval/realistic_a0_a7_v6"
DEFAULT_OUTPUT = DEFAULT_EVAL_ROOT / "o9_semantic_audit_v1.json"
DEFAULT_RECOLLECT = DEFAULT_EVAL_ROOT / "o9_recollection_required_v1.jsonl"
DEFAULT_SNAPSHOT_RECORDS = DEFAULT_EVAL_ROOT / "snapshots/snapshot_records.jsonl"
DEFAULT_SNAPSHOT_ARRAYS = DEFAULT_EVAL_ROOT / "snapshots/snapshots.npz"
DEFAULT_SCALE_ROOT = (
    ROOT / "outputs/kinofail_realistic/corpus_scale_v8_replication_v1"
)
DEFAULT_SCALE_SCHEDULE = (
    ROOT
    / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
)
DEFAULT_A4_ROOT = ROOT / "outputs/kinofail_realistic/corpus_a4_actual_action_v5"
DEFAULT_A5_ROOT = ROOT / "outputs/kinofail_realistic/corpus_a5_realization_v1"
DEFAULT_C4_ROOT = ROOT / "outputs/kinofail_realistic/corpus_c4_direct_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def max_consecutive(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask.tolist():
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def compact_semantic_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(result["passed"]),
        "definition": result["definition"],
        "thresholds": result["thresholds"],
        "regions": [
            {
                "region_index": row["region_index"],
                "passed": row["passed"],
                "checks": row["checks"],
                "measurements": row["measurements"],
            }
            for row in result["regions"]
        ],
    }


def audit_frozen_snapshot(
    records_path: Path,
    arrays_path: Path,
    scale_root: Path,
    schedule_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_rows = list(read_jsonl(records_path))
    o9_rows = [row for row in all_rows if row.get("target_operator") == "O9_high_centering"]
    anomaly_primary = [
        row
        for row in o9_rows
        if row.get("condition") == "anomaly"
        and row.get("appearance_intervention_id") == "primary"
    ]
    pair_ids = sorted({row["counterfactual_group_id"] for row in o9_rows})

    schedule_rows = [
        row
        for row in read_jsonl(schedule_path)
        if row.get("target_operator") == "O9_high_centering"
    ]
    schedule_by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in schedule_rows:
        schedule_by_pair.setdefault(row["counterfactual_group_id"], []).append(row)

    summary_status = []
    for pair_id in pair_ids:
        summary_path = scale_root / "pair_summaries" / f"{pair_id}.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifests = [Path(row["manifest"]) for row in summary.get("results", [])]
        summary_status.append(
            {
                "counterfactual_group_id": pair_id,
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "summary_passed": bool(summary.get("passed")),
                "referenced_manifest_count": len(manifests),
                "referenced_manifests_present": sum(path.exists() for path in manifests),
            }
        )

    proxy_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(arrays_path) as archive:
        members = set(archive.namelist())
        for row in anomaly_primary:
            name = f"{row['sample_id']}__proprio.npy"
            if name not in members:
                raise RuntimeError(f"missing frozen proprio member: {name}")
            proprio = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
            feature_names = list(row["proprio_feature_names"])
            support = np.asarray(
                proprio[:, feature_names.index("support_ratio")], dtype=np.float64
            )
            proxy_rows.append(
                {
                    "counterfactual_group_id": row["counterfactual_group_id"],
                    "sample_id": row["sample_id"],
                    "scene_family": row["scene_family"],
                    "domain": row["domain"],
                    "split": next(
                        item["split"]
                        for item in schedule_by_pair[row["counterfactual_group_id"]]
                    ),
                    "severity_id": row["severity_id"],
                    "decision_time_s": row["decision_time_s"],
                    "support_ratio_proxy": {
                        "minimum": float(np.min(support)),
                        "frames_at_or_below_0_75": int(np.sum(support <= 0.75)),
                        "frames_at_or_below_0_50": int(np.sum(support <= 0.50)),
                        "max_consecutive_at_or_below_0_50": max_consecutive(
                            support <= 0.50
                        ),
                    },
                    "direct_privileged_o9_semantics_available": False,
                    "semantic_status": "not_proven_recollection_required",
                }
            )

    recollect = []
    for pair_id in pair_ids:
        schedule_pair = schedule_by_pair[pair_id]
        anomaly = next(row for row in schedule_pair if row["condition"] == "anomaly")
        recollect.append(
            {
                "source": "scale_v8_frozen_snapshot",
                "counterfactual_group_id": pair_id,
                "scene_family": anomaly["scene_family"],
                "domain": anomaly["domain"],
                "split": anomaly["split"],
                "severity_id": anomaly["severity_id"],
                "old_physical_realization": anomaly["physical_realization"],
                "old_physics_parameters": anomaly["physics_parameters"],
                "reason_codes": [
                    "raw_privileged_telemetry_pruned",
                    "old_admission_was_region_exposure_not_belly_contact",
                    "high_centering_not_directly_proven",
                ],
                "required_action": "recollect_nominal_anomaly_pair_with_o9_semantic_gate",
            }
        )

    return (
        {
            "source": "scale_v8_frozen_snapshot",
            "records_path": str(records_path),
            "records_sha256": sha256(records_path),
            "arrays_path": str(arrays_path),
            "arrays_size_bytes": arrays_path.stat().st_size,
            "schedule_path": str(schedule_path),
            "schedule_sha256": sha256(schedule_path),
            "o9_snapshot_records": len(o9_rows),
            "o9_physical_pairs": len(pair_ids),
            "o9_anomaly_primary_records": len(anomaly_primary),
            "appearance_views_per_physical_episode": sorted(
                Counter(
                    (
                        row["counterfactual_group_id"],
                        row["condition"],
                    )
                    for row in o9_rows
                ).values()
            )[0],
            "raw_manifest_availability": {
                "referenced": sum(
                    row["referenced_manifest_count"] for row in summary_status
                ),
                "present": sum(
                    row["referenced_manifests_present"] for row in summary_status
                ),
                "pair_summaries_present": len(summary_status),
            },
            "direct_semantic_passes": 0,
            "direct_semantic_status": "unavailable_fail_closed",
            "proxy_only": {
                "description": (
                    "Frozen 19-D proprioception can reveal support reduction but "
                    "cannot identify the contacting body link; it cannot establish "
                    "belly-on-ridge high-centering."
                ),
                "pairs_with_no_frame_support_le_0_50": sum(
                    row["support_ratio_proxy"]["frames_at_or_below_0_50"] == 0
                    for row in proxy_rows
                ),
                "pairs_with_fewer_than_5_frames_support_le_0_50": sum(
                    row["support_ratio_proxy"]["frames_at_or_below_0_50"] < 5
                    for row in proxy_rows
                ),
                "per_pair": proxy_rows,
            },
            "verdict": "all_o9_pairs_require_recollection",
        },
        recollect,
    )


def audit_result_rows(
    *,
    name: str,
    files: list[Path],
    operator_key: str,
    unique_key: str,
    action_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        for row in read_jsonl(path):
            if row.get(operator_key) == "O9_high_centering":
                selected.append((path, row))

    audited = []
    for path, row in selected:
        telemetry_rows = row.get("telemetry_rows", [])
        telemetry = (
            telemetry_rows[-1].get("operator_telemetry", {}) if telemetry_rows else {}
        )
        semantic = evaluate_o9_high_centering(telemetry)
        audited.append(
            {
                "file": str(path),
                "row_id": row.get(unique_key),
                "action_or_branch": row.get(action_key),
                "scene_cluster": row.get("scene_cluster"),
                "direct_semantic": compact_semantic_result(semantic),
            }
        )

    unique_units = sorted({str(row.get(unique_key)) for _, row in selected})
    invalid_units = sorted(
        {
            str(row["row_id"])
            for row in audited
            if not row["direct_semantic"]["passed"]
        }
    )
    fail_checks = Counter()
    for row in audited:
        for region in row["direct_semantic"]["regions"]:
            for check, passed in region["checks"].items():
                if not passed:
                    fail_checks[check] += 1

    recollect = [
        {
            "source": name,
            "unit_id": unit,
            "reason_codes": ["direct_o9_semantic_gate_failed"],
            "required_action": "exclude_and_recollect",
        }
        for unit in invalid_units
    ]
    return (
        {
            "source": name,
            "files": [str(path) for path in files],
            "rows": len(selected),
            "unique_units": len(unique_units),
            "direct_semantic_passes": sum(
                row["direct_semantic"]["passed"] for row in audited
            ),
            "failed_check_counts": dict(sorted(fail_checks.items())),
            "per_row": audited,
            "verdict": (
                "passed"
                if audited and all(row["direct_semantic"]["passed"] for row in audited)
                else "exclude_and_recollect_o9"
            ),
        },
        recollect,
    )


def audit_a5_realization(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifests = sorted(root.glob("**/O9_high_centering/*_anomaly/manifest.json"))
    audited = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        readback = manifest.get("operator_readback", {})
        semantic = evaluate_o9_high_centering(readback.get("telemetry", {}))
        audited.append(
            {
                "manifest": str(path),
                "manifest_sha256": sha256(path),
                "counterfactual_group_id": manifest["counterfactual_group_id"],
                "evaluation_eligible_under_legacy_gate": bool(
                    manifest.get("evaluation_eligible")
                ),
                "physical_realization": (
                    readback.get("telemetry", {}).get("regions", [{}])[0].get(
                        "geometry_kind"
                    )
                ),
                "direct_semantic": compact_semantic_result(semantic),
            }
        )
    fail_checks = Counter()
    for row in audited:
        for region in row["direct_semantic"]["regions"]:
            for check, passed in region["checks"].items():
                if not passed:
                    fail_checks[check] += 1
    recollect = [
        {
            "source": "a5_alternate_realization",
            "counterfactual_group_id": row["counterfactual_group_id"],
            "manifest": row["manifest"],
            "reason_codes": ["direct_o9_semantic_gate_failed"],
            "required_action": "exclude_and_recollect",
        }
        for row in audited
        if not row["direct_semantic"]["passed"]
    ]
    return (
        {
            "source": "a5_alternate_realization",
            "anomaly_manifests": len(audited),
            "legacy_evaluation_eligible": sum(
                row["evaluation_eligible_under_legacy_gate"] for row in audited
            ),
            "direct_semantic_passes": sum(
                row["direct_semantic"]["passed"] for row in audited
            ),
            "physical_realizations": dict(
                Counter(row["physical_realization"] for row in audited)
            ),
            "failed_check_counts": dict(sorted(fail_checks.items())),
            "per_manifest": audited,
            "verdict": "exclude_and_recollect_o9",
        },
        recollect,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recollection-list", type=Path, default=DEFAULT_RECOLLECT)
    args = parser.parse_args()

    frozen, frozen_recollect = audit_frozen_snapshot(
        DEFAULT_SNAPSHOT_RECORDS,
        DEFAULT_SNAPSHOT_ARRAYS,
        DEFAULT_SCALE_ROOT,
        DEFAULT_SCALE_SCHEDULE,
    )
    a4, a4_recollect = audit_result_rows(
        name="a4_actual_action",
        files=sorted(DEFAULT_A4_ROOT.glob("*/results.jsonl")),
        operator_key="operator",
        unique_key="case_id",
        action_key="action",
    )
    a5, a5_recollect = audit_a5_realization(DEFAULT_A5_ROOT)
    c4, c4_recollect = audit_result_rows(
        name="a5_c4_direct",
        files=sorted(DEFAULT_C4_ROOT.glob("*/results.jsonl")),
        operator_key="operator",
        unique_key="source_physical_episode_id",
        action_key="branch",
    )

    all_recollect = (
        frozen_recollect + a4_recollect + a5_recollect + c4_recollect
    )
    report = {
        "schema_version": "kinofail.a0-a7-o9-semantic-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "o9_semantics_failed_repair_required",
        "semantic_contract": {
            "module": str(ROOT / "kino_vla/data/o9_semantics.py"),
            "module_sha256": sha256(ROOT / "kino_vla/data/o9_semantics.py"),
            "definition": (
                "sustained Go2 base/belly load on a transverse obstacle with "
                "partial foot unloading; head- or limb-only impact is rejected"
            ),
        },
        "evidence": {
            "frozen_scale_snapshot": frozen,
            "a4_actual_action": a4,
            "a5_alternate_realization": a5,
            "a5_c4_direct": c4,
        },
        "a0_a7_impact": {
            "A0": {
                "status": "o9_semantic_coverage_withheld",
                "reason": (
                    "The 11-operator certificate includes O9, but its source "
                    "episodes do not retain direct belly-contact proof."
                ),
                "required_after_recollection": "rerun operator coverage certificate",
            },
            "A1": {
                "status": "unaffected",
                "reason": "Construct-validity test uses O2/O4, not O9.",
            },
            "A2": {
                "status": "o9_cell_and_11_class_aggregate_require_rerun",
                "reason": "Headline classifier consumes the frozen scale snapshot.",
            },
            "A3": {
                "status": "o9_cell_and_macro_aggregate_require_rerun",
                "reason": (
                    "Cross-battery inference reuses the scale snapshot; the "
                    "separate reverse-conflict probe itself targets O7, not O9."
                ),
            },
            "A4": {
                "status": "o9_cell_invalid_other_operator_cells_unchanged",
                "reason": (
                    "All recorded O9 actual-action rows fail direct belly-contact "
                    "semantics."
                ),
            },
            "A5": {
                "status": "o9_main_realization_and_c4_cells_require_rerun",
                "reason": (
                    "Both alternate-realization and direct C4 O9 rows fail direct "
                    "semantic admission."
                ),
            },
            "A6": {
                "status": "unaffected",
                "reason": "Primary boundary experiment contains O1/O2/O4/O5/O10 only.",
            },
            "A7": {
                "status": "11_class_aggregate_requires_rerun",
                "reason": "Ablations consume the frozen scale snapshot.",
            },
        },
        "recollection_units": {
            "total_records": len(all_recollect),
            "scale_pairs": len(frozen_recollect),
            "a4_rows_or_cases": len(a4_recollect),
            "a5_realization_pairs": len(a5_recollect),
            "c4_source_episodes": len(c4_recollect),
            "path": str(args.recollection_list),
        },
        "publication_verdict": (
            "Non-O9 evidence remains usable.  Any 11-operator or macro claim must "
            "exclude O9 until semantically valid O9 pairs are recollected and the "
            "dependent A0/A2/A3/A4/A5/A7 analyses are rerun."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.recollection_list.open("w", encoding="utf-8") as handle:
        for row in all_recollect:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "report_sha256": sha256(args.output),
                "recollection_list": str(args.recollection_list),
                "recollection_list_sha256": sha256(args.recollection_list),
                "status": report["status"],
                "counts": report["recollection_units"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
