#!/usr/bin/env python3
"""Audit F25 one-to-one replacements and prove the final attrition rates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "outputs/kinofail_reconfirmation_v2"


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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    corpus_root = args.corpus_root.resolve()
    output_path = args.output.resolve()
    freeze = _json(freeze_path)

    errors: list[str] = []
    if freeze.get("status") != "sealed_before_collection":
        errors.append("freeze_status_invalid")
    if freeze.get("model_prediction_or_score_read") is not False:
        errors.append("freeze_was_not_model_blind")
    for section in ("schedules", "protocols", "execution_artifacts"):
        for artifact in freeze.get(section, []):
            path = ROOT / str(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                errors.append(f"frozen_artifact_mismatch:{path}")
    for key in ("design", "mapping", "scene_registry", "material_lock"):
        path = ROOT / str(freeze[key])
        if not path.is_file() or _sha256(path) != freeze[f"{key}_sha256"]:
            errors.append(f"frozen_artifact_mismatch:{path}")

    mapping_path = ROOT / str(freeze["mapping"])
    mapping = _jsonl(mapping_path)
    planned = int(freeze["scheduled_pairs"])
    if planned != 1_417 or len(mapping) != planned:
        errors.append(f"mapping_count_mismatch:{len(mapping)}")
    original_ids = [str(row["original_counterfactual_group_id"]) for row in mapping]
    replacement_ids = [
        str(row["replacement_counterfactual_group_id"]) for row in mapping
    ]
    if len(set(original_ids)) != len(original_ids):
        errors.append("duplicate_original_pair_mapping")
    if len(set(replacement_ids)) != len(replacement_ids):
        errors.append("duplicate_replacement_pair_mapping")
    if set(original_ids) & set(replacement_ids):
        errors.append("original_pair_id_reused")

    ledger_ids: set[str] = set()
    original_per_scene: Counter[str] = Counter()
    for ledger_path in sorted(
        (ORIGINAL / "attrition_ledgers_f13").glob("*/scale.json")
    ):
        ledger = _json(ledger_path)
        scene_id = str(ledger["scene_id"])
        for row in ledger["attrition"]:
            pair_id = str(row["counterfactual_group_id"])
            if pair_id in ledger_ids:
                errors.append(f"duplicate_original_ledger_id:{pair_id}")
            ledger_ids.add(pair_id)
            original_per_scene[scene_id] += 1
    if len(ledger_ids) != 1_417 or set(original_ids) != ledger_ids:
        errors.append("mapping_does_not_exactly_cover_original_scale_attrition")

    schedule_groups: dict[str, list[dict[str, Any]]] = {}
    for artifact in freeze["schedules"]:
        for row in _jsonl(ROOT / str(artifact["path"])):
            schedule_groups.setdefault(
                str(row["counterfactual_group_id"]), []
            ).append(row)
    if set(schedule_groups) != set(replacement_ids):
        errors.append("schedule_groups_do_not_match_mapping")
    for pair_id, rows in schedule_groups.items():
        if len(rows) != 2 or {str(row["condition"]) for row in rows} != {
            "nominal_counterfactual",
            "anomaly",
        }:
            errors.append(f"invalid_scheduled_pair:{pair_id}")
        seeds = {
            (
                int(row["operator_seed"]),
                int(row["physical_seed"]),
                int(row["physical_nuisance"]["physics_seed"]),
            )
            for row in rows
        }
        if len(seeds) != 1 or any(len(set(seed)) != 1 for seed in seeds):
            errors.append(f"pair_seed_contract_failed:{pair_id}")

    attempts_root = corpus_root / "attempts"
    attempts = {
        path.stem: _json(path) for path in sorted(attempts_root.glob("*.json"))
    }
    unexpected_attempts = set(attempts) - set(replacement_ids)
    if unexpected_attempts:
        errors.append(f"unexpected_attempt_ids:{len(unexpected_attempts)}")
    unattempted = set(replacement_ids) - set(attempts)
    started = {
        pair_id
        for pair_id, row in attempts.items()
        if row.get("state") == "started"
    }
    terminal = {
        pair_id
        for pair_id, row in attempts.items()
        if row.get("state") == "terminal"
    }
    if not args.allow_incomplete:
        if unattempted:
            errors.append(f"unattempted_pairs:{len(unattempted)}")
        if started:
            errors.append(f"nonterminal_pairs:{len(started)}")
        if terminal != set(replacement_ids):
            errors.append("terminal_attempts_do_not_cover_frozen_cohort")

    mapping_by_replacement = {
        str(row["replacement_counterfactual_group_id"]): row for row in mapping
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for pair_id in sorted(terminal):
        attempt = attempts[pair_id]
        mapping_row = mapping_by_replacement[pair_id]
        scene_id = str(mapping_row["scene_id"])
        summary_path = corpus_root / scene_id / "pair_summaries" / f"{pair_id}.json"
        reasons: list[str] = []
        summary: dict[str, Any] = {}
        if attempt.get("returncode") != 0:
            reasons.append(f"returncode:{attempt.get('returncode')}")
        if attempt.get("passed") is not True:
            reasons.append("attempt_not_passed")
        if not summary_path.is_file():
            reasons.append("summary_missing")
        else:
            summary = _json(summary_path)
            if summary.get("counterfactual_group_id") != pair_id:
                reasons.append("summary_pair_id_mismatch")
            if summary.get("passed") is not True:
                reasons.append("summary_not_passed")
            results = summary.get("results", [])
            if (
                not isinstance(results, list)
                or len(results) != 2
                or {row.get("condition") for row in results}
                != {"nominal_counterfactual", "anomaly"}
            ):
                reasons.append("summary_results_not_complete_pair")
            else:
                for result in results:
                    if result.get("passed") is not True or result.get("issues") != []:
                        reasons.append(
                            f"episode_validation_failed:{result.get('condition')}"
                        )
                    manifest = Path(str(result.get("manifest", "")))
                    if not manifest.is_file():
                        reasons.append(f"manifest_missing:{result.get('condition')}")
        record = {
            **mapping_row,
            "attempt": str((attempts_root / f"{pair_id}.json").relative_to(corpus_root)),
            "attempt_sha256": _sha256(attempts_root / f"{pair_id}.json"),
            "summary": (
                str(summary_path.relative_to(corpus_root))
                if summary_path.is_file()
                else None
            ),
            "summary_sha256": _sha256(summary_path) if summary_path.is_file() else None,
            "accepted": not reasons,
            "rejection_reasons": reasons,
        }
        (accepted if not reasons else rejected).append(record)

    successful_replacements = len(accepted)
    original_scale_planned = 10_560
    original_scale_attrited = 1_417
    original_t3_planned = 3_000
    original_t3_attrited = 114
    overall_planned = original_scale_planned + original_t3_planned
    scale_remaining = original_scale_attrited - successful_replacements
    overall_remaining = (
        original_scale_attrited
        + original_t3_attrited
        - successful_replacements
    )
    scale_rate = scale_remaining / original_scale_planned
    t3_rate = original_t3_attrited / original_t3_planned
    overall_rate = overall_remaining / overall_planned
    gates = {
        "all_frozen_replacements_terminal": terminal == set(replacement_ids),
        "at_least_890_successful_replacements": successful_replacements >= 890,
        "scale_attrition_strictly_below_5_percent": scale_rate < 0.05,
        "t3_attrition_strictly_below_5_percent": t3_rate < 0.05,
        "overall_attrition_strictly_below_5_percent": overall_rate < 0.05,
    }
    final_passed = not errors and all(gates.values())
    report = {
        "schema_version": "kinofail.f25-replenishment-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "final" if not args.allow_incomplete else "partial",
        "passed": final_passed,
        "model_prediction_feature_label_or_score_read": False,
        "freeze": str(freeze_path),
        "freeze_sha256": _sha256(freeze_path),
        "corpus_root": str(corpus_root),
        "counts": {
            "frozen_replacement_pairs": len(replacement_ids),
            "attempt_records": len(attempts),
            "unattempted_pairs": len(unattempted),
            "started_pairs": len(started),
            "terminal_pairs": len(terminal),
            "accepted_replacements": successful_replacements,
            "rejected_replacements": len(rejected),
        },
        "attrition": {
            "scale": {
                "planned": original_scale_planned,
                "original_attrited": original_scale_attrited,
                "replenished": successful_replacements,
                "remaining_attrited": scale_remaining,
                "rate": scale_rate,
            },
            "t3": {
                "planned": original_t3_planned,
                "original_attrited": original_t3_attrited,
                "replenished": 0,
                "remaining_attrited": original_t3_attrited,
                "rate": t3_rate,
            },
            "overall": {
                "planned": overall_planned,
                "original_attrited": original_scale_attrited
                + original_t3_attrited,
                "replenished": successful_replacements,
                "remaining_attrited": overall_remaining,
                "rate": overall_rate,
            },
        },
        "gates": gates,
        "errors": errors,
        "accepted_by_operator": dict(
            sorted(Counter(row["target_operator"] for row in accepted).items())
        ),
        "rejected_by_operator": dict(
            sorted(Counter(row["target_operator"] for row in rejected).items())
        ),
        "accepted_by_scene": dict(
            sorted(Counter(row["scene_id"] for row in accepted).items())
        ),
        "accepted": accepted,
        "rejected": rejected,
    }
    _atomic_json(output_path, report)
    print(json.dumps({key: report[key] for key in ("status", "passed", "counts", "attrition", "gates", "errors")}, indent=2, sort_keys=True))
    if args.allow_incomplete:
        return 0
    return 0 if final_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
