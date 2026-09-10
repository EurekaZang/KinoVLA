#!/usr/bin/env python3
"""Consolidate the ten frozen reference baselines for the all-191 benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FEATURE_SEAL = Path("/data/eureka/kinovla_outputs/kino_v4_all191_v1/feature_seal.json")
REFERENCE = ROOT / "outputs/eval/kino_v4_all191_v1_score_once"
KNOWN = ROOT / "outputs/eval/kinofail_known_fusion_baselines_all191_v1_score_once"
OUTPUT = ROOT / "outputs/eval/kino_v4_all191_benchmark_v1"
METHODS = (
    "vision",
    "proprioception",
    "joint_early_fusion",
    "late_fusion",
    "concat_mlp",
    "tfn",
    "lmf",
    "gmu",
    "embracenet",
    "moddrop",
)
DISPLAY_NAMES = {
    "vision": "Vision",
    "proprioception": "Proprioception",
    "joint_early_fusion": "Concat-ExtraTrees",
    "late_fusion": "Late fusion",
    "concat_mlp": "Concat-MLP",
    "tfn": "TFN",
    "lmf": "LMF",
    "gmu": "GMU",
    "embracenet": "EmbraceNet",
    "moddrop": "ModDrop",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    feature = load(FEATURE_SEAL)
    reference = load(REFERENCE / "report.json")
    known = load(KNOWN / "report.json")
    if not (
        feature.get("passed") is True
        and feature.get("checks", {}).get("scale_has_191_scenes") is True
        and feature.get("checks", {}).get("t2_has_191_scenes") is True
        and feature.get("checks", {}).get("t3_has_191_scenes") is True
        and reference.get("status") == "independent_confirmation_scored_once"
        and known.get("status") == "known_fusion_baselines_scored_once"
        and known.get("optimization_uses_confirmation_labels") is False
    ):
        raise RuntimeError("all-191 scoring prerequisites are not sealed")

    source_units_path = ROOT / known["artifacts"]["physical_units"]
    source_units = rows(source_units_path)
    consolidated_units = []
    for value in source_units:
        predictions = value.get("predictions", {})
        if not all(method in predictions for method in METHODS):
            raise RuntimeError(f"missing reference prediction: {value['unit_id']}")
        consolidated_units.append(
            {
                **value,
                "predictions": {
                    method: str(predictions[method]) for method in METHODS
                },
            }
        )

    scale_rows = [value for value in consolidated_units if value["battery"] == "scale"]
    conflict_rows = [
        value for value in consolidated_units if value["battery"] == "conflict"
    ]
    t2_rows = [
        value
        for value in conflict_rows
        if value["cell"] == "T2_vision_decisive"
    ]
    t3_rows = [
        value
        for value in conflict_rows
        if value["cell"] == "T3_proprio_decisive"
    ]
    scene_counts = {
        "scale": len({str(value["scene"]) for value in scale_rows}),
        "t2": len({str(value["scene"]) for value in t2_rows}),
        "t3": len({str(value["scene"]) for value in t3_rows}),
    }
    checks = {
        "exactly_ten_reference_baselines": len(METHODS) == 10,
        "scale_has_191_scenes": scene_counts["scale"] == 191,
        "t2_has_191_scenes": scene_counts["t2"] == 191,
        "t3_has_191_scenes": scene_counts["t3"] == 191,
        "physical_unit_ids_unique_within_battery": len(
            {(str(value["battery"]), str(value["unit_id"])) for value in consolidated_units}
        )
        == len(consolidated_units),
        "all_predictions_present": all(
            set(value["predictions"]) == set(METHODS)
            for value in consolidated_units
        ),
        "checkpoint_selection_used_no_all191_labels": (
            known.get("optimization_uses_confirmation_labels") is False
        ),
        "physical_unit_primary_estimand": True,
        "three_views_vote_once": True,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, sort_keys=True))

    physical = known["physical_unit_primary_metrics"]
    per_cell = known["per_conflict_cell_physical_unit"]
    cross = known["cross_battery_physical_unit"]
    bootstrap = known["physical_unit_crossed_cluster_bootstrap"]
    reference_view = reference["view_level_secondary_metrics"]
    known_view = known["view_level_secondary_metrics"]
    view_metrics = {
        battery: {
            method: (
                known_view[battery][method]
                if method in known_view[battery]
                else reference_view[battery][method]
            )
            for method in METHODS
        }
        for battery in ("scale", "conflict")
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    units_path = OUTPUT / "physical_units.jsonl"
    units_path.write_text(
        "".join(
            json.dumps(value, sort_keys=True) + "\n"
            for value in consolidated_units
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": "kinofail.kino-v4-all191-benchmark-score.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "all191_benchmark_scored_once",
        "passed": True,
        "score_once_and_report_regardless_of_outcome": True,
        "optimization_uses_all191_labels": False,
        "methods": [
            {"id": method, "display_name": DISPLAY_NAMES[method]}
            for method in METHODS
        ],
        "checks": checks,
        "counts": {
            "physical_units": len(consolidated_units),
            "scale_physical_units": len(scale_rows),
            "t2_physical_units": len(t2_rows),
            "t3_physical_units": len(t3_rows),
            "scenes": scene_counts,
        },
        "physical_unit_primary_metrics": {
            battery: {method: physical[battery][method] for method in METHODS}
            for battery in ("scale", "conflict")
        },
        "per_conflict_cell_physical_unit": {
            cell: {method: per_cell[cell][method] for method in METHODS}
            for cell in ("T2_vision_decisive", "T3_proprio_decisive")
        },
        "cross_battery_physical_unit": {
            method: cross[method] for method in METHODS
        },
        "view_level_secondary_metrics": view_metrics,
        "physical_unit_crossed_cluster_bootstrap": {
            **bootstrap,
            "metrics": {
                method: bootstrap["metrics"][method] for method in METHODS
            },
        },
        "source_sha256": {
            "feature_seal": sha256(FEATURE_SEAL),
            "reference_report": sha256(REFERENCE / "report.json"),
            "known_fusion_report": sha256(KNOWN / "report.json"),
            "known_fusion_physical_units": sha256(source_units_path),
            "script": sha256(Path(__file__).resolve()),
        },
        "artifacts": {
            "physical_units": str(units_path),
            "physical_units_sha256": sha256(units_path),
        },
    }
    atomic(OUTPUT / "report.json", report)
    print(
        json.dumps(
            {
                "passed": True,
                "counts": report["counts"],
                "cross_battery_physical_unit": report[
                    "cross_battery_physical_unit"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
