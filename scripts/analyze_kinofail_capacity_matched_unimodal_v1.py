#!/usr/bin/env python3
"""Crossed-cluster intervals for the capacity-matched unimodal controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/eval/kinofail_capacity_matched_unimodal_v1/physical_units.jsonl"
OUTPUT = ROOT / "outputs/eval/kinofail_capacity_matched_unimodal_v1/paired_cell_analysis.json"
METHODS = ("visual_mlp", "proprio_mlp", "concat_mlp", "gmu")
CELLS = ("T2_vision_decisive", "T3_proprio_decisive")
DRAWS = 20_000
SEED = 2026081419


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _balanced_accuracy(rows: list[dict], method: str) -> float:
    classes = sorted({str(row["truth"]) for row in rows})
    recalls = []
    for label in classes:
        selected = [row for row in rows if str(row["truth"]) == label]
        recalls.append(
            np.mean(
                [str(row["predictions"][method]) == label for row in selected]
            )
        )
    return float(np.mean(recalls))


def _cell_analysis(rows: list[dict], rng: np.random.Generator) -> dict:
    scenes = sorted({str(row["scene"]) for row in rows})
    materials = sorted({str(row["material"]) for row in rows})
    classes = sorted({str(row["truth"]) for row in rows})
    scene_index = {value: index for index, value in enumerate(scenes)}
    material_index = {value: index for index, value in enumerate(materials)}
    class_index = {value: index for index, value in enumerate(classes)}
    totals = np.zeros((len(scenes), len(materials), len(classes)), dtype=float)
    correct = np.zeros((len(METHODS), *totals.shape), dtype=float)
    for row in rows:
        index = (
            scene_index[str(row["scene"])],
            material_index[str(row["material"])],
            class_index[str(row["truth"])],
        )
        totals[index] += 1.0
        for method_index, method in enumerate(METHODS):
            correct[method_index][index] += float(
                str(row["predictions"][method]) == str(row["truth"])
            )
    values = np.empty((DRAWS, len(METHODS)), dtype=float)
    for draw in range(DRAWS):
        scene_count = np.bincount(
            rng.integers(0, len(scenes), size=len(scenes)), minlength=len(scenes)
        )
        material_count = np.bincount(
            rng.integers(0, len(materials), size=len(materials)),
            minlength=len(materials),
        )
        weights = scene_count[:, None] * material_count[None, :]
        class_total = np.einsum("sm,smc->c", weights, totals)
        class_correct = np.einsum("sm,ksmc->kc", weights, correct)
        values[draw] = np.nanmean(
            class_correct / np.where(class_total[None, :] > 0, class_total, np.nan),
            axis=1,
        )
    method_index = {method: index for index, method in enumerate(METHODS)}

    def interval(series: np.ndarray) -> list[float]:
        return [
            float(np.nanquantile(series, 0.025)),
            float(np.nanquantile(series, 0.975)),
        ]

    metrics = {
        method: {
            "balanced_accuracy": _balanced_accuracy(rows, method),
            "ci95": interval(values[:, method_index[method]]),
        }
        for method in METHODS
    }
    deltas = {}
    for multimodal in ("concat_mlp", "gmu"):
        for unimodal in ("visual_mlp", "proprio_mlp"):
            name = f"{multimodal}_minus_{unimodal}"
            series = (
                values[:, method_index[multimodal]]
                - values[:, method_index[unimodal]]
            )
            deltas[name] = {
                "estimate": metrics[multimodal]["balanced_accuracy"]
                - metrics[unimodal]["balanced_accuracy"],
                "ci95": interval(series),
            }
    return {
        "physical_units": len(rows),
        "scenes": len(scenes),
        "materials": len(materials),
        "metrics": metrics,
        "paired_deltas": deltas,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    all_rows = [row for row in _rows() if row["battery"] == "conflict"]
    rng = np.random.default_rng(SEED)
    report = {
        "schema_version": "kinofail.capacity-matched-unimodal-paired-cells.v1",
        "status": "complete",
        "draws": DRAWS,
        "clusters": ["scene", "displayed_material"],
        "cells": {
            cell: _cell_analysis(
                [row for row in all_rows if str(row["cell"]) == cell], rng
            )
            for cell in CELLS
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["cells"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
