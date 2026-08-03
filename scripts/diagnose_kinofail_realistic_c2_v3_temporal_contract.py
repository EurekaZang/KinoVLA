#!/usr/bin/env python3
"""Audit the C2 v3 T3 temporal-feature contract after the frozen failure.

This is development-only analysis.  It never rewrites the failed formal
artifact.  The audit replaces the variable-length whole-rollout summary with
a fixed 21-sample window aligned to a geometry-defined, outcome-independent
Go2 footprint/region encounter, then reports one physically specified
margin/delay contract.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval_kinofail_realistic_c2_bidirectional_v2 import (
    T2,
    T3,
    _aligned,
    _load_development,
    _load_test,
    _metric,
)
from eval_kinofail_realistic_c2_bidirectional_v3 import (
    OURS,
    _fit,
)
from kino_vla.eval.realistic_multimodal import proprio_summary


DIAGNOSTIC_METHODS = ("early_fusion_linear", "proprio_only", OURS)
V4 = "structured_class_proposal_router_v4"
PROPRIO_SIGNAL_INDICES = (
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
)


def _project_proprio(values: np.ndarray) -> np.ndarray:
    feature_count = 19
    blocks = values.shape[1] // feature_count
    if blocks * feature_count != values.shape[1]:
        raise RuntimeError("unexpected proprio summary dimension")
    indices = [
        block * feature_count + signal
        for block in range(blocks)
        for signal in PROPRIO_SIGNAL_INDICES
    ]
    return np.asarray(values[:, indices], dtype=np.float32)


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


def _distance_to_region(
    x: float,
    y: float,
    region: dict[str, Any],
) -> float:
    dx = max(abs(x - float(region["cx"])) - float(region["hx"]), 0.0)
    dy = max(abs(y - float(region["cy"])) - float(region["hy"]), 0.0)
    return float(np.hypot(dx, dy))


def _aligned_summary(
    episode_dir: Path,
    *,
    footprint_margin_m: float,
    post_encounter_delay_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = _json(episode_dir / "manifest.json")
    region = manifest["geometry_readback"]["operator_region"]
    telemetry_path = (
        episode_dir
        / manifest["artifacts"]["telemetry"]["path"]
    )
    telemetry = _jsonl(telemetry_path)
    encounters = [
        float(row["timestamp_s"])
        for row in telemetry
        if _distance_to_region(
            float(row["position_xy_m"][0]),
            float(row["position_xy_m"][1]),
            region,
        )
        <= footprint_margin_m
    ]
    if not encounters:
        raise RuntimeError(f"no geometry encounter: {episode_dir}")
    encounter_time_s = encounters[0]
    decision_time_s = encounter_time_s + post_encounter_delay_s
    artifact = manifest["artifacts"]["proprio"]
    with np.load(
        episode_dir / artifact["path"], allow_pickle=False
    ) as archive:
        features = np.asarray(
            archive[artifact["features_key"]], dtype=np.float32
        )
        timestamps = np.asarray(
            archive[artifact["timestamps_key"]], dtype=np.float64
        )
    candidates = np.flatnonzero(
        (timestamps > decision_time_s - 0.5 - 1.0e-9)
        & (timestamps <= decision_time_s + 0.021)
    )
    if len(candidates) < 21:
        raise RuntimeError(f"short aligned window: {episode_dir}")
    selected = candidates[-21:]
    end_skew_s = abs(float(timestamps[selected[-1]]) - decision_time_s)
    if end_skew_s > 0.021:
        raise RuntimeError(f"aligned-window end skew: {episode_dir}")
    window = features[selected]
    return proprio_summary(window), {
        "encounter_time_s": encounter_time_s,
        "decision_time_s": decision_time_s,
        "window_start_s": float(timestamps[selected[0]]),
        "window_end_s": float(timestamps[selected[-1]]),
        "end_skew_s": end_skew_s,
    }


def _evaluate(
    development: Path,
    test: Path,
    geometry: Path,
    corpus: Path,
    *,
    footprint_margin_m: float,
    post_encounter_delay_s: float,
) -> dict[str, Any]:
    (
        development_rows,
        development_visual,
        development_geometry,
        development_proprio,
        _,
    ) = _load_development(development)
    (
        test_rows,
        test_visual,
        test_geometry,
        test_proprio,
        _,
        _,
    ) = _load_test(test, geometry)
    development_labels = np.asarray(
        [str(row["attribution_category"]) for row in development_rows]
    )
    development_cells = np.asarray(
        [str(row["cell"]) for row in development_rows]
    )
    development_cases = np.asarray(
        [str(row["case_id"]) for row in development_rows]
    )
    test_labels = np.asarray(
        [str(row["attribution_category"]) for row in test_rows]
    )
    episode_dirs = {
        path.parent.name: path.parent
        for path in corpus.rglob("manifest.json")
    }
    cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    corrected = np.asarray(test_proprio, dtype=np.float32).copy()
    for index, row in enumerate(test_rows):
        if str(row["cell"]) != T3:
            continue
        episode_id = str(row["proprio_source_episode_id"])
        if episode_id not in cache:
            cache[episode_id] = _aligned_summary(
                episode_dirs[episode_id],
                footprint_margin_m=footprint_margin_m,
                post_encounter_delay_s=post_encounter_delay_s,
            )
        corrected[index] = cache[episode_id][0]
    development_proprio = _project_proprio(development_proprio)
    corrected = _project_proprio(corrected)

    results: dict[str, dict[str, Any]] = {}
    family_accuracy = None
    direct_t3_specialist_accuracy = None
    for method in DIAGNOSTIC_METHODS:
        model = _fit(
            method,
            development_visual,
            development_geometry,
            development_proprio,
            development_labels,
            development_cells,
            development_cases,
        )
        probability = model.predict_proba(
            test_visual, test_geometry, corrected
        )
        metric, _ = _metric(
            test_rows,
            test_labels,
            probability,
            model.classes,
        )
        results[method] = metric
        if method == OURS:
            family = model.family_prediction(
                test_visual, test_geometry, corrected
            )
            expected = np.asarray(
                [str(row["cell"]) for row in test_rows]
            )
            family_accuracy = float(np.mean(family == expected))
            t3 = expected == T3
            t3_probability = _aligned(
                model.t3_proprio,
                corrected[t3],
                ["invisible_obstacle", "low_friction"],
            )
            t3_prediction = np.asarray(
                ["invisible_obstacle", "low_friction"]
            )[t3_probability.argmax(axis=1)]
            direct_t3_specialist_accuracy = float(
                np.mean(t3_prediction == test_labels[t3])
            )
    best_baseline = max(
        [
            method
            for method in DIAGNOSTIC_METHODS
            if method != OURS
        ],
        key=lambda method: results[method]["balanced_accuracy"],
    )
    episode_audit = {
        episode_id: audit
        for episode_id, (_, audit) in sorted(cache.items())
    }
    expanded_predictions = {
        method: np.empty(len(test_rows), dtype=object)
        for method in (*DIAGNOSTIC_METHODS, V4)
    }
    test_scenes = np.asarray(
        [str(row["scene_cluster"]) for row in test_rows]
    )
    test_cells = np.asarray([str(row["cell"]) for row in test_rows])
    test_cases = np.asarray([str(row["case_id"]) for row in test_rows])
    for scene in sorted(set(test_scenes)):
        heldout = test_scenes == scene
        admitted = ~heldout
        train_visual = np.concatenate(
            [development_visual, test_visual[admitted]], axis=0
        )
        train_geometry = np.concatenate(
            [development_geometry, test_geometry[admitted]], axis=0
        )
        train_proprio = np.concatenate(
            [development_proprio, corrected[admitted]], axis=0
        )
        train_labels = np.concatenate(
            [development_labels, test_labels[admitted]], axis=0
        )
        train_cells = np.concatenate(
            [development_cells, test_cells[admitted]], axis=0
        )
        train_cases = np.concatenate(
            [development_cases, test_cases[admitted]], axis=0
        )
        heldout_models = {}
        for method in DIAGNOSTIC_METHODS:
            model = _fit(
                method,
                train_visual,
                train_geometry,
                train_proprio,
                train_labels,
                train_cells,
                train_cases,
            )
            heldout_models[method] = model
            probability = model.predict_proba(
                test_visual[heldout],
                test_geometry[heldout],
                corrected[heldout],
            )
            expanded_predictions[method][heldout] = np.asarray(
                model.classes
            )[probability.argmax(axis=1)]
        proprio_proposal = expanded_predictions["proprio_only"][
            heldout
        ]
        ours_model = heldout_models[OURS]
        t2_probability = _aligned(
            ours_model.t2_visual,
            np.concatenate(
                [
                    test_visual[heldout],
                    test_geometry[heldout],
                ],
                axis=1,
            ),
            ["adhesion", "compliant_terrain"],
        )
        t2_prediction = np.asarray(
            ["adhesion", "compliant_terrain"]
        )[t2_probability.argmax(axis=1)]
        expanded_predictions[V4][heldout] = np.where(
            np.isin(
                proprio_proposal,
                ["invisible_obstacle", "low_friction"],
            ),
            proprio_proposal,
            t2_prediction,
        )
    classes = sorted(set(test_labels.astype(str)))
    class_index = {label: index for index, label in enumerate(classes)}
    expanded_results = {
        method: _metric(
            test_rows,
            test_labels,
            np.eye(len(classes))[
                np.asarray(
                    [
                        class_index[str(value)]
                        for value in expanded_predictions[method]
                    ]
                )
            ],
            classes,
        )[0]
        for method in (*DIAGNOSTIC_METHODS, V4)
    }
    expanded_best_baseline = max(
        [
            method
            for method in DIAGNOSTIC_METHODS
            if method != OURS
        ],
        key=lambda method: expanded_results[method][
            "balanced_accuracy"
        ],
    )
    return {
        "footprint_margin_m": footprint_margin_m,
        "post_encounter_delay_s": post_encounter_delay_s,
        "proprio_projection": {
            "feature_names": [
                "imu_angular_velocity_xyz",
                "imu_linear_acceleration_xyz",
                "odom_velocity_xy",
                "slip_ratio",
                "base_height",
                "tilt",
                "effort_ratio",
                "support_ratio",
            ],
            "excluded": [
                "gravity_xyz",
                "absolute_odom_xy",
                "absolute_odom_heading",
            ],
            "summary_blocks": 10,
            "dimension": int(corrected.shape[1]),
        },
        "family_gate_accuracy": family_accuracy,
        "direct_t3_specialist_accuracy": direct_t3_specialist_accuracy,
        "results": results,
        "best_baseline": best_baseline,
        "ours_minus_best_baseline": (
            results[OURS]["balanced_accuracy"]
            - results[best_baseline]["balanced_accuracy"]
        ),
        "episode_audit": episode_audit,
        "expanded_v3_three_scene_loso": {
            "training": (
                "nine-scene prior development plus two of three v3 "
                "failure scenes; evaluate the held-out v3 scene"
            ),
            "results": expanded_results,
            "v4_rule": (
                "Use the four-class proprio proposal when it predicts a T3 "
                "class; otherwise invoke the frozen T2 visual specialist."
            ),
            "best_baseline": expanded_best_baseline,
            "ours_minus_best_baseline": (
                expanded_results[OURS]["balanced_accuracy"]
                - expanded_results[expanded_best_baseline][
                    "balanced_accuracy"
                ]
            ),
            "v4_minus_best_baseline": (
                expanded_results[V4]["balanced_accuracy"]
                - expanded_results[expanded_best_baseline][
                    "balanced_accuracy"
                ]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--test-geometry", type=Path, required=True)
    parser.add_argument("--t3-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    failures = []
    for margin in (0.40,):
        for delay in (0.2,):
            try:
                cells.append(
                    _evaluate(
                        args.development_features.resolve(),
                        args.test_features.resolve(),
                        args.test_geometry.resolve(),
                        args.t3_corpus.resolve(),
                        footprint_margin_m=margin,
                        post_encounter_delay_s=delay,
                    )
                )
            except RuntimeError as error:
                failures.append(
                    {
                        "footprint_margin_m": margin,
                        "post_encounter_delay_s": delay,
                        "error": str(error),
                    }
                )
    report = {
        "schema_version": "kinofail.realistic-c2-v3-temporal-contract-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_only_after_formal_v3_failure",
        "formal_v3_result_rewritten": False,
        "formal_v3_outcomes_used_for_development": True,
        "hypothesis": (
            "The frozen v3 builder summarized the entire variable-length T3 "
            "rollout, whereas every development T3 feature summarized a fixed "
            "21-sample event-aligned window."
        ),
        "alignment": {
            "event": (
                "first outcome-independent overlap between the measured Go2 "
                "base footprint margin and the frozen operator region"
            ),
            "proprio_samples": 21,
            "maximum_window_s": 0.5,
            "maximum_end_skew_s": 0.021,
            "footprint_margin_m": 0.40,
            "post_encounter_delay_s": 0.20,
            "selection_rationale": (
                "Go2 half-length plus a fixed 0.2 s post-encounter "
                "observation delay; no model score was used to choose a grid "
                "winner."
            ),
        },
        "cells": cells,
        "failures": failures,
        "claim_boundary": (
            "This audit diagnoses preprocessing and screens a correction for a "
            "future untouched confirmation.  It is not a corrected formal "
            "result and cannot support C2 by itself."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cells": len(cells),
                "failures": failures,
                "summary": [
                    {
                        "margin": cell["footprint_margin_m"],
                        "delay": cell["post_encounter_delay_s"],
                        "family": cell["family_gate_accuracy"],
                        "direct_t3": cell[
                            "direct_t3_specialist_accuracy"
                        ],
                        "ours": cell["results"][OURS][
                            "balanced_accuracy"
                        ],
                        "ours_t2": cell["results"][OURS][
                            "per_cell_accuracy"
                        ][T2],
                        "ours_t3": cell["results"][OURS][
                            "per_cell_accuracy"
                        ][T3],
                        "best_baseline": cell["best_baseline"],
                        "delta": cell["ours_minus_best_baseline"],
                        "expanded_loso_ours": cell[
                            "expanded_v3_three_scene_loso"
                        ]["results"][OURS]["balanced_accuracy"],
                        "expanded_loso_t3": cell[
                            "expanded_v3_three_scene_loso"
                        ]["results"][OURS]["per_cell_accuracy"][T3],
                        "expanded_loso_delta": cell[
                            "expanded_v3_three_scene_loso"
                        ]["ours_minus_best_baseline"],
                        "expanded_loso_v4": cell[
                            "expanded_v3_three_scene_loso"
                        ]["results"][V4]["balanced_accuracy"],
                        "expanded_loso_v4_t3": cell[
                            "expanded_v3_three_scene_loso"
                        ]["results"][V4]["per_cell_accuracy"][T3],
                        "expanded_loso_v4_delta": cell[
                            "expanded_v3_three_scene_loso"
                        ]["v4_minus_best_baseline"],
                    }
                    for cell in cells
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
