#!/usr/bin/env python3
"""Publication analysis for the single deployable 11-class KiNO GMU.

All recognition, route-intervention, and action-selection quantities in this
program are computed from the same frozen ensemble.  Battery and conflict-cell
metadata are used only to define estimands after prediction; they are never
model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_kinofail_prediction_driven_action_v1 import (  # noqa: E402
    CLASS_TO_ACTION,
    OPERATOR_TO_ACTION,
    arm,
    fixed_action_loso,
)
from scripts.build_kinofail_action_consequence_evidence_v1 import (  # noqa: E402
    read_complete_cases,
)
from scripts.evaluate_kinofail_single_gmu_v1 import (  # noqa: E402
    CANONICAL_CLASSES,
    TrainingConfig,
    build_model,
    load_all191,
    normalize,
    predict,
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth).astype(str)
    prediction = np.asarray(prediction).astype(str)
    classes = sorted(set(truth.tolist()))
    return float(
        np.mean([np.mean(prediction[truth == label] == label) for label in classes])
    )


def load_predictor(
    freeze: Path, all191: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    bundle = torch.load(
        freeze / "single_gmu.pt", map_location="cpu", weights_only=False
    )
    config = TrainingConfig(**bundle["config"])
    source = load_all191(all191, config.visual_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities: dict[str, np.ndarray] = {}
    for battery in ("scale", "conflict"):
        visual = normalize(
            source[battery]["visual"],
            bundle["normalizer"]["visual_mean"],
            bundle["normalizer"]["visual_scale"],
        )
        proprio = normalize(
            source[battery]["proprio"],
            bundle["normalizer"]["proprio_mean"],
            bundle["normalizer"]["proprio_scale"],
        )
        seed_values: list[np.ndarray] = []
        for state in bundle["states"]:
            model = build_model(visual.shape[1], config)
            model.load_state_dict(state)
            model.to(device)
            seed_values.append(predict(model, visual, proprio, device))
        probabilities[battery] = np.mean(seed_values, axis=0)
    return bundle, source, probabilities


def physical_rows(
    battery: str, source: dict[str, Any], probabilities: np.ndarray
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(source["rows"]):
        unit = (
            str(row["physical_episode_id"])
            if battery == "scale"
            else f"{row['case_id']}::{row['attribution_category']}"
        )
        grouped[(unit, str(row["attribution_category"]))].append(index)
    result: list[dict[str, Any]] = []
    classes = np.asarray(CANONICAL_CLASSES)
    for (unit, truth), indices in sorted(grouped.items()):
        if len(indices) != 3:
            raise RuntimeError(f"expected three renders for {unit}")
        p = probabilities[indices].mean(axis=0)
        reference = source["rows"][indices[0]]
        result.append(
            {
                "battery": battery,
                "unit_id": unit,
                "truth": truth,
                "prediction": str(classes[int(p.argmax())]),
                "scene": str(reference["scene_cluster"]),
                "material": str(reference["cluster_material"]),
                "cell": str(reference.get("cell", "Scale")),
            }
        )
    return result


def crossed_product_bootstrap(
    by_battery: dict[str, list[dict[str, Any]]], draws: int, seed: int
) -> dict[str, Any]:
    all_rows = by_battery["scale"] + by_battery["conflict"]
    scenes = sorted({str(row["scene"]) for row in all_rows})
    materials = sorted({str(row["material"]) for row in all_rows})
    scene_index = {value: index for index, value in enumerate(scenes)}
    material_index = {value: index for index, value in enumerate(materials)}
    tensors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for battery, rows in by_battery.items():
        classes = sorted({str(row["truth"]) for row in rows})
        class_index = {value: index for index, value in enumerate(classes)}
        total = np.zeros((len(scenes), len(materials), len(classes)), dtype=np.int64)
        correct = np.zeros_like(total)
        for row in rows:
            key = (
                scene_index[str(row["scene"])],
                material_index[str(row["material"])],
                class_index[str(row["truth"])],
            )
            total[key] += 1
            correct[key] += int(row["prediction"] == row["truth"])
        tensors[battery] = total, correct

    rng = np.random.default_rng(seed)
    estimates = {"scale": [], "conflict": [], "macro": []}
    rejected = 0
    while len(estimates["macro"]) < draws:
        scene_weight = np.bincount(
            rng.integers(0, len(scenes), size=len(scenes)), minlength=len(scenes)
        )
        material_weight = np.bincount(
            rng.integers(0, len(materials), size=len(materials)),
            minlength=len(materials),
        )
        product = scene_weight[:, None] * material_weight[None, :]
        draw_values: dict[str, float] = {}
        valid = True
        for battery, (total, correct) in tensors.items():
            denominator = np.einsum("sm,smc->c", product, total)
            if bool(np.any(denominator == 0)):
                valid = False
                break
            numerator = np.einsum("sm,smc->c", product, correct)
            draw_values[battery] = float(np.mean(numerator / denominator))
        if not valid:
            rejected += 1
            continue
        estimates["scale"].append(draw_values["scale"])
        estimates["conflict"].append(draw_values["conflict"])
        estimates["macro"].append(
            0.5 * (draw_values["scale"] + draw_values["conflict"])
        )
    return {
        "draws": draws,
        "seed": seed,
        "weighting": "independent multinomial scene and material counts; product weight per scene-material cell",
        "empty_class_convention": "discard the complete draw if any registered class has zero weighted support",
        "discarded_draws": rejected,
        "ci95": {
            key: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
            for key, value in estimates.items()
        },
    }


def appearance_analysis(
    source: dict[str, dict[str, Any]], probabilities: dict[str, np.ndarray]
) -> dict[str, Any]:
    classes = np.asarray(CANONICAL_CLASSES)
    result: dict[str, Any] = {}
    for battery in ("scale", "conflict"):
        rows = source[battery]["rows"]
        p = probabilities[battery]
        view_prediction = classes[p.argmax(axis=1)]
        view_key = "appearance_intervention_id" if battery == "scale" else "appearance_view_id"
        per_view: dict[str, float] = {}
        for view in ("primary", "swap_01", "swap_02"):
            selected = np.asarray([str(row[view_key]) == view for row in rows])
            truth = np.asarray([str(row["attribution_category"]) for row in rows])[selected]
            per_view[view] = balanced_accuracy(truth, view_prediction[selected])
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            unit = (
                str(row["physical_episode_id"])
                if battery == "scale"
                else f"{row['case_id']}::{row['attribution_category']}"
            )
            grouped[unit].append(index)
        agreement = np.mean(
            [len(set(view_prediction[indices].tolist())) == 1 for indices in grouped.values()]
        )
        result[battery] = {
            "three_view_prediction_agreement": float(agreement),
            "per_view_balanced_accuracy": per_view,
            "primary_view_balanced_accuracy": per_view["primary"],
        }
    return result


def route_intervention(
    bundle: dict[str, Any], source: dict[str, Any], base: np.ndarray
) -> dict[str, Any]:
    rows = source["rows"]
    by_case_view: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_case_view[(str(row["case_id"]), str(row["appearance_view_id"]))].append(index)
    mate = np.empty(len(rows), dtype=np.int64)
    for key, indices in by_case_view.items():
        if len(indices) != 2:
            raise RuntimeError(f"pair does not contain two causes: {key}")
        mate[indices[0]], mate[indices[1]] = indices[1], indices[0]

    visual = normalize(
        source["visual"],
        bundle["normalizer"]["visual_mean"],
        bundle["normalizer"]["visual_scale"],
    )
    proprio = normalize(
        source["proprio"],
        bundle["normalizer"]["proprio_mean"],
        bundle["normalizer"]["proprio_scale"],
    )
    cells = np.asarray([str(row["cell"]) for row in rows])
    t2 = cells == "T2_vision_decisive"
    identifying_visual = visual.copy()
    identifying_proprio = proprio.copy()
    held_visual = visual.copy()
    held_proprio = proprio.copy()
    identifying_visual[t2] = visual[mate[t2]]
    identifying_proprio[~t2] = proprio[mate[~t2]]
    held_proprio[t2] = proprio[mate[t2]]
    held_visual[~t2] = visual[mate[~t2]]

    config = TrainingConfig(**bundle["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ensemble(v: np.ndarray, p: np.ndarray) -> np.ndarray:
        values = []
        for state in bundle["states"]:
            model = build_model(v.shape[1], config)
            model.load_state_dict(state)
            model.to(device)
            values.append(predict(model, v, p, device))
        return np.mean(values, axis=0)

    identifying = ensemble(identifying_visual, identifying_proprio)
    held = ensemble(held_visual, held_proprio)
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(str(row["case_id"]), str(row["attribution_category"]))].append(index)
    units = []
    for (case_id, truth), indices in sorted(grouped.items()):
        donor = {str(rows[int(mate[index])]["attribution_category"]) for index in indices}
        if len(donor) != 1:
            raise RuntimeError(f"donor mismatch: {case_id}/{truth}")
        units.append(
            {
                "truth": truth,
                "donor": next(iter(donor)),
                "cell": str(rows[indices[0]]["cell"]),
                "base": base[indices].mean(axis=0),
                "identifying": identifying[indices].mean(axis=0),
                "held": held[indices].mean(axis=0),
            }
        )
    class_values = np.asarray(CANONICAL_CLASSES)
    class_index = {value: index for index, value in enumerate(CANONICAL_CLASSES)}
    result: dict[str, Any] = {}
    for cell in ("T2_vision_decisive", "T3_proprio_decisive"):
        selected = [row for row in units if row["cell"] == cell]
        truth = np.asarray([row["truth"] for row in selected])
        donor = np.asarray([row["donor"] for row in selected])
        p0 = np.stack([row["base"] for row in selected])
        p1 = np.stack([row["identifying"] for row in selected])
        ph = np.stack([row["held"] for row in selected])
        pred0 = class_values[p0.argmax(axis=1)]
        pred1 = class_values[p1.argmax(axis=1)]
        donor_index = np.asarray([class_index[value] for value in donor])
        source_index = np.asarray([class_index[value] for value in truth])
        index = np.arange(len(selected))
        donor_gain = p1[index, donor_index] - p0[index, donor_index]
        source_drop = p0[index, source_index] - p1[index, source_index]
        result[cell] = {
            "source_units": len(selected),
            "base_balanced_accuracy": balanced_accuracy(truth, pred0),
            "identifying_route_prediction_change_rate": float(np.mean(pred1 != pred0)),
            "identifying_route_donor_following_balanced_accuracy": balanced_accuracy(donor, pred1),
            "identifying_route_donor_probability_gain_pp": float(100 * donor_gain.mean()),
            "identifying_route_source_probability_drop_pp": float(100 * source_drop.mean()),
            "identifying_route_directional_response_rate": float(
                np.mean((donor_gain > 0) & (source_drop > 0))
            ),
            "held_route_max_posterior_total_variation": float(
                np.max(0.5 * np.abs(ph - p0).sum(axis=1))
            ),
        }
    return result


def scene_cluster_ci(rows: list[dict[str, Any]], key: str, draws: int, seed: int) -> list[float]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(row)
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(scenes), size=len(scenes))
        cases = [row for index in sampled for row in by_scene[scenes[index]]]
        values[draw] = np.mean([float(row[key]) for row in cases])
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def action_analysis(
    prediction_path: Path,
    formal_analysis: Path,
    formal_corpus: Path,
    o6_analysis: Path,
    o6_corpus: Path,
    draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = {
        str(row["counterfactual_group_id"]): row for row in jsonl(prediction_path)
    }
    cases = read_complete_cases(formal_analysis, formal_corpus, "O6_push")
    cases += read_complete_cases(o6_analysis, o6_corpus)
    cases = sorted(cases, key=lambda row: str(row["case_id"]))
    if len(cases) != 277 or len(predictions) != 277:
        raise RuntimeError("action linkage requires 277 cases and predictions")
    fixed = fixed_action_loso(cases)
    rows: list[dict[str, Any]] = []
    for case in cases:
        group = str(case["case_id"]).rsplit("__", 1)[-1]
        prediction = predictions[group]
        selected_family = CLASS_TO_ACTION[str(prediction["prediction"])]
        oracle_family = OPERATOR_TO_ACTION[str(case["operator"])]
        selected = arm(case, selected_family)
        oracle = arm(case, oracle_family)
        fixed_arm = arm(case, fixed[str(case["scene_id"])])
        continuing = arm(case, "continue")
        halt = arm(case, "safe_halt")
        row = {
            "case_id": str(case["case_id"]),
            "scene_id": str(case["scene_id"]),
            "operator": str(case["operator"]),
            "truth": str(prediction["truth"]),
            "prediction": str(prediction["prediction"]),
            "exact_cause_correct": prediction["prediction"] == prediction["truth"],
            "action_equivalent_correct": selected_family == oracle_family,
            "selected_success": bool(selected["operator_recovery_success"]),
            "selected_fell": bool(selected["fell"]),
            "oracle_success": bool(oracle["operator_recovery_success"]),
            "oracle_fell": bool(oracle["fell"]),
            "fixed_success": bool(fixed_arm["operator_recovery_success"]),
            "fixed_fell": bool(fixed_arm["fell"]),
            "continue_success": bool(continuing["operator_recovery_success"]),
            "continue_fell": bool(continuing["fell"]),
            "halt_success": bool(halt["operator_recovery_success"]),
            "halt_fell": bool(halt["fell"]),
            "selected_cost": float(selected["terminal_cost"]),
            "oracle_cost": float(oracle["terminal_cost"]),
            "fixed_cost": float(fixed_arm["terminal_cost"]),
            "continue_cost": float(continuing["terminal_cost"]),
            "halt_cost": float(halt["terminal_cost"]),
        }
        for comparator in ("oracle", "fixed", "continue", "halt"):
            row[f"selected_minus_{comparator}_cost"] = row["selected_cost"] - row[f"{comparator}_cost"]
            row[f"success_minus_{comparator}"] = float(row["selected_success"]) - float(row[f"{comparator}_success"])
            row[f"fall_minus_{comparator}"] = float(row["selected_fell"]) - float(row[f"{comparator}_fell"])
        rows.append(row)

    mean = lambda key: float(np.mean([float(row[key]) for row in rows]))
    summary: dict[str, Any] = {
        "cases": len(rows),
        "exact_cause_accuracy": mean("exact_cause_correct"),
        "action_equivalent_accuracy": mean("action_equivalent_correct"),
        "selected_success_rate": mean("selected_success"),
        "selected_fall_rate": mean("selected_fell"),
        "selected_mean_terminal_cost": mean("selected_cost"),
        "comparators": {},
    }
    for comparator in ("oracle", "fixed", "continue", "halt"):
        summary["comparators"][comparator] = {
            "success_rate": mean(f"{comparator}_success"),
            "fall_rate": mean(f"{comparator}_fell"),
            "mean_terminal_cost": mean(f"{comparator}_cost"),
            "selected_minus_comparator_cost": mean(f"selected_minus_{comparator}_cost"),
            "selected_minus_comparator_cost_ci95": scene_cluster_ci(
                rows, f"selected_minus_{comparator}_cost", draws, 2026082450 + len(comparator)
            ),
            "selected_minus_comparator_success_pp": 100 * mean(f"success_minus_{comparator}"),
            "selected_minus_comparator_success_pp_ci95": [
                100 * value
                for value in scene_cluster_ci(
                    rows, f"success_minus_{comparator}", draws, 2026082460 + len(comparator)
                )
            ],
        }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--all191", type=Path, required=True)
    parser.add_argument("--action-predictions", type=Path, required=True)
    parser.add_argument("--formal-analysis", type=Path, required=True)
    parser.add_argument("--formal-corpus", type=Path, required=True)
    parser.add_argument("--o6-analysis", type=Path, required=True)
    parser.add_argument("--o6-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    bundle, source, probability = load_predictor(args.freeze.resolve(), args.all191.resolve())
    by_battery = {
        battery: physical_rows(battery, source[battery], probability[battery])
        for battery in ("scale", "conflict")
    }
    recognition = {
        battery: {
            "physical_units": len(rows),
            "balanced_accuracy": balanced_accuracy(
                np.asarray([row["truth"] for row in rows]),
                np.asarray([row["prediction"] for row in rows]),
            ),
        }
        for battery, rows in by_battery.items()
    }
    action, action_rows = action_analysis(
        args.action_predictions.resolve(),
        args.formal_analysis.resolve(),
        args.formal_corpus.resolve(),
        args.o6_analysis.resolve(),
        args.o6_corpus.resolve(),
        args.bootstrap_draws,
    )
    report = {
        "schema_version": "kinofail.single-gmu-publication-analysis.v1",
        "model_identity": "one frozen three-seed 11-class GMU ensemble for every reported KiNO inference",
        "battery_metadata_available_to_model": False,
        "battery_specific_heads": False,
        "config": bundle["config"],
        "recognition": recognition,
        "crossed_scene_material_bootstrap": crossed_product_bootstrap(
            by_battery, args.bootstrap_draws, 2026082441
        ),
        "appearance": appearance_analysis(source, probability),
        "route_intervention": route_intervention(
            bundle, source["conflict"], probability["conflict"]
        ),
        "action": action,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "action_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(action_rows[0]))
        writer.writeheader()
        writer.writerows(action_rows)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
