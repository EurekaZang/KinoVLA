#!/usr/bin/env python3
"""Train and evaluate the frozen realistic Kino-Fail A2/A3/A5 model roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_multimodal import (  # noqa: E402
    ObservableClassifier,
    StructuredBidirectionalClassifier,
    predictions_from_probabilities,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split_labels(records: list[dict], registry: dict) -> tuple[np.ndarray, np.ndarray]:
    scene_split = {str(row["scene_id"]): str(row["split"]) for row in registry["scenes"]}
    scenes = np.asarray([scene_split[str(row["scene_family"])] for row in records])
    materials = np.asarray([str(row["material_family"]).split("_", 1)[0] for row in records])
    if not set(scenes) <= {"train", "val", "test"}:
        raise ValueError("invalid scene split label")
    if not set(materials) <= {"train", "val", "test"}:
        raise ValueError("invalid material split label")
    return scenes, materials


def _masks(axis: str, records: list[dict], scene: np.ndarray, material: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primary = np.asarray([row["appearance_intervention_id"] == "primary" for row in records])
    camera = np.asarray([str(row["camera_profile"]) for row in records])
    if axis == "scene":
        train = scene != "test"
        main_test = (scene == "test") & primary
        inference = scene == "test"
    elif axis == "material":
        train = (material != "test") & primary
        main_test = (material == "test") & primary
        inference = main_test.copy()
    elif axis == "scene_and_material":
        train = (scene != "test") & (material != "test") & primary
        main_test = (scene == "test") & (material == "test") & primary
        inference = main_test.copy()
    elif axis == "camera_profile":
        train = (camera != "go2_front_calib_c") & primary
        main_test = (camera == "go2_front_calib_c") & primary
        inference = main_test.copy()
    else:
        raise ValueError(axis)
    return train, main_test, inference


def _macro_recall(truth: list[str], predicted: list[str]) -> float:
    return float(balanced_accuracy_score(truth, predicted))


def _cluster_delta(
    ours: list[dict], baseline: list[dict], *, reps: int, seed: int
) -> dict[str, Any]:
    base = {row["sample_id"]: row for row in baseline if row["appearance_intervention_id"] == "primary"}
    paired: dict[str, list[float]] = defaultdict(list)
    for row in ours:
        if row["appearance_intervention_id"] != "primary":
            continue
        other = base[row["sample_id"]]
        paired[row["counterfactual_group_id"]].append(
            float(row["prediction"] == row["truth"])
            - float(other["prediction"] == other["truth"])
        )
    cluster_values = np.asarray([np.mean(values) for values in paired.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [np.mean(rng.choice(cluster_values, len(cluster_values), replace=True)) for _ in range(reps)]
    )
    return {
        "unit": "paired counterfactual physics group",
        "clusters": len(cluster_values),
        "point": float(cluster_values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "repetitions": reps,
        "seed": seed,
    }


def _jsd(left: np.ndarray, right: np.ndarray) -> float:
    eps = 1.0e-12
    left = np.maximum(np.asarray(left, dtype=np.float64), eps)
    right = np.maximum(np.asarray(right, dtype=np.float64), eps)
    left /= left.sum()
    right /= right.sum()
    middle = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(left / middle)) + 0.5 * np.sum(right * np.log(right / middle)))


def _a3_cell(row: dict, cells: dict) -> str:
    if row["condition"] == "nominal_counterfactual":
        return "T5"
    operator = row["target_operator"]
    for cell in ("T1", "T2", "T3", "T4"):
        if operator in cells[cell]:
            return cell
    raise ValueError(f"operator not assigned to A3 battery: {operator}")


def _rate(rows: list[dict]) -> float | None:
    return float(np.mean([row["prediction"] == row["truth"] for row in rows])) if rows else None


def _fit_method(
    name: str,
    visual: np.ndarray,
    proprio: np.ndarray,
    labels: np.ndarray,
    episodes: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> Any:
    if name == "structured_bidirectional":
        return StructuredBidirectionalClassifier.fit(
            visual, proprio, labels, episodes, groups, seed=seed
        )
    key = {"vision_only": "vision", "proprio_only": "proprio", "early_fusion": "early_fusion"}[name]
    return ObservableClassifier.fit(
        visual, proprio, labels, episodes, feature_key=key, seed=seed
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval/kinofail_realistic_multimodal_v1.json")
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "outputs/eval/realistic_a0_a7_v4/snapshots")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "outputs/eval/realistic_a0_a7_v4/features")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    snapshot_dir, feature_dir = args.snapshot_dir.resolve(), args.feature_dir.resolve()
    records_path = snapshot_dir / "snapshot_records.jsonl"
    features_path = feature_dir / "features.npz"
    feature_manifest_path = feature_dir / "feature_manifest.json"
    feature_manifest = _json(feature_manifest_path)
    if feature_manifest.get("status") != "complete" or feature_manifest["output_sha256"]["features"] != _sha256(features_path):
        raise RuntimeError("feature cache is incomplete or hash-mismatched")
    records = _jsonl(records_path)
    archive = np.load(features_path, allow_pickle=False)
    sample_ids = archive["sample_ids"].astype(str)
    if sample_ids.tolist() != [str(row["sample_id"]) for row in records]:
        raise RuntimeError("feature rows and snapshot metadata are not aligned")
    visual, proprio = archive["visual"], archive["proprio"]
    labels = np.asarray([str(row["attribution_category"]) for row in records])
    episodes = np.asarray([str(row["physical_episode_id"]) for row in records])
    groups = np.asarray([str(row["counterfactual_group_id"]) for row in records])
    registry_path = ROOT / config["scene_registry"]
    registry = _json(registry_path)
    a0_certificate_path = ROOT / config["output_root"] / "a0_certificate.json"
    a0_certificate = _json(a0_certificate_path)
    if a0_certificate.get("status") != "publication_ready":
        raise RuntimeError("formal model training requires the final publication-ready A0 certificate")
    a0_bundle_sha256 = str(a0_certificate["a0_evidence_bundle_sha256"])
    scene_split, material_split = _split_labels(records, registry)
    output = ROOT / config["output_root"]
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    predictions: list[dict] = []
    checkpoint_hashes = {}
    axes = list(config["heldout_axes"])
    methods = list(config["methods"])
    for axis in axes:
        train, main_test, inference = _masks(axis, records, scene_split, material_split)
        if not train.any() or not main_test.any():
            raise RuntimeError(f"empty frozen split: {axis}")
        if set(labels[main_test]) - set(labels[train]):
            raise RuntimeError(f"{axis} test contains unseen attribution classes")
        for seed in [int(value) for value in config["training_seeds"]]:
            for method_name in methods:
                model = _fit_method(
                    method_name,
                    visual[train], proprio[train], labels[train], episodes[train], groups[train],
                    seed=seed,
                )
                probability = model.predict_proba(visual[inference], proprio[inference])
                prediction = predictions_from_probabilities(probability, model.classes)
                indices = np.flatnonzero(inference)
                for local, index in enumerate(indices):
                    row = records[int(index)]
                    predictions.append(
                        {
                            "sample_id": row["sample_id"],
                            "physical_episode_id": row["physical_episode_id"],
                            "counterfactual_group_id": row["counterfactual_group_id"],
                            "appearance_intervention_id": row["appearance_intervention_id"],
                            "condition": row["condition"],
                            "target_operator": row["target_operator"],
                            "domain": row["domain"],
                            "scene_family": row["scene_family"],
                            "camera_profile": row["camera_profile"],
                            "material_family": row["material_family"],
                            "severity_id": row["severity_id"],
                            "axis": axis,
                            "seed": seed,
                            "method": method_name,
                            "truth": row["attribution_category"],
                            "prediction": str(prediction[local]),
                            "classes": list(model.classes),
                            "probabilities": probability[local].tolist(),
                            "headline_primary_view": bool(main_test[index]),
                            "appearance_views_are_independent_samples": False,
                        }
                    )
                checkpoint_path = checkpoint_root / axis / f"seed{seed}" / f"{method_name}.pkl"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("wb") as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                checkpoint_hashes[str(checkpoint_path.relative_to(ROOT))] = _sha256(checkpoint_path)

    predictions_path = output / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8"
    )
    reps, bootstrap_seed = int(config["bootstrap"]["repetitions"]), int(config["bootstrap"]["seed"])
    primary_rows = [row for row in predictions if row["headline_primary_view"]]

    a2_axes = {}
    for axis in ("scene", "material", "scene_and_material"):
        axis_rows = [row for row in primary_rows if row["axis"] == axis]
        method_seed = {}
        for method in methods:
            method_seed[method] = {
                str(seed): _macro_recall(
                    [row["truth"] for row in axis_rows if row["method"] == method and row["seed"] == seed],
                    [row["prediction"] for row in axis_rows if row["method"] == method and row["seed"] == seed],
                )
                for seed in config["training_seeds"]
            }
        deltas = []
        for seed in config["training_seeds"]:
            ours = [row for row in axis_rows if row["method"] == "structured_bidirectional" and row["seed"] == seed]
            base = [row for row in axis_rows if row["method"] == "early_fusion" and row["seed"] == seed]
            deltas.append(_cluster_delta(ours, base, reps=reps, seed=bootstrap_seed + int(seed)))
        a2_axes[axis] = {"balanced_accuracy_by_method_seed": method_seed, "ours_minus_early_fusion": deltas}
    combined_ours = list(a2_axes["scene_and_material"]["balanced_accuracy_by_method_seed"]["structured_bidirectional"].values())
    combined_delta_lowers = [row["ci95"][0] for row in a2_axes["scene_and_material"]["ours_minus_early_fusion"]]
    a2 = {
        "schema_version": "kinofail.realistic-a2.v1",
        "protocol_id": config["protocol_id"],
        "heldout_axes": a2_axes,
        "acceptance": {
            "ours_combined_balanced_accuracy_ge_0_8_every_seed": min(combined_ours) >= 0.8,
            "ours_minus_unshaped_ci_lower_gt_0_every_seed": min(combined_delta_lowers) > 0.0,
        },
    }
    a2["passed"] = all(a2["acceptance"].values())

    scene_rows = [row for row in primary_rows if row["axis"] == "scene"]
    a3_by_method_seed = {}
    for method in methods:
        a3_by_method_seed[method] = {}
        for seed in config["training_seeds"]:
            subset = [dict(row) for row in scene_rows if row["method"] == method and row["seed"] == seed]
            for row in subset:
                row["cell"] = _a3_cell(row, config["a3_cells"])
            per_cell = {cell: _rate([row for row in subset if row["cell"] == cell]) for cell in ("T1", "T2", "T3", "T4", "T5")}
            per_domain = {
                domain: {cell: _rate([row for row in subset if row["domain"] == domain and row["cell"] == cell]) for cell in ("T1", "T2", "T3", "T4", "T5")}
                for domain in sorted({row["domain"] for row in subset})
            }
            t3_subcells = {
                "looks_safe": _rate([row for row in subset if row["cell"] == "T3" and row["target_operator"] == "O7_visual_remap"]),
                "transparent_or_thin_obstacle": _rate([row for row in subset if row["cell"] == "T3" and row["target_operator"] == "O8_invisible_collider"]),
                "reverse_conflict": None,
            }
            a3_by_method_seed[method][str(seed)] = {
                "n": len(subset), "per_cell": per_cell, "per_domain": per_domain,
                "t3_subcells": t3_subcells,
                "macro": float(np.mean(list(per_cell.values()))),
                "worst_cell": float(min(per_cell.values())),
            }
    ours_seed = a3_by_method_seed["structured_bidirectional"]
    strict_groups = defaultdict(list)
    for row in scene_rows:
        if row["method"] == "structured_bidirectional":
            strict_groups[row["counterfactual_group_id"]].append(row["prediction"] == row["truth"])
    strict_success = sum(all(values) for values in strict_groups.values())
    strict_n = len(strict_groups)
    exact_ci = [
        0.0 if strict_success == 0 else float(beta.ppf(0.025, strict_success, strict_n - strict_success + 1)),
        1.0 if strict_success == strict_n else float(beta.ppf(0.975, strict_success + 1, strict_n - strict_success)),
    ]
    a3_checks = {
        "every_seed_T1_ge_0_9": all(row["per_cell"]["T1"] >= 0.9 for row in ours_seed.values()),
        "every_seed_T2_eq_1": all(row["per_cell"]["T2"] == 1.0 for row in ours_seed.values()),
        "every_seed_T3_gt_0_917": all(row["per_cell"]["T3"] > 0.917 for row in ours_seed.values()),
        "every_seed_T4_eq_1": all(row["per_cell"]["T4"] == 1.0 for row in ours_seed.values()),
        "every_seed_T5_ge_0_9": all(row["per_cell"]["T5"] >= 0.9 for row in ours_seed.values()),
        "every_seed_worst_ge_0_9": all(row["worst_cell"] >= 0.9 for row in ours_seed.values()),
        "every_seed_macro_ge_0_95": all(row["macro"] >= 0.95 for row in ours_seed.values()),
        "reverse_conflict_realization_present": False,
        "all_three_domains_present": {row["domain"] for row in scene_rows} == {"life", "production", "wild"},
        "minimum_33_physics_clusters": strict_n >= 33,
    }
    a3 = {
        "schema_version": "kinofail.realistic-a3.v1",
        "protocol_id": config["protocol_id"],
        "results": a3_by_method_seed,
        "strict_primary_view_all_seeds_counterfactual_pair_certificate": {"success": strict_success, "total": strict_n, "exact_ci95": exact_ci},
        "acceptance": a3_checks,
        "passed": all(a3_checks.values()),
        "missing_design_guard": config["a3_missing_design_guard"],
    }

    texture_rows = [row for row in predictions if row["axis"] == "scene"]
    texture = {}
    for method in methods:
        values = []
        for seed in config["training_seeds"]:
            grouped = defaultdict(list)
            for row in texture_rows:
                if row["method"] == method and row["seed"] == seed:
                    grouped[row["physical_episode_id"]].append(row)
            for episode, rows in grouped.items():
                if len(rows) != 3:
                    raise RuntimeError(f"texture group {episode} does not have three views")
                agreement = len({row["prediction"] for row in rows}) == 1
                probabilities = [np.asarray(row["probabilities"], dtype=np.float64) for row in rows]
                jsds = [_jsd(probabilities[i], probabilities[j]) for i in range(3) for j in range(i + 1, 3)]
                values.append({"seed": seed, "physical_episode_id": episode, "agreement": agreement, "mean_pairwise_jsd": float(np.mean(jsds))})
        texture[method] = {
            "physical_episode_seed_units": len(values),
            "hard_consistency": float(np.mean([row["agreement"] for row in values])),
            "mean_probability_jsd": float(np.mean([row["mean_pairwise_jsd"] for row in values])),
        }
    a5_axes = {}
    for axis in axes:
        rows = [row for row in primary_rows if row["axis"] == axis]
        a5_axes[axis] = {
            method: {
                str(seed): _macro_recall(
                    [row["truth"] for row in rows if row["method"] == method and row["seed"] == seed],
                    [row["prediction"] for row in rows if row["method"] == method and row["seed"] == seed],
                )
                for seed in config["training_seeds"]
            }
            for method in methods
        }
    ours_texture = texture["structured_bidirectional"]
    a5_checks = {
        "texture_swap_hard_consistency_ge_0_95": ours_texture["hard_consistency"] >= 0.95,
        "texture_swap_mean_jsd_le_0_05": ours_texture["mean_probability_jsd"] <= 0.05,
        "scene_material_camera_axes_present": set(a5_axes) >= {"scene", "material", "scene_and_material", "camera_profile"},
        "lighting_heldout_axis_present": False,
        "physical_realization_heldout_axis_present": False,
        # C4 is a direct policy-level estimand: selective policy minus always-safe.
        # A4 measures recovery minus continue and therefore cannot stand in for this gate.
        "direct_selective_vs_always_safe_evaluation_present": False,
    }
    a5 = {
        "schema_version": "kinofail.realistic-a5.v1",
        "protocol_id": config["protocol_id"],
        "heldout_balanced_accuracy": a5_axes,
        "texture_swap_consistency": texture,
        "acceptance": a5_checks,
        "passed": all(a5_checks.values()),
    }
    model_runs = [
        {
            "axis": axis,
            "seed": seed,
            "method": method,
            "checkpoint": path,
            "checkpoint_sha256": checkpoint_hashes[path],
        }
        for path in sorted(checkpoint_hashes)
        for axis, seed, method in [(
            path.split("/")[-3],
            int(path.split("/")[-2].removeprefix("seed")),
            Path(path).stem,
        )]
    ]
    training_manifest = {
        "schema_version": "kinofail.realistic-training-manifest.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "confirmatory_complete",
        "dataset_scope": "kinofail_realistic",
        "a0_evidence_bundle_sha256": a0_bundle_sha256,
        "protocol_id": config["protocol_id"],
        "protocol_sha256": _sha256(config_path),
        "feature_manifest_sha256": _sha256(feature_manifest_path),
        "scene_registry_sha256": _sha256(registry_path),
        "training_seeds": config["training_seeds"],
        "methods": methods,
        "model_runs": model_runs,
        "checkpoints_sha256": checkpoint_hashes,
        "forbidden_deployment_inputs": feature_manifest["forbidden_deployment_inputs"],
    }
    _write(output / "training_manifest.json", training_manifest)
    training_manifest_path = output / "training_manifest.json"
    inference_manifest = {
        "schema_version": "kinofail.realistic-inference-manifest.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "confirmatory_complete",
        "dataset_scope": "kinofail_realistic",
        "a0_evidence_bundle_sha256": a0_bundle_sha256,
        "protocol_id": config["protocol_id"],
        "training_manifest_sha256": _sha256(training_manifest_path),
        "predictions": len(predictions),
        "predictions_sha256": _sha256(predictions_path),
    }
    _write(output / "inference_manifest.json", inference_manifest)
    inference_manifest_path = output / "inference_manifest.json"
    bindings = {
        "dataset_scope": "kinofail_realistic",
        "a0_evidence_bundle_sha256": a0_bundle_sha256,
        "training_manifest_sha256": _sha256(training_manifest_path),
        "inference_manifest_sha256": _sha256(inference_manifest_path),
        "predictions_sha256": _sha256(predictions_path),
    }
    for value, status in (
        (a2, "confirmatory_passed" if a2["passed"] else "confirmatory_complete"),
        (a3, "confirmatory_passed" if a3["passed"] else "partial_complete"),
        (a5, "confirmatory_passed" if a5["passed"] else "partial_complete"),
    ):
        value.update(bindings)
        value["status"] = status
    _write(output / "a2_headline.json", a2)
    _write(output / "a3_publication_report.json", a3)
    _write(output / "a5_selective.json", a5)
    print(json.dumps({"predictions": len(predictions), "A2": a2["passed"], "A3": a3["passed"], "A5": a5["passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
