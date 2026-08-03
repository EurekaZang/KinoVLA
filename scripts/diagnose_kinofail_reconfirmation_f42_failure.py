#!/usr/bin/env python3
"""Post-hoc, non-confirmatory diagnosis of the sealed F42 negative result."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import eval_kinofail_realistic_c2_bidirectional_v4  # noqa: E402,F401

F42 = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42"
NEW = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f39/conflict/features"
DEV = ROOT / "outputs/eval/c2_bidirectional_v5/development_features_six_scene"
OLD_FORMAL = ROOT / "outputs/eval/c2_bidirectional_v5/formal_features"
SPECIALIST = ROOT / "outputs/eval/c2_bidirectional_v5/formal/structured_post_interaction_router_v5.pkl"
OUTPUT = F42 / "posthoc_failure_diagnosis.json"
OLD_CORPORA = (
    ROOT / "outputs/kinofail_realistic/corpus_c2_bidirectional_confirmation_v3_t2_amendment1",
    ROOT / "outputs/kinofail_realistic/corpus_c2_bidirectional_confirmation_v4_t2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def arrays(root: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows = read_jsonl(root / "records.jsonl")
    with np.load(root / "features.npz", allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        proprio = np.asarray(archive["proprio"], dtype=np.float32)
        embedded_geometry = (
            np.asarray(archive["geometry"], dtype=np.float32)
            if "geometry" in archive.files
            else None
        )
    geometry_path = root / "geometry.npz"
    if embedded_geometry is not None:
        geometry = embedded_geometry
    else:
        with np.load(geometry_path, allow_pickle=False) as archive:
            geometry_ids = archive["sample_ids"].astype(str)
            geometry = np.asarray(archive["geometry"], dtype=np.float32)
        if geometry_ids.tolist() != ids.tolist():
            raise RuntimeError(f"geometry misalignment: {root}")
    if ids.tolist() != [str(row["sample_id"]) for row in rows]:
        raise RuntimeError(f"feature-record misalignment: {root}")
    return rows, visual, geometry, proprio


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
    }


def paired_distances(
    rows: list[dict[str, Any]], visual: np.ndarray, geometry: np.ndarray
) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        if row.get("cell") != "T2_vision_decisive":
            continue
        groups[(str(row["case_id"]), str(row["appearance_view_id"]))][
            str(row["target_operator"])
        ] = index
    valid = [value for value in groups.values() if set(value) == {"O2_compliance", "O4_tether"}]
    visual_l2 = []
    visual_cosine = []
    geometry_l2 = []
    for pair in valid:
        a, b = pair["O2_compliance"], pair["O4_tether"]
        va, vb = visual[a].astype(np.float64), visual[b].astype(np.float64)
        visual_l2.append(float(np.linalg.norm(va - vb)))
        visual_cosine.append(
            float(1.0 - np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12))
        )
        geometry_l2.append(float(np.linalg.norm(geometry[a].astype(np.float64) - geometry[b].astype(np.float64))))
    return {
        "paired_case_views": len(valid),
        "global_visual_l2": quantiles(np.asarray(visual_l2)),
        "global_visual_cosine_distance": quantiles(np.asarray(visual_cosine)),
        "ground_roi_hog_l2": quantiles(np.asarray(geometry_l2)),
    }


def standardized_shift(reference: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    keep = std > 1e-6
    z = np.abs((target[:, keep] - mean[keep]) / std[keep])
    return {
        "reference_row_norm_mean": float(np.linalg.norm(reference, axis=1).mean()),
        "target_row_norm_mean": float(np.linalg.norm(target, axis=1).mean()),
        "target_to_reference_norm_ratio": float(
            np.linalg.norm(target, axis=1).mean()
            / np.linalg.norm(reference, axis=1).mean()
        ),
        "median_absolute_reference_z": float(np.median(z)),
        "p95_absolute_reference_z": float(np.quantile(z, 0.95)),
        "fraction_absolute_reference_z_above_5": float((z > 5).mean()),
        "mean_shift_rms_in_reference_sd": float(
            np.sqrt(np.mean(((target.mean(axis=0)[keep] - mean[keep]) / std[keep]) ** 2))
        ),
    }


def resolve_old_case(case_id: str) -> Path:
    matches = [root / case_id for root in OLD_CORPORA if (root / case_id).is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve one old T2 case: {case_id}: {matches}")
    return matches[0]


def image_pair_metric(first: Path, second: Path) -> dict[str, float]:
    a = np.asarray(Image.open(first).convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(second).convert("RGB"), dtype=np.float32) / 255.0
    delta = np.abs(a - b).mean(axis=2)
    changed = delta > 0.05
    if changed.any():
        ys, xs = np.where(changed)
        bbox_fraction = float(
            (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) / changed.size
        )
        centroid_y_fraction = float(ys.mean() / max(delta.shape[0] - 1, 1))
    else:
        bbox_fraction = 0.0
        centroid_y_fraction = 1.0
    return {
        "mean_rgb_l1": float(delta.mean()),
        "changed_pixel_fraction_at_0_05": float(changed.mean()),
        "changed_bbox_fraction": bbox_fraction,
        "changed_centroid_y_fraction": centroid_y_fraction,
    }


def representative_pixel_cues(rows: list[dict[str, Any]], *, new: bool) -> dict[str, Any]:
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("cell") != "T2_vision_decisive" or row.get("appearance_view_id") != "primary":
            continue
        scene = str(row["scene_cluster"])
        if candidates[scene] and str(row["case_id"]) not in {
            str(value["case_id"]) for value in candidates[scene].values()
        }:
            continue
        candidates[scene][str(row["target_operator"])] = row
    metrics = []
    examples = []
    for scene, pair in sorted(candidates.items()):
        if set(pair) != {"O2_compliance", "O4_tether"}:
            continue
        o2, o4 = pair["O2_compliance"], pair["O4_tether"]
        case_root = (
            Path(str(o2["source_case_dir"]))
            if new
            else resolve_old_case(str(o2["case_id"]))
        )
        first = case_root / str(o2["rgb_paths"][-1])
        second = case_root / str(o4["rgb_paths"][-1])
        metric = image_pair_metric(first, second)
        metric["scene"] = scene
        metrics.append(metric)
        examples.append(
            {
                "scene": scene,
                "o2": str(first.relative_to(ROOT)),
                "o4": str(second.relative_to(ROOT)),
                "o2_sha256": sha256(first),
                "o4_sha256": sha256(second),
            }
        )
    numeric = {
        key: quantiles(np.asarray([row[key] for row in metrics], dtype=np.float64))
        for key in (
            "mean_rgb_l1",
            "changed_pixel_fraction_at_0_05",
            "changed_bbox_fraction",
            "changed_centroid_y_fraction",
        )
    }
    return {"scene_representatives": len(metrics), "summaries": numeric, "examples": examples}


def frozen_specialist(
    rows: list[dict[str, Any]], visual: np.ndarray, geometry: np.ndarray, proprio: np.ndarray
) -> dict[str, Any]:
    with SPECIALIST.open("rb") as stream:
        model = pickle.load(stream)  # noqa: S301 - hash-pinned local post-hoc audit
    labels = np.asarray([str(row["attribution_category"]) for row in rows])
    cells = np.asarray([str(row["cell"]) for row in rows])
    probability = model.predict_proba(visual, geometry, proprio)
    prediction = np.asarray(model.classes)[probability.argmax(axis=1)]
    family = model.family_prediction(visual, geometry, proprio)
    return {
        "checkpoint_sha256": sha256(SPECIALIST),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(np.mean(labels == prediction)),
        "per_cell_accuracy": {
            cell: float(np.mean(labels[cells == cell] == prediction[cells == cell]))
            for cell in sorted(set(cells))
        },
        "per_cell_family_route_fidelity": {
            cell: float(np.mean(family[cells == cell] == cell))
            for cell in sorted(set(cells))
        },
        "posthoc_only": True,
        "refit": False,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = read_json(F42 / "confirmatory_report.json")
    a6 = read_json(F42 / "a6_operator_boundary_report.json")
    dev_rows, dev_visual, dev_geometry, dev_proprio = arrays(DEV)
    old_rows, old_visual, old_geometry, old_proprio = arrays(OLD_FORMAL)
    new_rows, new_visual, new_geometry, new_proprio = arrays(NEW)
    if report.get("passed_all_preregistered_gates") is not False or a6.get("passed") is not False:
        raise RuntimeError("F42/A6 must remain negative for this diagnostic")
    result = {
        "schema_version": "kinofail.reconfirmation-f42-posthoc-diagnosis.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "posthoc_failure_diagnosis_complete",
        "confirmatory_evidence": False,
        "model_or_threshold_selected_from_f42": False,
        "f42_negative_result_preserved": True,
        "headline": {
            "scale_ours_minus_late": report["primary_cross_battery_gates"][
                "gate_1_battery_noninferiority"
            ]["batteries"]["scale"]["difference_ours_minus_late"],
            "conflict_ours_minus_late": report["primary_cross_battery_gates"][
                "gate_1_battery_noninferiority"
            ]["batteries"]["conflict"]["difference_ours_minus_late"],
            "t2_route_fidelity": report["route_fidelity_gates"]["T2_vision_decisive"][
                "case_weighted_fidelity"
            ],
            "t3_route_fidelity": report["route_fidelity_gates"]["T3_proprio_decisive"][
                "case_weighted_fidelity"
            ],
            "o9_point_accuracy": a6["operator_cells"]["O9_high_centering"][
                "learned_router"
            ]["overall"]["five_checkpoint_sample_mean"],
            "minimum_o9_parameter_point_accuracy": a6["summaries"][
                "minimum_o9_parameter_point_accuracy"
            ],
        },
        "t2_cue_shift": {
            "development": paired_distances(dev_rows, dev_visual, dev_geometry),
            "old_formal": paired_distances(old_rows, old_visual, old_geometry),
            "f42": paired_distances(new_rows, new_visual, new_geometry),
            "representative_pixel_cues_development": representative_pixel_cues(
                dev_rows, new=False
            ),
            "representative_pixel_cues_f42": representative_pixel_cues(new_rows, new=True),
        },
        "feature_distribution_shift": {
            "visual_dev_to_old_formal": standardized_shift(dev_visual, old_visual),
            "visual_dev_to_f42": standardized_shift(dev_visual, new_visual),
            "proprio_dev_to_old_formal": standardized_shift(dev_proprio, old_proprio),
            "proprio_dev_to_f42": standardized_shift(dev_proprio, new_proprio),
        },
        "frozen_structured_v5_on_f42": frozen_specialist(
            new_rows, new_visual, new_geometry, new_proprio
        ),
        "interpretation": {
            "t2": (
                "Global visual embeddings dilute small bottom-of-frame tether cues; the "
                "frozen HOG-augmented specialist recovers part, but not all, of T2."
            ),
            "t3": (
                "The frozen specialist routes every F42 T3 sample to the T2 family; raw "
                "proprio summaries are not invariant to the new physical regime."
            ),
            "o9": (
                "Failure across the full 16-point direct-contact continuum rules out one "
                "isolated parameter point and indicates a representation/temporal-domain mismatch."
            ),
        },
        "source_sha256": {
            "f42_report": sha256(F42 / "confirmatory_report.json"),
            "f42_a6": sha256(F42 / "a6_operator_boundary_report.json"),
            "new_features": sha256(NEW / "features.npz"),
            "new_geometry": sha256(NEW / "geometry.npz"),
            "new_records": sha256(NEW / "records.jsonl"),
            "development_features": sha256(DEV / "features.npz"),
            "old_formal_features": sha256(OLD_FORMAL / "features.npz"),
            "script": sha256(Path(__file__).resolve()),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "headline": result["headline"],
        "frozen_structured_v5_on_f42": result["frozen_structured_v5_on_f42"],
        "feature_distribution_shift": result["feature_distribution_shift"],
        "output": str(OUTPUT),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
