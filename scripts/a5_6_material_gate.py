#!/usr/bin/env python
"""A5.6 — fit or independently evaluate the supported structured recovery gate.

``fit-development`` uses the original calibration split plus data that have already been inspected
(the original held-out appearances and the first extreme-colour stress set).  It may improve and
freeze the method, but it can never create a headline result.

``evaluate-final`` loads a frozen YAML gate and scores a separately collected final corpus without
refitting any component.  Both modes compose decisions with the unchanged A4 intervention matrix;
the deployed gate reads only the VLA action, its emitted attribution, and an RGB support distance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from kino_vla.eval.a4_consequence import (
    Outcome,
    a3_row_to_a4_scenario,
    physical_cost,
    primitive_to_label,
)
from kino_vla.eval.material_support import (
    dominant_chromatic_rgb,
    fit_material_prototypes,
    material_support_distance,
)
from kino_vla.eval.structured_gate import (
    bootstrap_gate,
    evaluate_gate,
    fit_supported_structured_gate,
    gate_acts,
)
from kino_vla.utils.config import load_config

SAFE_LABEL = "backstep_detour"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _episode_costs(path: Path) -> dict[tuple[str, str], list[float]]:
    cells: dict[tuple[str, str], list[float]] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        outcome = Outcome(**json.loads(line))
        cells.setdefault((outcome.scenario, outcome.label), []).append(physical_cost(outcome))
    return cells


class RgbIndex:
    """Read-only sample-id index over frozen corpus archives."""

    def __init__(self, directories: list[str]) -> None:
        self._entries: dict[str, tuple[np.lib.npyio.NpzFile, str, str]] = {}
        self._archives: list[np.lib.npyio.NpzFile] = []
        self.manifests: dict[str, dict[str, Any]] = {}
        for value in directories:
            directory = Path(value)
            archive = np.load(directory / "frames.npz")
            self._archives.append(archive)
            for line in (directory / "samples.jsonl").read_text().splitlines():
                if not line:
                    continue
                record = json.loads(line)
                sid = str(record["sample_id"])
                key = f"{sid}__rgb"
                proprio_key = f"{sid}__proprio"
                if sid in self._entries:
                    raise ValueError(f"duplicate sample id across RGB corpora: {sid}")
                if key not in archive:
                    raise KeyError(f"{directory}: missing {key}")
                if proprio_key not in archive:
                    raise KeyError(f"{directory}: missing {proprio_key}")
                self._entries[sid] = (archive, key, proprio_key)
            manifest_path = directory / "frozen_manifest.json"
            if manifest_path.exists():
                self.manifests[str(directory)] = _json(manifest_path)

    def feature(self, sample_id: str) -> list[float]:
        archive, key, _ = self._entries[sample_id]
        return dominant_chromatic_rgb(archive[key]).round(6).tolist()

    def tracking_peak(self, sample_id: str) -> float:
        """Maximum observable velocity-tracking error in the frozen 500 ms window."""
        archive, _, key = self._entries[sample_id]
        proprio = np.asarray(archive[key], dtype=np.float64)
        if proprio.ndim != 2 or proprio.shape[1] <= 6:
            raise ValueError(f"invalid proprio tensor for {sample_id}: {proprio.shape}")
        return float(np.max(proprio[:, 6]))

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self._entries


def _agent_label(row: dict[str, Any]) -> str:
    primitive = row.get("primitive")
    if not primitive:
        return "continue"
    if primitive == "Switch_Gait" and not row.get("action_params_observed", False):
        raise ValueError(f"row {row.get('sid')} lost the observed Switch_Gait mode")
    return primitive_to_label(str(primitive), row.get("primitive_params") or {}, strict_params=True)


def _attach_support(
    points: list[dict[str, Any]], index: RgbIndex, material_model: dict[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for point in points:
        sid = str(point["sample_id"])
        if sid not in index:
            raise KeyError(f"no RGB tensor for scored row {sid}")
        feature = index.feature(sid)
        out.append(
            {
                **point,
                "material_rgb": feature,
                "material_distance": material_support_distance(feature, material_model),
                "tracking_error_peak": index.tracking_peak(sid),
            }
        )
    return out


def _original_points(
    points_path: Path,
    decisions_path: Path,
    index: RgbIndex,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    points = _json(points_path)
    decisions = {str(row["sid"]): row for row in _json(decisions_path)}
    featured: list[dict[str, Any]] = []
    for row in points:
        sid = str(row["sample_id"])
        decision = decisions.get(sid)
        if decision is None:
            raise KeyError(f"original decision missing for A7 point {sid}")
        feature = index.feature(sid)
        featured.append(
            {
                **row,
                "attribution": decision.get("attribution"),
                "semantic_class": row.get("truth"),
                "material_rgb": feature,
                "method_split": (
                    "train" if str(row["appearance_split"]) == "train" else "development"
                ),
            }
        )
    calibration = [row for row in featured if row["method_split"] == "train"]
    model = fit_material_prototypes(calibration, target_class="compliant_terrain")
    return _attach_support(featured, index, model), model


def _decision_points(
    decisions_path: Path,
    index: RgbIndex,
    a4: dict[str, Any],
    material_model: dict[str, Any],
    *,
    method_split: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    decisions = _json(decisions_path)
    for row in decisions:
        scenario = a3_row_to_a4_scenario(row)
        if scenario is None or scenario not in a4["M_full"]:
            raise ValueError(f"final decision cannot be mapped to A4: {row.get('sid')}")
        sid = str(row["sid"])
        label = _agent_label(row)
        cells = a4["M_full"][scenario]
        if int(cells.get(label, {}).get("n", 0)) == 0:
            raise ValueError(f"unmeasured A4 action for {scenario}|{label}")
        points.append(
            {
                "sample_id": sid,
                "scenario": scenario,
                "truth": row.get("truth"),
                "cell": row.get("cell"),
                "appearance_id": str(row.get("appearance_id", "")),
                "appearance_split": str(row.get("appearance_split", "")),
                "method_split": method_split,
                "cluster_id": f"{scenario}|{row.get('appearance_id', '')}",
                "agent_label": label,
                "attribution": row.get("attribution"),
                "attr_ok": bool(row.get("attr_ok")),
                "safe_label": SAFE_LABEL,
                "continue_label": "continue",
                "cost_agent": float(cells[label]["mean_cost"]),
                "cost_safe": float(cells[SAFE_LABEL]["mean_cost"]),
                "cost_continue": float(cells["continue"]["mean_cost"]),
            }
        )
    if len(points) != len(decisions):
        raise AssertionError("decision-to-cost join dropped rows")
    return _attach_support(points, index, material_model)


def _corpus_evidence(index: RgbIndex) -> dict[str, Any]:
    return {
        directory: {
            "corpus_hash_sha256": manifest.get("corpus_hash_sha256"),
            "config_sha256": manifest.get("config_sha256"),
            "gate_config_sha256": manifest.get("gate_config_sha256"),
            "n_snapshots": manifest.get("n_snapshots"),
        }
        for directory, manifest in index.manifests.items()
    }


def fit_development(args: argparse.Namespace) -> dict[str, Any]:
    original_index = RgbIndex(args.original_dirs.split(","))
    original, material_model = _original_points(
        Path(args.original_points), Path(args.original_decisions), original_index
    )
    development_index = RgbIndex(args.development_dirs.split(","))
    a4 = _json(Path(args.a4_results))
    extreme = _decision_points(
        Path(args.development_decisions),
        development_index,
        a4,
        material_model,
        method_split="development",
    )
    fit_rows = original + extreme
    canonical_cfg = load_config("data/hindsight.yaml")
    canonical = {
        str(key): str(value) for key, value in canonical_cfg.recovery.canonical.to_dict().items()
    }
    gate = fit_supported_structured_gate(
        fit_rows,
        canonical,
        min_fit_coverage=float(args.minimum_coverage),
        allowed_strata=[("high_step", "compliant_terrain")],
        validity_field="attr_ok",
        evidence_field="tracking_error_peak",
    )
    calibration = [row for row in original if row["method_split"] == "train"]
    old_test = [row for row in original if row["method_split"] == "development"]
    episode_costs = _episode_costs(Path(args.matrix))
    return {
        "schema_version": 2,
        "status": "method_development_not_headline",
        "reason": "all development rows were inspected before v2 was frozen",
        "method": "See-Feel-Act consensus plus calibrated RGB material-support fallback",
        "input_sha256": {
            "original_points": _sha256(Path(args.original_points)),
            "original_decisions": _sha256(Path(args.original_decisions)),
            "development_decisions": _sha256(Path(args.development_decisions)),
            "a4_results": _sha256(Path(args.a4_results)),
            "matrix": _sha256(Path(args.matrix)),
        },
        "corpus_evidence": {
            **_corpus_evidence(original_index),
            **_corpus_evidence(development_index),
        },
        "n": {
            "calibration": len(calibration),
            "original_heldout_development": len(old_test),
            "extreme_colour_development": len(extreme),
        },
        "material_model": material_model,
        "gate": gate,
        "calibration": evaluate_gate(calibration, gate),
        "original_heldout_development": evaluate_gate(old_test, gate),
        "extreme_colour_development": evaluate_gate(extreme, gate),
        "extreme_colour_two_stage_bootstrap": bootstrap_gate(
            extreme,
            gate,
            episode_costs,
            reps=int(args.bootstrap_reps),
            seed=int(args.bootstrap_seed),
        ),
        "headline_gate": {
            "eligible": False,
            "required_next": "freeze v2 and evaluate once on a new untouched final-v2 corpus",
        },
        "non_interference": {
            "attributor_weights_modified": False,
            "projector_weights_modified": False,
            "training_data_modified": False,
            "a1_a4_estimands_modified": False,
        },
    }


def evaluate_final(args: argparse.Namespace) -> dict[str, Any]:
    gate_path = Path(args.gate_config)
    frozen = yaml.safe_load(gate_path.read_text())
    gate = frozen["gate"]
    material_model = frozen["material_model"]
    index = RgbIndex(args.final_dirs.split(","))
    a4 = _json(Path(args.a4_results))
    rows = _decision_points(
        Path(args.final_decisions),
        index,
        a4,
        material_model,
        method_split="final",
    )
    point = evaluate_gate(rows, gate)
    bootstrap = bootstrap_gate(
        rows,
        gate,
        _episode_costs(Path(args.matrix)),
        reps=int(args.bootstrap_reps),
        seed=int(args.bootstrap_seed),
    )
    coverage_ok = float(point["coverage"]) >= float(frozen["primary_endpoint"]["minimum_coverage"])
    ci_ok = bool(bootstrap["strictly_better_with_95ci"]["safe"])
    all_final = bool(rows) and all(row["method_split"] == "final" for row in rows)
    acted = [row for row in rows if gate_acts(row, gate)]
    acted_clusters = sorted({str(row["cluster_id"]) for row in acted})
    acted_appearances = sorted({str(row["appearance_id"]) for row in acted})
    release_precision = sum(bool(row["attr_ok"]) for row in acted) / max(1, len(acted))
    semantic_ok = bool(acted) and all(bool(row["attr_ok"]) for row in acted)
    cluster_deltas: dict[str, list[float]] = {}
    for row in rows:
        delta = float(row["cost_agent"]) - float(row["cost_safe"]) if gate_acts(row, gate) else 0.0
        cluster_deltas.setdefault(str(row["cluster_id"]), []).append(delta)
    mean_cluster_deltas = {
        cluster: float(np.mean(values)) for cluster, values in sorted(cluster_deltas.items())
    }
    released_cluster_deltas = {cluster: mean_cluster_deltas[cluster] for cluster in acted_clusters}
    n_strict_cluster_gains = sum(value < 0.0 for value in released_cluster_deltas.values())
    n_released_clusters = len(released_cluster_deltas)
    cluster_sign_p = sum(
        math.comb(n_released_clusters, index)
        for index in range(n_strict_cluster_gains, n_released_clusters + 1)
    ) / (2**n_released_clusters)
    no_cluster_harm = max(mean_cluster_deltas.values()) <= 0.0
    gate_hash = _sha256(gate_path)
    gate_hash_match = all(
        manifest.get("gate_config_sha256") == gate_hash for manifest in index.manifests.values()
    )
    return {
        "schema_version": 2,
        "status": "confirmatory_final",
        "method": frozen["method"],
        "frozen_gate_config": str(gate_path),
        "frozen_gate_config_sha256": gate_hash,
        "input_sha256": {
            "final_decisions": _sha256(Path(args.final_decisions)),
            "a4_results": _sha256(Path(args.a4_results)),
            "matrix": _sha256(Path(args.matrix)),
        },
        "corpus_evidence": _corpus_evidence(index),
        "n": len(rows),
        "all_rows_final_split": all_final,
        "point_estimate": point,
        "two_stage_bootstrap": bootstrap,
        "released_interventions": {
            "n": len(acted),
            "n_appearance_clusters": len(acted_clusters),
            "n_unique_appearances": len(acted_appearances),
            "appearance_clusters": acted_clusters,
            "cluster_agent_minus_safe": {
                cluster: round(value, 6) for cluster, value in released_cluster_deltas.items()
            },
            "n_strictly_better_clusters": n_strict_cluster_gains,
            "one_sided_exact_sign_p": round(cluster_sign_p, 8),
            "attribution_precision": round(release_precision, 4),
            "all_attributions_correct": semantic_ok,
            "max_cluster_agent_minus_safe": round(max(mean_cluster_deltas.values()), 6),
            "no_final_cluster_harm": no_cluster_harm,
        },
        "headline_gate": {
            "minimum_coverage": float(frozen["primary_endpoint"]["minimum_coverage"]),
            "coverage_pass": coverage_ok,
            "paired_ci_upper_lt_zero_vs_always_safe": ci_ok,
            "independent_final_split": all_final,
            "corpus_gate_hash_matches_frozen_config": gate_hash_match,
            "released_intervention_precision_is_one": semantic_ok,
            "no_final_appearance_cluster_harm": no_cluster_harm,
            "eligible": (
                coverage_ok
                and ci_ok
                and all_final
                and gate_hash_match
                and semantic_ok
                and no_cluster_harm
            ),
        },
        "non_interference": frozen["non_interference"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A5.6 supported structured gate")
    parser.add_argument(
        "--mode", choices=("fit-development", "evaluate-final"), default="fit-development"
    )
    parser.add_argument(
        "--original-points", default="outputs/eval/a7/abstention_baselines/per_item_scores.json"
    )
    parser.add_argument(
        "--original-decisions", default="outputs/eval/a3/per_item_B5_conflict_bi.json"
    )
    parser.add_argument(
        "--original-dirs", default="outputs/eval/a0/corpus,outputs/eval/a3/corpus_t3"
    )
    parser.add_argument(
        "--development-decisions",
        default="outputs/eval/a5/c4_final_eval/per_item_B5_conflict_bi.json",
    )
    parser.add_argument(
        "--development-dirs",
        default="outputs/eval/a5/c4_final_main,outputs/eval/a5/c4_final_t3",
    )
    parser.add_argument("--gate-config", default="configs/eval/c4_structured_gate_v2.yaml")
    parser.add_argument("--final-decisions", default="")
    parser.add_argument("--final-dirs", default="")
    parser.add_argument("--a4-results", default="outputs/eval/a4/a4_results.json")
    parser.add_argument("--matrix", default="outputs/eval/a4/matrix.jsonl")
    parser.add_argument("--minimum-coverage", type=float, default=0.10)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--out", default="outputs/eval/a5/a5_6_material_gate_development.json")
    args = parser.parse_args()
    if args.mode == "evaluate-final" and (not args.final_decisions or not args.final_dirs):
        parser.error("evaluate-final requires --final-decisions and --final-dirs")
    result = fit_development(args) if args.mode == "fit-development" else evaluate_final(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
