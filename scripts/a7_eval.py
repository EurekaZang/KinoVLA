#!/usr/bin/env python
# ruff: noqa: E501
"""A7 frozen-snapshot evaluation and artifact aggregation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kino_vla.eval.a4_consequence import (
    Outcome,
    a3_row_to_a4_scenario,
    attribution_to_label,
    physical_cost,
    primitive_to_label,
)
from kino_vla.eval.a7_ablation import (
    aggregate_dose_curves,
    artifact_meta,
    auroc,
    conformal_risk_control,
    dose_curve_summary,
    ers_regret_from_agent_rows,
    expected_calibration_error,
    grounding_summary,
    heldout_threshold_selection,
    load_json,
    load_yaml,
    posterior_mean_variance,
    repo_path,
    risk_coverage_auc,
    risk_coverage_curve,
    status_counts,
    wilson,
    write_json,
)


def _policy(route: str, adapter: str | None, *, proprio_detail: str = "binned"):
    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.utils.config import load_config
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy

    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    overrides = {"route": route, "data.proprio_detail": proprio_detail}
    vcfg = load_config("vla/sft.yaml", overrides)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = KinoVLA.from_pretrained(vcfg, device=dev, adapter_dir=adapter)
    model.eval()
    return ModelVlaPolicy(
        model,
        pcfg,
        tax,
        route=route,
        n_images=int(vcfg.data.get("n_images", 1)),
        temperature=0.0,
        proprio_detail=proprio_detail,
    )


ATTR_TO_PRIMITIVE = {
    "adhesion": "Backstep",
    "compliant_terrain": "Switch_Gait",
    "low_friction": "Set_Constraint",
    "invisible_obstacle": "Update_Topology",
    "overload": "Hold_and_Request",
    "effort_decay": "Switch_Gait",
    "nominal": "Continue",
    "external_push": "Set_Constraint",
    "region_collapse": "Update_Topology",
    "high_centering": "Adjust_Posture",
    "obs_bias": "Set_Constraint",
}


PRIMITIVE_PARAMS = {
    "Backstep": {"distance_m": 1.0},
    "Switch_Gait": {"mode": "high_step"},
    "Set_Constraint": {"max_speed": 0.4, "stiffness": 0.5},
    "Adjust_Posture": {"body_height_m": 0.25, "pitch_deg": 0.0},
    "Hold_and_Request": {"reason": "uncertain failure attribution"},
    "Update_Topology": {"region_xy": [0.0, 0.0], "radius_m": 1.0, "status": "untraversable"},
    "Replan_Waypoint": {"point_px": [480, 360]},
}


def eval_a2_adapter(
    adapter: str | None, route: str, proprio_detail: str, corpus: str
) -> tuple[list[dict], dict]:
    from kino_vla.eval.a2_headline import aggregate, eval_policy, load_matched_corpus

    items = load_matched_corpus(corpus)
    pol = _policy(route, adapter, proprio_detail=proprio_detail)
    res = eval_policy(pol, items)
    rows = [asdict(r) for r in res]
    return rows, aggregate(res)


def eval_a2_adapter_with_thoughts(
    adapter: str | None, route: str, proprio_detail: str, corpus: str
) -> list[dict[str, Any]]:
    """Matched-corpus eval that also records emitted ``<Thought>`` text for grounding checks."""
    from kino_vla.eval.a2_headline import load_matched_corpus

    items = load_matched_corpus(corpus)
    pol = _policy(route, adapter, proprio_detail=proprio_detail)
    rows: list[dict[str, Any]] = []
    for it in items:
        decision = pol.decide(it.snapshot)
        parsed = bool(decision.ok and decision.annotation is not None)
        attribution = decision.attribution if parsed else None
        primitive = decision.primitive_name if parsed else None
        attr_ok = bool(parsed and attribution == it.truth_category)
        rows.append(
            {
                "sample_id": it.sample_id,
                "operator": it.operator,
                "appearance_id": it.appearance_id,
                "appearance_split": it.appearance_split,
                "truth_category": it.truth_category,
                "parsed": parsed,
                "attribution": attribution,
                "attr_ok": attr_ok,
                "primitive": primitive,
                "prim_feasible": bool(parsed and primitive in it.admissible_set),
                "joint_ok": bool(attr_ok and parsed and primitive in it.admissible_set),
                "thought": decision.thought if parsed else "",
            }
        )
    return rows


def _load_eval_a3():
    """Load ``scripts/eval_a3.py`` as a module (script path is not a package)."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parent / "eval_a3.py"
    name = "kino_eval_a3_mod"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load eval_a3 from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def eval_a3_adapter(
    adapter: str | None, route: str, proprio_detail: str, a0: str, t3: str
) -> tuple[list[dict], dict]:
    ea = _load_eval_a3()

    snaps = ea.load_merged(a0, t3)
    pol = _policy(route, adapter, proprio_detail=proprio_detail)

    def key_fn(snapshot):
        return ea.vla_input_key(
            snapshot,
            cfg=pol._cfg,
            route=pol._route,
            n_images=pol._n_images,
            proprio_detail=pol._proprio_detail,
            mask=pol._mask_proprio,
        )

    rows = ea.score_agent(pol, snaps, input_key=key_fn)
    heatmap = {cell: ea.cell_acc(rows, cell) for cell in ea.CELLS}
    return rows, heatmap


def eval_a3_adapter_with_thoughts(
    adapter: str | None, route: str, proprio_detail: str, a0: str, t3: str
) -> tuple[list[dict], dict]:
    """Full-taxonomy eval that records emitted thoughts (for T4 + test-time grounding)."""
    ea = _load_eval_a3()

    snaps = ea.load_merged(a0, t3)
    pol = _policy(route, adapter, proprio_detail=proprio_detail)
    rows: list[dict] = []
    for s in snaps:
        dec = pol.decide(s.snapshot)
        parsed = bool(dec.ok and dec.annotation is not None)
        attr = dec.attribution if parsed else None
        prim = dec.primitive_name if parsed else None
        attr_ok = bool(parsed and attr == s.truth)
        rows.append(
            {
                "sid": s.sid,
                "cell": s.cell,
                "t3_sub": s.t3_sub,
                "truth": s.truth,
                "truth_category": s.truth,
                "attribution": attr,
                "primitive": prim,
                "attr_ok": attr_ok,
                "mu": s.mu,
                "parsed": parsed,
                "thought": dec.thought if parsed else "",
            }
        )
    heatmap = {cell: ea.cell_acc(rows, cell) for cell in ea.CELLS}
    return rows, heatmap


def _m7_route_summary(cfg: dict[str, Any], config_path: str) -> dict[str, Any]:
    t00 = load_json("outputs/vla/ablation/summary_t00.json")
    t08 = load_json("outputs/vla/ablation/summary_t08.json")
    by_arm_t00 = {r["arm"]: r for r in t00}
    by_arm_t08 = {r["arm"]: r for r in t08}
    arms = {}
    for arm in ("vision_only", "text_scalar", "text_binned", "latent"):
        r0 = by_arm_t00[arm]
        r8 = by_arm_t08.get(arm, {})
        arms[arm] = {
            "route": r0["route"],
            "proprio_detail": r0["proprio_detail"],
            "attr_ambiguous_greedy": r0["attr_ambiguous"],
            "attr_ambiguous_greedy_ci": list(
                wilson(round(r0["attr_ambiguous"] * r0["n_ambiguous"]), r0["n_ambiguous"])
            ),
            "attr_overall_greedy": r0["attr_overall"],
            "parse_rate": r0["parse_rate"],
            "mean_prompt_tokens": r0["mean_prompt_tokens"],
            "theta_mae": r0.get("theta_mae"),
            "sampled_ambiguous_mean": (r8.get("sampled") or {}).get("ambiguous_attr_acc_mean"),
            "sampled_ambiguous_std": (r8.get("sampled") or {}).get("ambiguous_attr_acc_std"),
        }
    token_ratio = round(
        arms["text_binned"]["mean_prompt_tokens"] / arms["latent"]["mean_prompt_tokens"], 2
    )
    result = {
        **artifact_meta(
            config_path,
            sources={
                "m7_summary_t00": "outputs/vla/ablation/summary_t00.json",
                "m7_summary_t08": "outputs/vla/ablation/summary_t08.json",
            },
        ),
        "stage": "text_schema_existing",
        "status": "available",
        "metric": "M7 ambiguity-regime route ablation; real Qwen3-VL-4B adapters",
        "arms": arms,
        "headline": {
            "vision_only_ambiguous": arms["vision_only"]["attr_ambiguous_greedy"],
            "text_scalar_ambiguous": arms["text_scalar"]["attr_ambiguous_greedy"],
            "text_binned_ambiguous": arms["text_binned"]["attr_ambiguous_greedy"],
            "latent_ambiguous": arms["latent"]["attr_ambiguous_greedy"],
            "latent_vs_text_binned_token_ratio": token_ratio,
            "latent_theta_decodable": arms["latent"]["theta_mae"] is not None,
        },
        "interpretation_guardrail": "greedy accuracy ties among proprio routes; latent claim is efficiency + theta grounding + sampling robustness unless rich text later opens/closes another gap",
    }
    out = repo_path(cfg["output_dir"]) / "text_schema_existing.json"
    write_json(out, result)
    return result


def _a3_method_summary(cfg: dict[str, Any], config_path: str) -> dict[str, Any]:
    a3 = load_json(cfg["sources"]["a3_battery"])
    heat = a3["heatmap"]
    rows = {}
    for agent in ["B1", "B-T", "B-V", "B-F", "B5-unshaped", "B5-conflict", "B5-conflict-bi"]:
        if agent not in heat:
            continue
        rows[agent] = {
            "T2": heat[agent]["T2"],
            "T3": heat[agent]["T3"],
            "T4": heat[agent]["T4"],
            "T5": heat[agent]["T5"],
        }
    result = {
        **artifact_meta(config_path, sources={"a3_battery": cfg["sources"]["a3_battery"]}),
        "stage": "a3_method_slices",
        "status": "available",
        "metric": "A3 taxonomy-cell attribution accuracy reused as A7 method slices",
        "rows": rows,
        "headline": {
            "text_scalar_BT_T4": rows.get("B-T", {}).get("T4", {}).get("acc"),
            "latent_conflict_B5_T2": rows.get("B5-conflict", {}).get("T2", {}).get("acc"),
            "bidirectional_B5_bi_T3": rows.get("B5-conflict-bi", {}).get("T3", {}).get("acc"),
            "proprio_B1_T4": rows.get("B1", {}).get("T4", {}).get("acc"),
        },
        "scope_note": "B5-conflict-bi is a conflict-only specialist; T4 is carried by B1/proprio and by the original M7 route ablation, not by the A3 conflict-only adapter.",
    }
    write_json(repo_path(cfg["output_dir"]) / "a3_method_slices.json", result)
    return result


def _addons(cfg: dict[str, Any], config_path: str) -> dict[str, Any]:
    a5 = load_json(cfg["sources"]["a5_ood_theta"])
    risk = load_json(cfg["sources"]["a5_risk_coverage"])
    a6_eval = load_json(cfg["sources"]["a6_eval"])
    result = {
        **artifact_meta(
            config_path,
            sources={
                "a5_ood_theta": cfg["sources"]["a5_ood_theta"],
                "a5_risk_coverage": cfg["sources"]["a5_risk_coverage"],
                "a6_theta_sweep": cfg["sources"]["a6_theta_sweep"],
                "a6_eval": cfg["sources"]["a6_eval"],
            },
        ),
        "stage": "addons",
        "status": "available",
        "not_core_a7_claims": True,
        "ood_theta_abstention": a5.get("abstention_utility", {}),
        "risk_coverage": risk,
        "theta_boundary_bracket": a6_eval.get("base_boundary", {}).get(
            "primary_identified_bracket"
        ),
        "theta_star_descriptive": a6_eval.get("base_boundary", {}).get(
            "logistic_midpoint_descriptive"
        ),
        "theta_star_warning": a6_eval.get("base_boundary", {}).get("logistic_fit_warning"),
        "theta_star_rows": a6_eval.get("base_boundary", {}).get("rows", []),
        "a6_decision_flip_scope": a6_eval.get("conclusion"),
    }
    write_json(repo_path(cfg["output_dir"]) / "addons.json", result)
    return result


OPERATOR_TO_A4_SCENARIO = {
    "O4_tether": "matched_O4_twophase",
    "O2_compliance": "matched_O2",
    "O7_visual_physics": "O7_looks_safe",
    "O8_invisible_collider": "O8_invisible",
    "O5_payload": "O5_payload_B",
    "O10_effort_decay": "O10_decay_B",
    "O1_low_friction": "O1_A_nominal",
}


def _row_to_a4_projection(row: dict[str, Any]) -> dict[str, Any] | None:
    op = row.get("operator") or row.get("operator_name") or row.get("true_operator")
    scenario = OPERATOR_TO_A4_SCENARIO.get(str(op))
    pred_attr = row.get("pred_attr") or row.get("attribution") or row.get("predicted_attribution")
    if scenario is None or pred_attr is None:
        return None
    return {
        "scenario": scenario,
        "primitive": attribution_to_label(str(pred_attr)),
    }


def _ers_regret(cfg: dict[str, Any], config_path: str) -> dict[str, Any]:
    a4 = load_json(cfg["sources"]["a4_results"])
    agents: dict[str, list[dict[str, Any]]] = {}
    text_dir = repo_path(cfg["output_dir"]) / "text_schema"
    conflict_dir = repo_path(cfg["output_dir"]) / "conflict_dose"

    for name in cfg["text_schemas"]:
        p = text_dir / f"per_item_{name}.json"
        if p.exists():
            rows = load_json(p)
            agents[f"text_schema:{name}"] = [
                x for r in rows if (x := _row_to_a4_projection(r)) is not None
            ]

    complete_conflict_seeds = []
    for seed in [int(x) for x in cfg["seeds"]["train"]]:
        paths = [
            conflict_dir / f"per_item_seed{seed}_dose{int(d):02d}.json"
            for d in cfg["conflict_doses"]["matched_samples"]
        ]
        if all(p.exists() for p in paths):
            complete_conflict_seeds.append((seed, paths))
    for _seed, paths in complete_conflict_seeds:
        for p in paths:
            rows = load_json(p)
            stem = p.stem.replace("per_item_", "")
            agents[f"conflict_dose:{stem}"] = [
                x for r in rows if (x := _row_to_a4_projection(r)) is not None
            ]

    projected = ers_regret_from_agent_rows(agents, a4["M_mean_cost"], a4["canonical_label"])
    by_dose: dict[str, list[dict[str, Any]]] = {}
    for name, row in projected.items():
        if name.startswith("conflict_dose:") and "_dose" in name:
            dose = str(int(name.rsplit("_dose", 1)[1]))
            by_dose.setdefault(dose, []).append(row)
    aggregate_by_dose = {
        dose: {
            "n_seeds": len(rows),
            "mean_cost": round(sum(float(r["mean_cost"]) for r in rows) / len(rows), 3),
            "mean_regret": round(sum(float(r["mean_regret"]) for r in rows) / len(rows), 3),
            "min_regret": round(min(float(r["mean_regret"]) for r in rows), 3),
            "max_regret": round(max(float(r["mean_regret"]) for r in rows), 3),
        }
        for dose, rows in sorted(by_dose.items(), key=lambda kv: int(kv[0]))
    }
    best_regret_dose = None
    if aggregate_by_dose:
        best_regret_dose = int(
            min(aggregate_by_dose.items(), key=lambda kv: (kv[1]["mean_regret"], int(kv[0])))[0]
        )
    result = {
        **artifact_meta(
            config_path,
            sources={
                "a4_results": cfg["sources"]["a4_results"],
                "text_schema_dir": str(text_dir),
                "conflict_dose_dir": str(conflict_dir),
            },
        ),
        "stage": "ers_regret",
        "status": "available" if projected else "requires_evaluated_per_item",
        "projection_rule": (
            "attribution-implied A4 label projection; cached A7/A2 rows do not preserve action "
            "parameters, so this is a mechanism diagnostic rather than actual-action ERS"
        ),
        "complete_conflict_dose_seeds": [seed for seed, _paths in complete_conflict_seeds],
        "partial_conflict_dose_seeds_excluded": [
            int(seed)
            for seed in cfg["seeds"]["train"]
            if seed not in {complete_seed for complete_seed, _paths in complete_conflict_seeds}
        ],
        "aggregate_by_dose": aggregate_by_dose,
        "best_regret_dose": best_regret_dose,
        "rows": projected,
    }
    write_json(repo_path(cfg["output_dir"]) / "ers_regret.json", result)
    return result


def _cot_filter(cfg: dict[str, Any], config_path: str) -> dict[str, Any]:
    filtered_card = (
        repo_path(cfg["output_dir"]) / "datasets" / "cot_filter" / "truth_filtered_card.json"
    )
    src = repo_path(cfg["sources"]["hindsight_filtered"])
    kept_path = src / "samples.jsonl"
    dropped_path = src / "dropped.jsonl"
    kept = (
        [json.loads(line) for line in kept_path.read_text().splitlines() if line]
        if kept_path.exists()
        else []
    )
    dropped = (
        [json.loads(line) for line in dropped_path.read_text().splitlines() if line]
        if dropped_path.exists()
        else []
    )
    judged = kept + [r for r in dropped if r.get("annotation") is not None]
    oracle_errors = [r for r in dropped if r.get("verdict", {}).get("reason") == "oracle_error"]
    filter_rejects = [r for r in dropped if r.get("annotation") is not None]

    def row(rec: dict[str, Any]) -> dict[str, Any]:
        ann = rec.get("annotation") or {}
        truth = (rec.get("ground_truth") or {}).get("category")
        return {
            "truth": truth,
            "thought": ann.get("thought"),
            "attr_ok": ann.get("attribution") == truth,
        }

    unfiltered_grounding = grounding_summary([row(r) for r in judged]) if judged else {}
    filtered_grounding = grounding_summary([row(r) for r in kept]) if kept else {}
    result = {
        **artifact_meta(
            config_path,
            sources={
                "hindsight_filtered": cfg["sources"]["hindsight_filtered"],
                "hindsight_kept": str(kept_path),
                "hindsight_dropped": str(dropped_path),
            },
        ),
        "stage": "cot_filter",
        "truth_filtered_available": filtered_card.exists() and bool(kept),
        "unfiltered_api_oracle_available": bool(judged),
        "status": "available" if judged and kept else "blocked_for_comparison",
        "finding": bool(judged and kept),
        "n_kept_truth_filtered": len(kept),
        "n_filter_rejected_with_annotation": len(filter_rejects),
        "n_oracle_errors_excluded": len(oracle_errors),
        "filter_keep_rate": round(len(kept) / max(1, len(judged)), 3),
        "unfiltered_grounding": unfiltered_grounding,
        "truth_filtered_grounding": filtered_grounding,
        "grounded_correct_gain": round(
            float(filtered_grounding.get("grounded_correct_rate", 0.0))
            - float(unfiltered_grounding.get("grounded_correct_rate", 0.0)),
            3,
        ),
        "cards": {
            "truth_filtered": str(filtered_card),
            "unfiltered_api_oracle_source": str(dropped_path),
        },
        "scope_note": "Uses the real ApiOracle kept+dropped Hindsight stream already on disk; oracle transport errors are excluded from filter-effect denominators.",
    }
    write_json(repo_path(cfg["output_dir"]) / "cot_filter_summary.json", result)
    return result


def _encoder(cfg: dict[str, Any], config_path: str, *, run_grid: bool = False) -> dict[str, Any]:
    a5 = load_json(cfg["sources"]["a5_ood_theta"])
    grid_path = repo_path(cfg["output_dir"]) / "encoder_grid" / "summary.json"
    grid = None
    if run_grid or not grid_path.exists():
        try:
            from a7_encoder_grid import run_grid as run_encoder_grid
        except ModuleNotFoundError:  # pragma: no cover - supports `python -m scripts.a7_eval`
            from scripts.a7_encoder_grid import run_grid as run_encoder_grid

        grid = run_encoder_grid(config_path)
    elif grid_path.exists():
        grid = load_json(grid_path)

    grid_status = (grid or {}).get("status", "missing")
    result = {
        **artifact_meta(
            config_path,
            sources={
                "a5_ood_theta": cfg["sources"]["a5_ood_theta"],
                "encoder_grid": str(grid_path.parent),
            },
        ),
        "stage": "encoder",
        "status": "available" if grid_status == "available" else "partial_available",
        "privileged_projector_residual": {
            "adapter": a5.get("adapter"),
            "mean_residual_per_target": a5.get("mean_residual_per_target"),
            "abstention_utility": a5.get("abstention_utility", {}).get("B5-conflict-bi"),
        },
        "grid_status": grid_status,
        "grid_summary": str(grid_path),
        "grid": {
            "seeds": (grid or {}).get("seeds"),
            "n_samples": (grid or {}).get("n_samples"),
            "labels": (grid or {}).get("labels"),
            "aggregate": (grid or {}).get("aggregate"),
            "train_protocol": (grid or {}).get("train_protocol"),
        },
        "scope_note": "Full declared encoder grid is run on real frozen binding windows when encoder_grid/summary.json is present; A5 projector residual remains the deployed latent θ-head add-on.",
    }
    write_json(repo_path(cfg["output_dir"]) / "encoder_summary.json", result)
    return result


def _load_a3_rows(agent: str = "B5-conflict-bi") -> list[dict[str, Any]]:
    stem = agent.replace("-", "_")
    return load_json(f"outputs/eval/a3/per_item_{stem}.json")


def _a7_abs_points(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    a4 = load_json(cfg["sources"]["a4_results"])
    a5 = load_json(cfg["sources"]["a5_ood_theta"])
    rows = _load_a3_rows("B5-conflict-bi")
    residuals = _compute_projector_residuals(a5["adapter"], cfg["encoder"]["corpus_dirs"])
    points: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row["sid"])
        scenario = a3_row_to_a4_scenario(row)
        if (
            scenario is None
            or scenario not in a4["M_full"]
            or sid not in residuals
            or not row.get("parsed")
        ):
            continue
        cells = a4["M_full"][scenario]
        primitive = row.get("primitive")
        if primitive == "Switch_Gait" and not row.get("action_params_observed", False):
            raise ValueError(f"legacy A3 row {sid} lost Switch_Gait mode; rerun scripts.eval_a3")
        pred_label = primitive_to_label(
            primitive or "continue",
            row.get("primitive_params") or {},
            strict_params=True,
        )
        appearance_id = str(row.get("appearance_id", ""))
        appearance_split = str(row.get("appearance_split", ""))
        if not appearance_id or appearance_split not in {"train", "test"}:
            raise ValueError(f"A3 row {sid} lacks frozen appearance split metadata")
        points.append(
            {
                "sample_id": sid,
                "scenario": scenario,
                "truth": row.get("truth"),
                "cell": row.get("cell"),
                "appearance_id": appearance_id,
                "appearance_split": appearance_split,
                "cluster_id": f"{scenario}|{appearance_id}",
                "attr_ok": bool(row.get("attr_ok")),
                "cost_agent": float(cells.get(pred_label, {}).get("mean_cost", 4.0)),
                "cost_safe": float(cells.get("backstep_detour", {}).get("mean_cost", 4.0)),
                "cost_continue": float(cells.get("continue", {}).get("mean_cost", 4.0)),
                "ood_theta_residual": float(residuals[sid]),
                "pred_primitive": row.get("primitive"),
                "pred_cost_label": pred_label,
                "agent_label": pred_label,
                "safe_label": "backstep_detour",
                "continue_label": "continue",
            }
        )
    return points


def _a4_episode_costs(
    path: str | Path = "outputs/eval/a4/matrix.jsonl",
) -> dict[tuple[str, str], list[float]]:
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for line in repo_path(path).read_text().splitlines():
        if not line:
            continue
        outcome = Outcome(**json.loads(line))
        cells[(outcome.scenario, outcome.label)].append(float(physical_cost(outcome)))
    return dict(cells)


def _resample_a7_clusters(rows: list[dict[str, Any]], rng) -> list[dict[str, Any]]:
    """Resample appearance clusters within each physical scenario."""
    by_scenario: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_scenario[str(row["scenario"])][str(row["cluster_id"])].append(row)
    sampled: list[dict[str, Any]] = []
    for clusters in by_scenario.values():
        keys = list(clusters)
        for index in rng.integers(0, len(keys), size=len(keys)):
            sampled.extend(clusters[keys[int(index)]])
    return sampled


def _a7_policy_cost(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    tau: float,
    policy: str,
    cost_map: dict[tuple[str, str], float],
) -> tuple[float, float]:
    costs: list[float] = []
    covered = 0
    for row in rows:
        if policy == "selective":
            keep = float(row[score_field]) <= float(tau)
            label = row["agent_label"] if keep else row["safe_label"]
            covered += int(keep)
        elif policy == "agent":
            label = row["agent_label"]
        elif policy == "safe":
            label = row["safe_label"]
        elif policy == "continue":
            label = row["continue_label"]
        else:
            raise ValueError(policy)
        costs.append(float(cost_map[(row["scenario"], label)]))
    return sum(costs) / len(costs), covered / len(rows)


def _a7_two_stage_bootstrap(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    tau: float,
    episode_costs: dict[tuple[str, str], list[float]],
    reps: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap test appearance clusters and A4 physical episodes jointly."""
    import numpy as np

    rng = np.random.default_rng(seed)
    policies = ("selective", "agent", "safe", "continue")
    draws: dict[str, list[float]] = {name: [] for name in policies}
    coverages: list[float] = []
    for _ in range(reps):
        sampled_rows = _resample_a7_clusters(rows, rng)
        sampled_costs = {
            key: float(np.mean(rng.choice(values, size=len(values), replace=True)))
            for key, values in episode_costs.items()
        }
        for policy in policies:
            cost, coverage = _a7_policy_cost(
                sampled_rows,
                score_field=score_field,
                tau=tau,
                policy=policy,
                cost_map=sampled_costs,
            )
            draws[policy].append(cost)
            if policy == "selective":
                coverages.append(coverage)

    def ci(values: list[float]) -> list[float]:
        return [
            round(float(np.quantile(values, 0.025)), 4),
            round(float(np.quantile(values, 0.975)), 4),
        ]

    deltas = {
        f"selective_minus_{name}": [
            selective - baseline
            for selective, baseline in zip(draws["selective"], draws[name], strict=True)
        ]
        for name in ("agent", "safe", "continue")
    }
    delta_ci = {name: ci(values) for name, values in deltas.items()}
    return {
        "reps": int(reps),
        "unit": "appearance cluster within scenario + A4 episode within (scenario,label)",
        "cost_ci": {name: ci(values) for name, values in draws.items()},
        "coverage_ci": ci(coverages),
        "paired_delta_ci": delta_ci,
        "strictly_better_with_95ci": {
            name.removeprefix("selective_minus_"): bounds[1] < 0.0
            for name, bounds in delta_ci.items()
        },
    }


def _compute_projector_residuals(adapter: str, corpus_dirs: list[str]) -> dict[str, float]:
    import numpy as np
    import torch

    from kino_vla.vla.projector import KinoProjector

    pj = KinoProjector(
        vlm_dim=2560,
        conv_channels=(48, 48),
        conv_kernels=(5, 3),
        latent_dim=96,
        n_latents=6,
        n_heads=4,
        mlp_hidden=96,
        proj_hidden=512,
    )
    pj.load_state_dict(torch.load(repo_path(adapter) / "kino_projector.pt", map_location="cpu"))
    pj.eval()
    out: dict[str, float] = {}
    with torch.no_grad():
        for d in corpus_dirs:
            root = repo_path(d)
            samples_path = root / "samples.jsonl"
            frames_path = root / "frames.npz"
            if not samples_path.exists() or not frames_path.exists():
                continue
            npz = np.load(frames_path)
            for line in samples_path.read_text().splitlines():
                if not line:
                    continue
                rec = json.loads(line)
                sid = rec["sample_id"]
                theta = rec.get("target_theta")
                key = f"{sid}__proprio"
                if key not in npz or theta is None or len(theta) != 4:
                    continue
                w = torch.from_numpy(npz[key].astype("float32")).unsqueeze(0)
                _soft, pred = pj(w)
                out[sid] = float(
                    np.linalg.norm(pred.squeeze(0).numpy() - np.asarray(theta, dtype="float32"))
                )
    return out


def _aggregate_cached_a2_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from kino_vla.eval.a2_headline import ItemResult, aggregate

    items = [
        ItemResult(
            sample_id=str(r.get("sample_id")),
            operator=str(r.get("operator")),
            appearance_id=str(r.get("appearance_id", "")),
            appearance_split=str(r.get("appearance_split", "train")),
            truth_category=str(r.get("truth_category", "")),
            parsed=bool(r.get("parsed")),
            attribution=r.get("attribution"),
            attr_ok=bool(r.get("attr_ok")),
            primitive=r.get("primitive"),
            prim_feasible=bool(r.get("prim_feasible")),
            joint_ok=bool(r.get("joint_ok")),
        )
        for r in rows
    ]
    return aggregate(items)


def _sample_id_index(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for d in cfg["encoder"]["corpus_dirs"]:
        root = repo_path(d)
        frames_path = root / "frames.npz"
        if not (root / "samples.jsonl").exists() or not frames_path.exists():
            continue
        import numpy as np

        npz = np.load(frames_path)
        for line in (root / "samples.jsonl").read_text().splitlines():
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            if f"{sid}__rgb" in npz and f"{sid}__proprio" in npz:
                out[sid] = {"record": rec, "root": root, "npz": npz}
    return out


def _target_for_attr(category: str) -> str:
    from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive
    from kino_vla.vla.prompt import format_target

    primitive = ATTR_TO_PRIMITIVE.get(category, "Continue")
    if primitive == "Continue":
        primitive = "Set_Constraint"
        params = {"max_speed": 0.8, "stiffness": 1.0}
        category = "nominal"
    else:
        params = PRIMITIVE_PARAMS[primitive]
    ann = CoTAnnotation(
        thought="Attribution candidate.",
        attribution=category,
        primitive=RecoveryPrimitive(primitive, params),
        attribution_raw=category,
        raw_text="",
    )
    return format_target(ann)


def _posterior_cache_complete(
    cache: dict[str, dict[str, float]], points: list[dict[str, Any]]
) -> bool:
    return all(str(pt["sample_id"]) in cache for pt in points)


def _posterior_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_meta.json")


def _posterior_source_paths(cache_paths: dict[str, str | Path]) -> dict[str, str]:
    """Return all posterior caches and verifier sidecars used by an A7.1 run."""
    sources: dict[str, str] = {}
    for cache in cache_paths.values():
        cache_path = Path(cache)
        name = cache_path.stem
        sources[name] = str(cache_path)
        sources[f"{name}_meta"] = str(_posterior_meta_path(cache_path))
    return sources


def _posterior_cache_has_verified_meta(cache_path: Path, *, use_dropout: bool = False) -> bool:
    if not cache_path.exists():
        return False
    meta_path = _posterior_meta_path(cache_path)
    if not meta_path.exists():
        return False
    meta = load_json(meta_path)
    return (
        meta.get("method") == "direct_completion_logprob"
        and bool(meta.get("length_normalized"))
        and bool(meta.get("use_dropout", False)) == bool(use_dropout)
    )


def _posterior_cache_is_verified(
    cache_path: Path, points: list[dict[str, Any]], *, use_dropout: bool = False
) -> bool:
    if not _posterior_cache_has_verified_meta(cache_path, use_dropout=use_dropout):
        return False
    cache = load_json(cache_path)
    return _posterior_cache_complete(cache, points)


def _score_completion_posteriors(
    cfg: dict[str, Any],
    points: list[dict[str, Any]],
    *,
    adapter: str | None = None,
    limit: int | None = None,
    cache_path: Path | None = None,
    use_dropout: bool = False,
) -> dict[str, dict[str, float]]:
    import math

    import torch

    from kino_vla.utils.config import load_config
    from kino_vla.vla.dataset_build import _snapshot_from_record
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.prompt import build_messages, context_from_snapshot

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pcfg = load_config("data/hindsight.yaml")
    vcfg = load_config("vla/sft.yaml", {"route": "latent", "data.proprio_detail": "binned"})
    model = KinoVLA.from_pretrained(
        vcfg,
        device=device,
        adapter_dir=adapter or cfg["text_schemas"]["latent"]["adapter"],
    )
    if use_dropout:
        if cache_path is not None:
            seed = sum(ord(ch) for ch in cache_path.stem)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        model.train()
    else:
        model.eval()
    index = _sample_id_index(cfg)
    categories = [
        "adhesion",
        "compliant_terrain",
        "low_friction",
        "invisible_obstacle",
        "overload",
        "effort_decay",
        "nominal",
    ]
    targets = {cat: _target_for_attr(cat) for cat in categories}
    out: dict[str, dict[str, float]] = {}
    if cache_path is not None and _posterior_cache_has_verified_meta(
        cache_path, use_dropout=use_dropout
    ):
        out.update(load_json(cache_path))
    selected = points if limit is None else points[: int(limit)]
    try:
        for idx, pt in enumerate(selected, start=1):
            if pt["sample_id"] in out:
                continue
            sid = pt["sample_id"]
            meta = index.get(sid)
            if meta is None:
                continue
            rec = meta["record"]
            npz = meta["npz"]
            frames = {k: npz[f"{sid}__{k}"] for k in ("rgb", "depth", "proprio")}
            snap = _snapshot_from_record(rec, frames)
            ctx = context_from_snapshot(snap, route="latent", proprio_detail="binned")
            messages = build_messages(
                ctx, pcfg, route="latent", n_images=int(vcfg.data.get("n_images", 1))
            )
            images = list(snap.rgb[-int(vcfg.data.get("n_images", 1)) :]) if snap.rgb.size else []
            vals: dict[str, float] = {}
            with torch.no_grad():
                for cat, target in targets.items():
                    x = model.build_inputs(
                        messages,
                        images,
                        target_text=target,
                        proprio_window=snap.proprio_window,
                        target_theta=rec.get("target_theta"),
                        loss_span="action",
                    )
                    labels = x.labels
                    n_tokens = (
                        int((labels[:, 1:] != -100).sum().item()) if labels is not None else 1
                    )
                    vals[cat] = float(model.completion_logprob(x).detach().cpu()) / max(1, n_tokens)
            mx = max(vals.values())
            exps = {k: math.exp(v - mx) for k, v in vals.items()}
            denom = sum(exps.values())
            probs = {k: v / denom for k, v in exps.items()}
            out[sid] = probs
            if cache_path is not None:
                write_json(cache_path, out)
                write_json(
                    _posterior_meta_path(cache_path),
                    {
                        "method": "direct_completion_logprob",
                        "length_normalized": True,
                        "use_dropout": bool(use_dropout),
                        "adapter": adapter or cfg["text_schemas"]["latent"]["adapter"],
                        "n_cached": len(out),
                    },
                )
            if idx % 25 == 0:
                print(f"[a7.1] direct posterior scored {idx}/{len(selected)}", flush=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def _score_from_posteriors(
    points: list[dict[str, Any]], posteriors: dict[str, dict[str, float]]
) -> None:
    for pt in points:
        probs = posteriors.get(pt["sample_id"])
        if not probs:
            continue
        values = list(probs.values())
        pred = max(probs, key=probs.get)
        pmax = float(probs[pred])
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in values) / math.log(len(values))
        pt["msp_uncertainty"] = 1.0 - pmax
        pt["entropy"] = entropy
        pt["posterior_argmax"] = pred
        pt["posterior_confidence"] = pmax
        pt["posterior_correct"] = pred == pt.get("truth")


def _attach_posterior_variance(
    points: list[dict[str, Any]],
    posteriors: list[dict[str, dict[str, float]]],
    *,
    field: str,
) -> bool:
    """Attach mean per-category predictive variance when ≥2 real posterior caches exist."""
    if len(posteriors) < 2:
        return False
    for pt in points:
        pt[field] = posterior_mean_variance(posteriors, pt["sample_id"])
    return True


def _ensemble_variance(
    points: list[dict[str, Any]], posteriors: list[dict[str, dict[str, float]]]
) -> bool:
    """Attach the A7.1 ensemble-variance abstention score when multiple caches exist."""
    return _attach_posterior_variance(points, posteriors, field="ensemble_variance")


def _write_abstention_figures(
    out_dir: Path, curves: dict[str, list[dict[str, Any]]], reliability: dict[str, Any]
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    palette = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (name, curve) in enumerate(curves.items()):
        xs = [p["coverage"] for p in curve]
        ys = [p["expected_cost"] for p in curve]
        ax.plot(xs, ys, marker="o", ms=4, lw=2, color=palette[i % len(palette)], label=name)
    ax.set_xlabel("coverage (agent acts; high-score rows abstain)")
    ax.set_ylabel("expected physical cost via A4 M")
    ax.set_title("A7.1 abstention baselines on frozen A3/A4 artifacts")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=160)
    plt.close(fig)

    bins = reliability.get("bins", [])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], color="#898781", lw=1.5, ls="--", label="ideal")
    ax.plot(
        [b["confidence"] for b in bins if b["n"]],
        [b["accuracy"] for b in bins if b["n"]],
        marker="o",
        lw=2,
        color="#2a78d6",
        label="B5 posterior",
    )
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability (ECE={reliability.get('ece', 'n/a')})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir.parent / "calibration" / "reliability.png", dpi=160)
    plt.close(fig)


def _abstention_baselines(
    cfg: dict[str, Any], config_path: str, *, evaluate: bool = False
) -> dict[str, Any]:
    out_dir = repo_path(cfg["output_dir"]) / "abstention_baselines"
    cal_dir = repo_path(cfg["output_dir"]) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)
    points = _a7_abs_points(cfg)
    posterior_path = out_dir / "posterior_scores.json"
    posterior_cache_paths: dict[str, Path] = {"posterior_scores": posterior_path}
    if _posterior_cache_is_verified(posterior_path, points):
        posteriors = load_json(posterior_path)
        posterior_method = "cached_completion_logprob"
    elif evaluate:
        posteriors = _score_completion_posteriors(cfg, points, cache_path=posterior_path)
        posterior_method = "direct_completion_logprob"
        write_json(posterior_path, posteriors)
    else:
        raise RuntimeError(
            f"A7.1 requires a verified direct completion-logprob posterior cache at {posterior_path}. "
            "Run scripts/a7_eval.py --stage abstention-baselines --evaluate to create it."
        )
    _score_from_posteriors(points, posteriors)
    seed_posteriors = [posteriors]
    ensemble_adapters = {
        "seed1": "outputs/eval/a3/b5_conflict_bi_s1/adapter_best",
        "seed2": "outputs/eval/a3/b5_conflict_bi_s2/adapter_best",
    }
    for seed_name, adapter in ensemble_adapters.items():
        seed_path = out_dir / f"posterior_scores_{seed_name}.json"
        if _posterior_cache_is_verified(seed_path, points):
            seed_posteriors.append(load_json(seed_path))
            posterior_cache_paths[f"posterior_scores_{seed_name}"] = seed_path
        elif evaluate and repo_path(adapter).exists():
            seed_posteriors.append(
                _score_completion_posteriors(
                    cfg,
                    points,
                    adapter=adapter,
                    cache_path=seed_path,
                )
            )
            posterior_cache_paths[f"posterior_scores_{seed_name}"] = seed_path
    has_ensemble = _ensemble_variance(points, seed_posteriors)

    score_fields = {
        "ood_theta_residual": "OOD-θ residual",
        "msp_uncertainty": "MSP uncertainty",
        "entropy": "predictive entropy",
    }
    if has_ensemble:
        score_fields["ensemble_variance"] = "3-seed ensemble variance"
    curves = {
        name: risk_coverage_curve(points, score_field=field) for field, name in score_fields.items()
    }
    aurc = {
        name: {
            "aurc": risk_coverage_auc(curve),
            "best": min(curve, key=lambda p: (float(p["expected_cost"]), -float(p["coverage"]))),
            "error_auroc": round(
                auroc(
                    [float(p[field]) for p in points], [0 if p["attr_ok"] else 1 for p in points]
                ),
                3,
            ),
        }
        for field, name in score_fields.items()
        for curve in [curves[name]]
    }
    heldout = {
        field: heldout_threshold_selection(
            points,
            score_field=field,
            split_seed=int(cfg["seeds"]["split_seed"]),
            split_field="appearance_split",
        )
        for field in score_fields
    }
    episode_costs = _a4_episode_costs()
    heldout_test = [point for point in points if point["appearance_split"] == "test"]
    for index, field in enumerate(score_fields):
        heldout[field]["two_stage_bootstrap"] = _a7_two_stage_bootstrap(
            heldout_test,
            score_field=field,
            tau=float(heldout[field]["tau"]),
            episode_costs=episode_costs,
            reps=2000,
            seed=int(cfg["seeds"]["split_seed"]) + index,
        )

    calibration_points = [point for point in points if point["appearance_split"] == "train"]
    risk_budget = sum(float(point["cost_safe"]) for point in calibration_points) / len(
        calibration_points
    )
    conformal = {
        field: conformal_risk_control(
            calibration_points,
            score_field=field,
            risk_budget=risk_budget,
            alpha=0.1,
            cost_range=6.0,
        )
        for field in score_fields
    }
    reliability = expected_calibration_error(
        [
            {
                "confidence": p.get("posterior_confidence", 0.0),
                "correct": p.get("posterior_correct", False),
            }
            for p in points
            if "posterior_confidence" in p
        ],
        n_bins=10,
    )
    write_json(out_dir / "per_item_scores.json", points)
    write_json(
        out_dir / "risk_coverage.json",
        {"curves": curves, "n": len(points), "posterior_method": posterior_method},
    )
    write_json(out_dir / "aurc.json", aurc)
    write_json(out_dir / "heldout_threshold.json", heldout)
    write_json(out_dir / "conformal_bound.json", conformal)
    write_json(cal_dir / "reliability.json", reliability)
    write_json(
        cal_dir / "ece.json",
        {"ece": reliability["ece"], "n": reliability["n"], "posterior_method": posterior_method},
    )
    _write_abstention_figures(out_dir, curves, reliability)
    result = {
        **artifact_meta(
            config_path,
            sources={
                "a3_battery": cfg["sources"]["a3_battery"],
                "a4_results": cfg["sources"]["a4_results"],
                "a5_ood_theta": cfg["sources"]["a5_ood_theta"],
                **_posterior_source_paths(posterior_cache_paths),
            },
        ),
        "stage": "abstention_baselines",
        "status": "available",
        "n_points": len(points),
        "posterior_method": posterior_method,
        "score_fields": score_fields,
        "best_aurc": min(aurc.items(), key=lambda kv: kv[1]["aurc"]),
        "risk_budget": risk_budget,
        "risk_budget_definition": "A7 frozen-calibration always-safe cost",
    }
    write_json(out_dir / "summary.json", result)
    return result


def _conflict_dose(
    cfg: dict[str, Any], config_path: str, *, evaluate: bool, seeds_filter: set[int] | None = None
) -> dict[str, Any]:
    out_dir = repo_path(cfg["output_dir"]) / "conflict_dose"
    out_dir.mkdir(parents=True, exist_ok=True)
    dose_values = [int(x) for x in cfg["conflict_doses"]["matched_samples"]]
    seeds = [int(x) for x in cfg["seeds"]["train"]]
    if seeds_filter is not None:
        seeds = [s for s in seeds if s in seeds_filter]
    summary: dict[str, Any] = {
        **artifact_meta(config_path, sources={"a0_corpus": cfg["sources"]["a0_corpus"]}),
        "stage": "conflict_dose",
        "dose_values": dose_values,
        "seeds": seeds,
        "reuse_policy": "cached per-item artifacts are used before any VLA inference; seed0 0/5/10/20/40 is not rerun",
        "rows": {},
    }
    for seed in seeds:
        dose_rows: dict[str, list[dict[str, Any]]] = {}
        for dose in dose_values:
            adapter = (
                repo_path(cfg["output_dir"])
                / "adapters"
                / "conflict_dose"
                / f"seed{seed}"
                / f"dose_{dose:02d}"
                / "adapter_best"
            )
            per_item = out_dir / f"per_item_seed{seed}_dose{dose:02d}.json"
            key = f"seed{seed}_dose{dose:02d}"
            if per_item.exists():
                rows = load_json(per_item)
                summary["rows"][key] = {
                    "status": "available_cached",
                    "adapter": str(adapter),
                    "per_item": str(per_item),
                    "summary": _aggregate_cached_a2_rows(rows),
                }
                dose_rows[str(dose)] = [r for r in rows if r["operator"] == "O4_tether"]
                continue
            if not adapter.exists():
                summary["rows"][key] = {
                    "status": "requires_run",
                    "adapter": str(adapter),
                    "finding": False,
                }
                continue
            if evaluate:
                rows, agg = eval_a2_adapter(
                    str(adapter), "latent", "binned", cfg["sources"]["a0_corpus"]
                )
                write_json(per_item, rows)
                summary["rows"][key] = {
                    "status": "available",
                    "adapter": str(adapter),
                    "per_item": str(per_item),
                    "summary": agg,
                }
                dose_rows[str(dose)] = [r for r in rows if r["operator"] == "O4_tether"]
            else:
                summary["rows"][key] = {
                    "status": "adapter_available_needs_eval",
                    "adapter": str(adapter),
                    "finding": False,
                }
        if dose_rows:
            summary.setdefault("curves", {})[f"seed{seed}"] = dose_curve_summary(
                dose_rows, "attr_ok"
            )
    status_values = [r["status"] for r in summary["rows"].values()]
    complete_statuses = {"available", "available_cached"}
    summary["status_counts"] = status_counts(summary["rows"])
    if summary.get("curves"):
        summary["aggregate"] = aggregate_dose_curves(summary["curves"])
    summary["status"] = (
        "available"
        if status_values and all(s in complete_statuses for s in status_values)
        else "partial_available"
    )
    write_json(out_dir / "summary.json", summary)
    return summary


def _text_schema_eval(cfg: dict[str, Any], config_path: str, *, evaluate: bool) -> dict[str, Any]:
    out_dir = repo_path(cfg["output_dir"]) / "text_schema"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        **artifact_meta(config_path, sources={"a0_corpus": cfg["sources"]["a0_corpus"]}),
        "stage": "text_schema_eval",
        "rows": {},
    }
    for name, spec in cfg["text_schemas"].items():
        adapter = spec.get("adapter")
        adapter_exists = adapter is None or repo_path(adapter).exists()
        per_item = out_dir / f"per_item_{name}.json"
        if per_item.exists():
            rows = load_json(per_item)
            summary["rows"][name] = {
                "status": "available_cached",
                "adapter": adapter,
                "per_item": str(per_item),
                "summary": _aggregate_cached_a2_rows(rows),
            }
            continue
        if not adapter_exists:
            summary["rows"][name] = {"status": "requires_run", "adapter": adapter, "finding": False}
            continue
        if evaluate:
            rows, agg = eval_a2_adapter(
                adapter, spec["route"], spec["proprio_detail"], cfg["sources"]["a0_corpus"]
            )
            write_json(per_item, rows)
            summary["rows"][name] = {
                "status": "available",
                "adapter": adapter,
                "per_item": str(per_item),
                "summary": agg,
            }
        else:
            summary["rows"][name] = {
                "status": "adapter_available_needs_eval",
                "adapter": adapter,
                "finding": False,
            }
    statuses = [r["status"] for r in summary["rows"].values()]
    summary["status_counts"] = status_counts(summary["rows"])
    summary["status"] = (
        "available"
        if statuses and all(s in {"available", "available_cached"} for s in statuses)
        else "partial_available"
    )
    write_json(out_dir / "summary.json", summary)
    return summary


def _test_time_grounding(
    cfg: dict[str, Any], config_path: str, *, evaluate: bool
) -> dict[str, Any]:
    """Design §A7: run the automatic rationale checker on *emitted* model thoughts at test time."""
    out_dir = repo_path(cfg["output_dir"]) / "test_time_grounding"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prefer the deployed bidirectional conflict model + best mean-dose seed0/1/2 adapters at dose 10.
    targets: list[dict[str, Any]] = [
        {
            "name": "text_schema:latent",
            "adapter": cfg["text_schemas"]["latent"]["adapter"],
            "route": cfg["text_schemas"]["latent"]["route"],
            "proprio_detail": cfg["text_schemas"]["latent"]["proprio_detail"],
            "corpus": "matched",
        },
        {
            "name": "text_schema:rich_stats",
            "adapter": cfg["text_schemas"]["rich_stats"]["adapter"],
            "route": cfg["text_schemas"]["rich_stats"]["route"],
            "proprio_detail": cfg["text_schemas"]["rich_stats"]["proprio_detail"],
            "corpus": "matched",
        },
    ]
    for seed in cfg["seeds"]["train"]:
        adapter = (
            repo_path(cfg["output_dir"])
            / "adapters"
            / "conflict_dose"
            / f"seed{seed}"
            / "dose_10"
            / "adapter_best"
        )
        if adapter.exists():
            targets.append(
                {
                    "name": f"conflict_dose:seed{seed}_dose10",
                    "adapter": str(adapter),
                    "route": "latent",
                    "proprio_detail": "binned",
                    "corpus": "matched",
                }
            )

    rows_out: dict[str, Any] = {}
    for spec in targets:
        name = spec["name"]
        per_item = out_dir / f"per_item_{name.replace(':', '__')}.json"
        adapter = spec["adapter"]
        adapter_ok = adapter is None or repo_path(str(adapter)).exists()
        if per_item.exists() and not evaluate:
            rows = load_json(per_item)
            g = grounding_summary(rows)
            rows_out[name] = {
                "status": "available_cached",
                "adapter": adapter,
                "per_item": str(per_item),
                "grounding": g,
                "n": len(rows),
                "attr_acc": round(sum(1 for r in rows if r.get("attr_ok")) / max(1, len(rows)), 3),
            }
            continue
        if not adapter_ok:
            rows_out[name] = {"status": "requires_run", "adapter": adapter, "finding": False}
            continue
        if not evaluate and not per_item.exists():
            rows_out[name] = {
                "status": "adapter_available_needs_eval",
                "adapter": adapter,
                "finding": False,
            }
            continue
        rows = eval_a2_adapter_with_thoughts(
            str(adapter) if adapter is not None else None,
            spec["route"],
            spec["proprio_detail"],
            cfg["sources"]["a0_corpus"],
        )
        write_json(per_item, rows)
        g = grounding_summary(rows)
        rows_out[name] = {
            "status": "available",
            "adapter": adapter,
            "per_item": str(per_item),
            "grounding": g,
            "n": len(rows),
            "attr_acc": round(sum(1 for r in rows if r.get("attr_ok")) / max(1, len(rows)), 3),
        }

    statuses = [r.get("status") for r in rows_out.values()]
    complete = {"available", "available_cached"}
    result = {
        **artifact_meta(
            config_path,
            sources={
                "a0_corpus": cfg["sources"]["a0_corpus"],
                "latent_adapter": cfg["text_schemas"]["latent"]["adapter"],
            },
        ),
        "stage": "test_time_grounding",
        "status": "available"
        if statuses and all(s in complete for s in statuses)
        else "partial_available",
        "checker": "kino_vla.eval.a7_ablation.rationale_grounding",
        "scope_note": (
            "Test-time emitted-rationale grounding on real Qwen3-VL-4B generations over frozen "
            "matched O4/O2 snapshots; uses the same automatic cue/contradiction checker as the "
            "training-stream ApiOracle filter comparison."
        ),
        "rows": rows_out,
        "status_counts": status_counts(rows_out),
    }
    # Headline: best grounded-correct among complete rows
    complete_rows = [
        r for r in rows_out.values() if r.get("status") in complete and r.get("grounding")
    ]
    if complete_rows:
        best = max(
            complete_rows, key=lambda r: float(r["grounding"].get("grounded_correct_rate", 0.0))
        )
        result["headline"] = {
            "best_grounded_correct_rate": best["grounding"].get("grounded_correct_rate"),
            "best_grounding_rate": best["grounding"].get("grounding_rate"),
            "n_agents_scored": len(complete_rows),
        }
    write_json(out_dir / "summary.json", result)
    return result


def _text_schema_taxonomy(
    cfg: dict[str, Any], config_path: str, *, evaluate: bool
) -> dict[str, Any]:
    """Paired latent vs REFLECT vs rich taxonomy slices (design T4 falsifier)."""
    out_dir = repo_path(cfg["output_dir"]) / "text_schema_taxonomy"
    out_dir.mkdir(parents=True, exist_ok=True)
    a0 = cfg["sources"]["a0_corpus"]
    t3 = cfg["sources"]["a3_corpus_t3"]
    rows_out: dict[str, Any] = {}
    for name, spec in cfg["text_schemas"].items():
        adapter = spec.get("adapter")
        per_item = out_dir / f"per_item_{name}.json"
        heatmap_path = out_dir / f"heatmap_{name}.json"
        adapter_ok = adapter is None or repo_path(str(adapter)).exists()
        if per_item.exists() and heatmap_path.exists() and not evaluate:
            rows = load_json(per_item)
            # T5 is a decision task: the correct output is nominal/continue even though the
            # simulator retains a weak physical operator.  Upgrade cached predictions without
            # rerunning the VLA; no model output is changed.
            changed = False
            for row in rows:
                if row.get("cell") != "T5":
                    continue
                changed = changed or row.get("truth") != "nominal"
                row["truth"] = "nominal"
                row["truth_category"] = "nominal"
                row["attr_ok"] = bool(row.get("parsed") and row.get("attribution") == "nominal")
            ea = _load_eval_a3()
            heatmap = {cell: ea.cell_acc(rows, cell) for cell in ea.CELLS}
            if changed:
                write_json(per_item, rows)
            write_json(heatmap_path, heatmap)
            rows_out[name] = {
                "status": "available_cached",
                "adapter": adapter,
                "route": spec["route"],
                "proprio_detail": spec["proprio_detail"],
                "per_item": str(per_item),
                "heatmap": heatmap,
            }
            continue
        if not adapter_ok:
            rows_out[name] = {"status": "requires_run", "adapter": adapter, "finding": False}
            continue
        if not evaluate and not per_item.exists():
            rows_out[name] = {
                "status": "adapter_available_needs_eval",
                "adapter": adapter,
                "finding": False,
            }
            continue
        rows, heatmap = eval_a3_adapter(
            str(adapter) if adapter is not None else None,
            spec["route"],
            spec["proprio_detail"],
            a0,
            t3,
        )
        write_json(per_item, rows)
        write_json(heatmap_path, heatmap)
        rows_out[name] = {
            "status": "available",
            "adapter": adapter,
            "route": spec["route"],
            "proprio_detail": spec["proprio_detail"],
            "per_item": str(per_item),
            "heatmap": heatmap,
        }

    statuses = [r.get("status") for r in rows_out.values()]
    complete = {"available", "available_cached"}
    # T4 comparison among complete schemas
    t4: dict[str, Any] = {}
    for name, row in rows_out.items():
        if row.get("status") in complete:
            t4[name] = (row.get("heatmap") or {}).get("T4", {})
    latent_t4 = float((t4.get("latent") or {}).get("acc") or 0.0)
    rich_t4 = float((t4.get("rich_stats") or {}).get("acc") or 0.0)
    reflect_t4 = float((t4.get("reflect_scalar") or {}).get("acc") or 0.0)
    both_fail = bool(t4) and latent_t4 == 0.0 and rich_t4 == 0.0
    result = {
        **artifact_meta(
            config_path,
            sources={
                "a0_corpus": a0,
                "a3_corpus_t3": t3,
            },
        ),
        "stage": "text_schema_taxonomy",
        "status": "available"
        if statuses and all(s in complete for s in statuses)
        else "partial_available",
        "rows": rows_out,
        "status_counts": status_counts(rows_out),
        "t4_comparison": {
            "latent": t4.get("latent"),
            "reflect_scalar": t4.get("reflect_scalar"),
            "rich_stats": t4.get("rich_stats"),
            "rich_matches_or_beats_latent": bool(t4) and rich_t4 >= latent_t4,
            "both_fail_t4": both_fail,
            "latent_t4_acc": latent_t4,
            "rich_t4_acc": rich_t4,
            "reflect_t4_acc": reflect_t4,
            "interpretation": (
                "Both latent and rich conflict-route adapters score T4=0 on the merged corpus; "
                "T4 fine-structure is carried by B1/proprio (A3 method slices), not these conflict "
                "specialists. Design falsifier 'rich matches latent on T4' holds as a match-at-floor; "
                "latent claim narrows to bandwidth/θ/sampling/test-time grounding rather than T4 accuracy."
                if both_fail
                else (
                    "Rich matches or beats latent on T4; latent claim narrows to bandwidth/integration/θ."
                    if rich_t4 >= latent_t4
                    else "Latent retains a T4 edge over rich text on this paired eval."
                )
            ),
        },
        "scope_note": (
            "Paired latent vs REFLECT-scalar vs rich-stats taxonomy evaluation on the merged "
            "A0+A3 frozen corpus. If rich text matches latent on T4, the latent claim narrows to "
            "bandwidth/integration/θ (design §A7 / risk register)."
        ),
    }
    write_json(out_dir / "summary.json", result)
    return result


def run(
    config_path: str, stage: str, *, evaluate: bool, seeds_filter: set[int] | None = None
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    repo_path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if stage in ("all", "existing"):
        results["text_schema_existing"] = _m7_route_summary(cfg, config_path)
        results["a3_method_slices"] = _a3_method_summary(cfg, config_path)
        results["addons"] = _addons(cfg, config_path)
        results["cot_filter"] = _cot_filter(cfg, config_path)
        results["encoder"] = _encoder(cfg, config_path, run_grid=(stage == "all"))
        # Prefer cached test-time grounding / taxonomy when present; do not force VLA.
        ttg_path = repo_path(cfg["output_dir"]) / "test_time_grounding" / "summary.json"
        tax_path = repo_path(cfg["output_dir"]) / "text_schema_taxonomy" / "summary.json"
        if ttg_path.exists():
            results["test_time_grounding"] = load_json(ttg_path)
        if tax_path.exists():
            results["text_schema_taxonomy"] = load_json(tax_path)
    if stage in ("all", "abstention-baselines"):
        results["abstention_baselines"] = _abstention_baselines(cfg, config_path, evaluate=evaluate)
    if stage == "encoder-grid":
        results["encoder"] = _encoder(cfg, config_path, run_grid=True)
    if stage in ("all", "text-schema"):
        results["text_schema_eval"] = _text_schema_eval(cfg, config_path, evaluate=evaluate)
    if stage in ("all", "conflict-dose"):
        results["conflict_dose"] = _conflict_dose(
            cfg, config_path, evaluate=evaluate, seeds_filter=seeds_filter
        )
    if stage in ("test-time-grounding",):
        results["test_time_grounding"] = _test_time_grounding(cfg, config_path, evaluate=evaluate)
    if stage in ("text-schema-taxonomy",):
        results["text_schema_taxonomy"] = _text_schema_taxonomy(cfg, config_path, evaluate=evaluate)
    if stage in ("all", "existing"):
        results["ers_regret"] = _ers_regret(cfg, config_path)
    manifest = {
        **artifact_meta(
            config_path,
            sources={
                "config": config_path,
                "a0_corpus": cfg["sources"]["a0_corpus"],
                "a3_battery": cfg["sources"]["a3_battery"],
                "a4_results": cfg["sources"]["a4_results"],
                "a5_ood_theta": cfg["sources"]["a5_ood_theta"],
                "a6_theta_sweep": cfg["sources"]["a6_theta_sweep"],
            },
        ),
        "stage": stage,
        "evaluate": evaluate,
        "seeds_filter": None if seeds_filter is None else sorted(seeds_filter),
        "results": {k: v.get("status") for k, v in results.items()},
    }
    write_json(repo_path(cfg["output_dir"]) / "a7_eval_manifest.json", manifest)
    return {"manifest": manifest, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate/aggregate A7 method-ablation artifacts")
    ap.add_argument("--config", default="configs/eval/a7.yaml")
    ap.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "existing",
            "text-schema",
            "conflict-dose",
            "encoder-grid",
            "abstention-baselines",
            "test-time-grounding",
            "text-schema-taxonomy",
        ],
    )
    ap.add_argument(
        "--evaluate", action="store_true", help="run VLA inference for available adapters"
    )
    ap.add_argument("--seeds", default=None, help="comma-separated seed subset, e.g. 0")
    args = ap.parse_args()
    seeds_filter = None
    if args.seeds:
        seeds_filter = {int(x) for x in args.seeds.split(",") if x.strip()}
    out = run(args.config, args.stage, evaluate=args.evaluate, seeds_filter=seeds_filter)
    print(json.dumps(out["manifest"], indent=2))


if __name__ == "__main__":
    main()
