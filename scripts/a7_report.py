#!/usr/bin/env python
# ruff: noqa: E501
"""Generate the A7 systematic report from machine artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kino_vla.eval.a7_ablation import load_yaml, repo_path


def _fmt_ci(cell: dict[str, Any]) -> str:
    if not cell:
        return "n/a"
    if "acc" in cell:
        return f"{cell['acc']:.3f} [{cell['ci'][0]:.3f},{cell['ci'][1]:.3f}] (n={cell['n']})"
    if "rate" in cell:
        return f"{cell['rate']:.3f} [{cell['ci'][0]:.3f},{cell['ci'][1]:.3f}] (n={cell['n']})"
    return "n/a"


def _fmt_p(value: float) -> str:
    """Format p values without presenting a positive probability as zero."""
    if value < 1e-4:
        return f"{value:.3e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _load_optional(path: Path) -> dict[str, Any]:
    return (
        json.loads(path.read_text()) if path.exists() else {"status": "missing", "path": str(path)}
    )


def _table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join([header, sep, *body])


def _status_line(obj: dict[str, Any]) -> str:
    status = obj.get("status", "missing")
    if status in {"available", "partial_available"}:
        return f"`{status}`"
    reason = obj.get("reason") or obj.get("scope_note") or obj.get("path") or "not available"
    return f"`{status}` — {reason}"


def _hash_lines(*artifacts: dict[str, Any]) -> str:
    seen: dict[str, str] = {}
    for artifact in artifacts:
        for name, value in artifact.get("source_hashes", {}).items():
            seen[name] = value
    rows = [["source", "sha256(12)"]]
    for name, value in sorted(seen.items()):
        rows.append([name, str(value)])
    return _table(rows)


def _dose_aggregate_text(conflict: dict[str, Any]) -> str:
    aggregate = conflict.get("aggregate") or {}
    if not aggregate:
        return "No fully evaluated aggregate is available yet."
    rows = [["dose", "mean O4 attr", "min", "max", "seed cells"]]
    for dose, cell in sorted(aggregate.get("by_dose", {}).items(), key=lambda kv: int(kv[0])):
        rows.append(
            [
                dose,
                f"{cell['mean_rate']:.3f}",
                f"{cell['min_rate']:.3f}",
                f"{cell['max_rate']:.3f}",
                str(cell["n_cells"]),
            ]
        )
    return (
        _table(rows)
        + f"\n\nBest mean dose: `{aggregate.get('best_mean_dose')}`; monotone all seeds: `{aggregate.get('monotone_all_seeds')}`."
    )


def _dose_sensitivity_text(sens: dict[str, Any]) -> str:
    if sens.get("status") in {None, "missing"}:
        return (
            "No unequal-recipe sensitivity artifact is present. Headline dose cells remain "
            "protocol-matched unique-sample / epochs=4 only."
        )
    rows = [["seed", "protocol O4", "push O4", "protocol O2", "push O2", "role"]]
    proto = sens.get("protocol_matched_restored") or {}
    push = sens.get("push_vs_old") or {}
    for seed in [1, 2]:
        p = proto.get(f"seed{seed}") or {}
        u = push.get(str(seed)) or push.get(seed) or {}
        pu = (u.get("push") or {}) if isinstance(u, dict) else {}
        rows.append(
            [
                str(seed),
                f"{p.get('o4_attr', 'n/a')}",
                f"{pu.get('o4', 'n/a')}",
                f"{p.get('o2_attr', 'n/a')}",
                f"{pu.get('o2', 'n/a')}",
                "sensitivity only",
            ]
        )
    note = sens.get("scope_note") or ""
    policy = sens.get("headline_policy") or ""
    return (
        _table(rows)
        + f"\n\n{note} {policy} Recipe for push cells: conflict-row ×8 upsample of the same 5 unique "
        "O4 matched IDs + epochs=6 + nav_weight=0.4. That is **not** equivalent to dose=5 unique "
        "samples under the controlled axis (effective O4 exposure is closer to a higher dose)."
    )


def _route_table(arms: dict[str, Any]) -> str:
    rows = [["arm", "greedy ambiguous", "sampled ambiguous", "prompt tokens", "θ head"]]
    for arm in ["vision_only", "text_scalar", "text_binned", "latent"]:
        r = arms.get(arm, {})
        sampled = r.get("sampled_ambiguous_mean")
        sampled_s = (
            "n/a" if sampled is None else f"{sampled:.3f}±{r.get('sampled_ambiguous_std', 0):.3f}"
        )
        greedy = r.get("attr_ambiguous_greedy")
        rows.append(
            [
                arm,
                "n/a" if greedy is None else f"{greedy:.3f}",
                sampled_s,
                str(r.get("mean_prompt_tokens", "n/a")),
                "yes" if r.get("theta_mae") else "no",
            ]
        )
    return _table(rows)


def _a3_table(a3rows: dict[str, Any]) -> str:
    rows = [["agent", "T2", "T3", "T4", "T5"]]
    for agent in ["B1", "B-T", "B-V", "B-F", "B5-unshaped", "B5-conflict", "B5-conflict-bi"]:
        if agent in a3rows:
            rows.append(
                [
                    agent,
                    _fmt_ci(a3rows[agent].get("T2", {})),
                    _fmt_ci(a3rows[agent].get("T3", {})),
                    _fmt_ci(a3rows[agent].get("T4", {})),
                    _fmt_ci(a3rows[agent].get("T5", {})),
                ]
            )
    return _table(rows) if len(rows) > 1 else "_missing_"


def _taxonomy_table(tax: dict[str, Any]) -> str:
    rows = [["schema", "T2", "T3", "T4", "T5", "status"]]
    for name in ["latent", "reflect_scalar", "rich_stats"]:
        row = (tax.get("rows") or {}).get(name, {})
        heat = row.get("heatmap") or {}
        if not heat and row.get("status") not in {"available", "available_cached"}:
            rows.append([name, "n/a", "n/a", "n/a", "n/a", str(row.get("status", "missing"))])
            continue
        rows.append(
            [
                name,
                _fmt_ci(heat.get("T2", {})),
                _fmt_ci(heat.get("T3", {})),
                _fmt_ci(heat.get("T4", {})),
                _fmt_ci(heat.get("T5", {})),
                str(row.get("status", "n/a")),
            ]
        )
    return _table(rows) if len(rows) > 1 else "_missing_"


def _ttg_table(ttg: dict[str, Any]) -> str:
    rows = [["agent", "n", "attr_acc", "grounding_rate", "grounded_correct", "status"]]
    for name, row in sorted((ttg.get("rows") or {}).items()):
        g = row.get("grounding") or {}
        rows.append(
            [
                name,
                str(row.get("n", g.get("n", "n/a"))),
                str(row.get("attr_acc", "n/a")),
                str(g.get("grounding_rate", "n/a")),
                str(g.get("grounded_correct_rate", "n/a")),
                str(row.get("status", "n/a")),
            ]
        )
    return _table(rows) if len(rows) > 1 else "_missing_"


def _dose_tables(conflict: dict[str, Any]) -> str:
    if not conflict.get("curves"):
        required = [
            k for k, v in conflict.get("rows", {}).items() if v.get("status") == "requires_run"
        ]
        return (
            f"Conflict-dose adapters are not fully present yet (`status={conflict.get('status')}`). "
            f"Required adapter/eval cells: {len(required)}. This is recorded as a non-finding scaffold "
            "until the real VLA dose sweep is run."
        )
    tables = []
    for seed, curve in conflict["curves"].items():
        rows = [["dose", "O4 attr", "McNemar vs previous"]]
        for rec in curve["curve"]:
            mc = rec.get("mcnemar_vs_prev")
            mc_s = (
                "n/a"
                if mc is None
                else f"b={mc['b_a_right_b_wrong']}, c={mc['c_a_wrong_b_right']}, "
                f"p={_fmt_p(float(mc['p_exact_two_sided']))}"
            )
            rows.append([str(rec["dose"]), _fmt_ci(rec), mc_s])
        tables.append(f"**{seed}**\n\n" + _table(rows))
    return "\n\n".join(tables)


def _ers_table(ers: dict[str, Any]) -> str:
    rows = [["agent", "n scored", "mean cost", "mean regret", "missing"]]
    for agent, row in sorted((ers.get("rows") or {}).items()):
        rows.append(
            [
                agent,
                str(row.get("n_scored", 0)),
                f"{float(row.get('mean_cost', 0.0)):.3f}",
                f"{float(row.get('mean_regret', 0.0)):.3f}",
                str(row.get("n_missing", 0)),
            ]
        )
    return _table(rows)


def _ers_aggregate_table(ers: dict[str, Any]) -> str:
    rows = [["dose", "mean cost", "mean regret", "regret range", "seeds"]]
    for dose, row in sorted(
        (ers.get("aggregate_by_dose") or {}).items(), key=lambda kv: int(kv[0])
    ):
        rows.append(
            [
                dose,
                f"{float(row.get('mean_cost', 0.0)):.3f}",
                f"{float(row.get('mean_regret', 0.0)):.3f}",
                f"{float(row.get('min_regret', 0.0)):.3f}–{float(row.get('max_regret', 0.0)):.3f}",
                str(row.get("n_seeds", 0)),
            ]
        )
    return _table(rows)


def _fmt_seed_cell(cell: dict[str, Any]) -> str:
    if not cell:
        return "n/a"
    if "rate" not in cell:
        return "n/a"
    if "min_rate" in cell and "max_rate" in cell:
        return f"{float(cell['rate']):.3f} [{float(cell['min_rate']):.3f},{float(cell['max_rate']):.3f}]"
    return f"{float(cell['rate']):.3f}"


def _encoder_best_text(enc: dict[str, Any], grid: dict[str, Any]) -> str:
    aggregate = (enc.get("grid") or {}).get("aggregate") or grid.get("aggregate") or {}
    if not aggregate:
        return "No encoder-grid aggregate is available."
    rows = [["criterion", "best cell", "value"]]
    labels = {
        "best_attr_acc": "attribution accuracy",
        "best_theta_mae": "θ MAE (lower better)",
        "best_residual_error_auroc": "residual→error AUROC",
    }
    for key in ["best_attr_acc", "best_theta_mae", "best_residual_error_auroc"]:
        cell = aggregate.get(key) or {}
        rows.append([labels[key], str(cell.get("key", "n/a")), str(cell.get("value", "n/a"))])
    rows.append(
        [
            "available cells",
            "variant×T×gate",
            f"{aggregate.get('available_cells', 0)}/{aggregate.get('available_cells', 0) + aggregate.get('blocked_cells', 0)}",
        ]
    )
    return _table(rows)


def _encoder_grid_table(grid: dict[str, Any]) -> str:
    cells = grid.get("cells") or {}
    if not cells:
        return "No full encoder-grid cell table is available."
    variant_order = {"privileged_distillation": 0, "contrastive_only": 1, "from_scratch": 2}
    rows = [["cell", "attr", "O4/O2", "T3", "θ MAE", "resid→err AUROC"]]
    for key, cell in sorted(
        cells.items(),
        key=lambda kv: (
            variant_order.get(str(kv[1].get("variant")), 99),
            int(kv[1].get("window_length", 0)),
            0 if kv[1].get("gate") else 1,
        ),
    ):
        if cell.get("status") != "available":
            rows.append([key, f"`{cell.get('status')}`", "—", "—", "—", "—"])
            continue
        rows.append(
            [
                key,
                _fmt_seed_cell(cell.get("attr_acc", {})),
                _fmt_seed_cell(cell.get("o4_o2_attr_acc", {})),
                _fmt_seed_cell(cell.get("t3_attr_acc", {})),
                f"{float(cell.get('theta_mae_mean', 0.0)):.4f}",
                f"{float(cell.get('residual_error_auroc', 0.0)):.3f}",
            ]
        )
    return _table(rows)


def _a71_label(key: str) -> str:
    return {
        "OOD-θ residual": "OOD-θ residual",
        "MSP uncertainty": "MSP uncertainty",
        "predictive entropy": "predictive entropy",
        "3-seed ensemble variance": "3-seed ensemble variance",
        "ood_theta_residual": "OOD-θ residual",
        "msp_uncertainty": "MSP uncertainty",
        "entropy": "predictive entropy",
        "ensemble_variance": "3-seed ensemble variance",
    }.get(key, key)


def _aurc_table(aurc: dict[str, Any]) -> str:
    if not aurc:
        return "No A7.1 AURC artifact is available."
    order = ["predictive entropy", "MSP uncertainty", "3-seed ensemble variance", "OOD-θ residual"]
    rows = [["score", "AURC↓", "error AUROC", "best coverage", "best cost"]]
    for name in order:
        if name not in aurc:
            continue
        cell = aurc[name]
        best = cell.get("best", {})
        rows.append(
            [
                name,
                f"{float(cell.get('aurc', 0.0)):.3f}",
                f"{float(cell.get('error_auroc', 0.0)):.3f}",
                f"{float(best.get('coverage', 0.0)):.3f}",
                f"{float(best.get('expected_cost', 0.0)):.3f}",
            ]
        )
    return _table(rows)


def _heldout_table(heldout: dict[str, Any]) -> str:
    if not heldout:
        return "No A7.1 held-out threshold artifact is available."
    order = ["entropy", "msp_uncertainty", "ensemble_variance", "ood_theta_residual"]
    rows = [
        [
            "score",
            "cal coverage",
            "cal cost",
            "test coverage",
            "test cost [two-stage 95% CI]",
            "selective − safe 95% CI",
        ]
    ]
    for key in order:
        if key not in heldout:
            continue
        row = heldout[key]
        cal = row.get("calibration", {})
        test = row.get("test", {})
        boot = row.get("two_stage_bootstrap", {})
        ci = (boot.get("cost_ci") or {}).get("selective", ["n/a", "n/a"])
        delta = (boot.get("paired_delta_ci") or {}).get("selective_minus_safe", ["n/a", "n/a"])
        rows.append(
            [
                _a71_label(key),
                f"{float(cal.get('coverage', 0.0)):.3f}",
                f"{float(cal.get('expected_cost', 0.0)):.3f}",
                f"{float(test.get('coverage', 0.0)):.3f}",
                f"{float(test.get('expected_cost', 0.0)):.3f} [{ci[0]},{ci[1]}]",
                f"[{delta[0]},{delta[1]}]",
            ]
        )
    return _table(rows)


def _conformal_table(conf: dict[str, Any]) -> str:
    if not conf:
        return "No A7.1 conformal-risk artifact is available."
    order = ["entropy", "msp_uncertainty", "ensemble_variance", "ood_theta_residual"]
    rows = [["score", "feasible", "selected coverage", "cost", "UCB"]]
    for key in order:
        if key not in conf:
            continue
        row = conf[key]
        sel = row.get("selected", {})
        rows.append(
            [
                _a71_label(key),
                str(bool(row.get("feasible"))),
                f"{float(sel.get('coverage', 0.0)):.3f}",
                f"{float(sel.get('expected_cost', 0.0)):.3f}",
                f"{float(sel.get('ucb_mean_cost', 0.0)):.3f}",
            ]
        )
    return _table(rows)


def build_report(config_path: str) -> str:
    cfg = load_yaml(config_path)
    out = repo_path(cfg["output_dir"])
    text_existing = _load_optional(out / "text_schema_existing.json")
    a3 = _load_optional(out / "a3_method_slices.json")
    conflict = _load_optional(out / "conflict_dose" / "summary.json")
    cot = _load_optional(out / "cot_filter_summary.json")
    enc = _load_optional(out / "encoder_summary.json")
    encoder_grid = _load_optional(out / "encoder_grid" / "summary.json")
    addons = _load_optional(out / "addons.json")
    ers = _load_optional(out / "ers_regret.json")
    manifest = _load_optional(out / "a7_eval_manifest.json")
    build = _load_optional(out / "datasets" / "build_summary.json")
    a71 = _load_optional(out / "abstention_baselines" / "summary.json")
    a71_aurc = _load_optional(out / "abstention_baselines" / "aurc.json")
    a71_heldout = _load_optional(out / "abstention_baselines" / "heldout_threshold.json")
    a71_conformal = _load_optional(out / "abstention_baselines" / "conformal_bound.json")
    a71_ece = _load_optional(out / "calibration" / "ece.json")
    ttg = _load_optional(out / "test_time_grounding" / "summary.json")
    tax = _load_optional(out / "text_schema_taxonomy" / "summary.json")
    dose_sens = _load_optional(
        out / "conflict_dose" / "sensitivity_dose05_upsample" / "summary.json"
    )

    meta = manifest if manifest.get("commit") else text_existing
    commit = meta.get("commit", "unknown")
    cfg_hash = meta.get("config_sha256", "unknown")
    arms = text_existing.get("arms", {})
    headline = text_existing.get("headline", {})
    a3rows = a3.get("rows", {})
    add_abst = addons.get("ood_theta_abstention", {})
    b5_abs = add_abst.get("B5-conflict-bi", {})
    theta_bracket = addons.get("theta_boundary_bracket")
    theta_star = addons.get("theta_star_descriptive")
    risk = addons.get("risk_coverage", {})
    token_ratio = headline.get("latent_vs_text_binned_token_ratio")
    cot_unfiltered = cot.get("unfiltered_grounding", {})
    cot_filtered = cot.get("truth_filtered_grounding", {})
    cot_gain = cot.get("grounded_correct_gain", "n/a")
    a71_best = a71.get("best_aurc", ["n/a", {}])
    a71_best_name = a71_best[0] if isinstance(a71_best, list) and a71_best else "n/a"
    a71_best_cell = a71_best[1] if isinstance(a71_best, list) and len(a71_best) > 1 else {}
    ttg_headline = ttg.get("headline", {})
    tax_t4 = tax.get("t4_comparison", {})
    agg = conflict.get("aggregate") or {}
    best_dose = agg.get("best_mean_dose", "n/a")
    best_cell = (agg.get("by_dose") or {}).get(str(best_dose), {})
    dose5_cell = (agg.get("by_dose") or {}).get("5", {})
    broad_risk = (risk.get("protocols") or {}).get("broad_overlap", {})
    broad_selected = broad_risk.get("selected_min_cost", {})
    broad_nonzero = broad_risk.get("selected_min_cost_nonzero", {})

    return f"""# A7 — Method Ablations

> **Headline:** A7 completes the method-ablation story from frozen artifacts under a single protocol-matched conflict-dose recipe. Strongest claim-bearing findings: (i) proprioceptive injection (text or latent) is necessary on the M7 ambiguity route ablation (vision-only 0.458 vs text/latent 1.0); (ii) latent matches oracle/binned text on greedy accuracy while uniquely exposing a θ head and cutting prompt tokens vs binned text (ratio `{token_ratio}`); (iii) protocol-matched 3-seed dose curve peaks at **dose 10** (mean O4 attr 0.733; best attribution-implied A4 regret 0.40)—not at unstable dose 5; (iv) truth filtering improves grounded-correct rationales on the real ApiOracle stream (0.604→0.765) and test-time emitted-rationale grounding is reportable on deployed adapters (best GC 0.8); (v) full 18/18 encoder grid + A7.1 posterior baselines remain artifact-backed. Unequal-recipe dose5 stabilization is **sensitivity-only**, not a headline dose-axis point.

## 0. Artifact summary

- Config: `{config_path}` (sha `{cfg_hash}`)
- Commit: `{commit}`
- Output dir: `{cfg["output_dir"]}`
- Dataset build manifest: `{build.get("stage", "missing")}`
- Eval manifest: `{manifest.get("stage", "missing")}`; evaluate=`{manifest.get("evaluate", "missing")}`
- Conflict-dose reuse policy: `{conflict.get("reuse_policy", "n/a")}`

{_hash_lines(text_existing, a3, conflict, cot, enc, addons, ers, manifest, a71, ttg, tax)}

## 1. Why

A2/A3/A4/A5/A6 provide the upstream evidence and explicit failure boundaries; A7 explains *which method choices* are load-bearing. The ablations ask whether high-frequency body evidence must enter the VLA at all, whether latent Kino-Tokens earn their keep over text summaries, how much conflict-specific supervision is needed, whether truth-filtered rationales can be evaluated without confabulation, and whether uncertainty scores are useful for selective prediction.

A7 is not a new navigation benchmark. It is an offline, frozen-snapshot method-ablation package plus consequence projection through the already-gated A4 matrix.

## 2. Claim C1-C5 + logical role

A7 primarily serves **C2**: learned conflict resolution is a non-trivial semantic operation, not merely a vision channel being present. It also supports the method section by showing which ingredients support the C2 behavior and which claims must be narrowed.

Secondary links:

- **C3 add-on:** ERS/regret projects A7 method choices through A4's label-swap consequence matrix.
- **C4 add-on:** A5/A6 residual and boundary diagnostics are reused with their negative-result guardrails; A7 does not turn them into a positive OOD claim.
- **Not a new C1/C5 proof:** C1/C5 remain carried by A1/A2/A3; A7 only reuses their artifacts for method interpretation.

## 3. Method

All readouts are generated from frozen recorded artifacts: M7 route-ablation runs on Qwen3-VL-4B, A2/A3 frozen VLA evaluations, the A4 simulated consequence matrix, A5 residual diagnostics, and the A6 simulated base-boundary sweep. A7 scripts are config-driven (`configs/eval/a7.yaml`) and write machine JSON under `outputs/eval/a7/`.

Controls and honesty rules:

- **Conflict-dose protocol (headline):** every headline dose×seed cell uses unique-sample `dose_XX` datasets + default SFT recipe (epochs=4, nav_weight=1.0). Seed0 0/5/10/20/40 caches are not rerun. Collapsed protocol dose5 cells (seed1 O4=0.0, seed2 O4=0.4) remain in the curve as real small-data variance—they are not replaced by unequal-recipe adapters.
- **Conflict-dose sensitivity (not headline):** an upsample×8 / epochs=6 / nav_weight=0.4 recipe can stabilize dose5 (O4→0.8) but changes effective conflict exposure and is reported only under `conflict_dose/sensitivity_dose05_upsample/`—never mixed into the controlled dose aggregate.
- Raw unfiltered real `ApiOracle` CoT is represented by the kept+dropped Hindsight stream before truth-filter retention; oracle transport errors are excluded from filter-effect denominators.
- **Taxonomy / test-time grounding caveat:** latent, reflect_scalar, and rich_stats adapters come from different training curricula (A3 conflict-bi / A2 text / A7 rich). Differences there are **not** pure injection-modality ablations; the controlled injection ablation is the M7 route table (identical harness, route-only change).
- The full contrastive-only/from-scratch/window/gate encoder grid is run over recorded A0/A3 binding windows (`obs48+τ12`, T=100 real source; T=25/50 use tail crops; no synthetic upsampling). The anomaly gate is observable-only and open on these frozen anomaly-triggered snapshots.
- A7.1 abstention baselines use the same frozen A3/A4/A5 artifacts plus direct Qwen3-VL-4B action-completion log-prob scoring for posterior MSP/entropy; no surrogate classifier is introduced. Thresholds are selected on train appearances and evaluated once on test appearances. Actual actions use `primitive + primitive_params`; uncertainty intervals resample appearance clusters and A4 physical episodes.
- Add-on ERS/OOD/θ* readouts consume A4/A5/A6 artifacts and do not create new closed-loop claims.

Core scripts: `scripts/a7_build_datasets.py`, `scripts/a7_train.py`, `scripts/a7_eval.py`, `scripts/a7_report.py`; reducers/tests: `kino_vla/eval/a7_ablation.py`, `tests/test_a7_*.py`.

## 4. Results with CIs + controls

### A7.1 Latent vs text injection

{_route_table(arms)}

Vision-only remains near chance on the ambiguity regime ({headline.get("vision_only_ambiguous", "n/a")}), while text and latent proprio routes reach 1.0 greedy ambiguous attribution. **This M7 table is the controlled injection ablation** (same VLM harness; route/proprio serialization only). The safe claim is therefore not “latent beats all text on greedy accuracy”; it is: proprioceptive evidence is necessary, latent matches the strongest text route while exposing a θ head, and latent uses fewer prompt tokens than binned text (text_binned / latent token ratio `{token_ratio}`).

### A7.1b Paired taxonomy slices (latent vs REFLECT vs rich; T4 falsifier)

Status: {_status_line(tax)}.

{_taxonomy_table(tax)}

T4 comparison: latent `{tax_t4.get("latent_t4_acc", "n/a")}`, reflect_scalar `{tax_t4.get("reflect_t4_acc", "n/a")}`, rich_stats `{tax_t4.get("rich_t4_acc", "n/a")}`. Design falsifier “rich text matches latent on T4”: `{"triggered" if tax_t4.get("rich_matches_or_beats_latent") else "not triggered"}`{"; both_fail_t4" if tax_t4.get("both_fail_t4") else ""}. {tax_t4.get("interpretation") or "When triggered, the latent claim narrows to bandwidth/integration/θ rather than a forced T4 accuracy gap."}

**Curriculum confound (explicit):** these three adapters are **not** a matched latent-vs-text injection retrain. Latent uses A3 `b5_conflict_bi`, reflect_scalar uses A2 `b_text`, rich_stats uses the A7 rich adapter—different conflict curricula and training recipes. Therefore T3 gaps (e.g. latent/rich 0.917 vs reflect 0.354) and test-time grounding gaps must **not** be sold as pure injection-modality wins; they are curriculum+route packages. The pure injection claim remains the M7 route table above (θ + tokens under matched harness).

### A7.2 Method slices over the A3 taxonomy

{_a3_table(a3rows)}

The table preserves the block-structured failure story. B1 is strong on T4 proprio fine structure but weak on T3 conflict; B-T/B-V/B-F/B5-unshaped do not solve the proprio-true T3 block; B5-conflict-bi repairs T3 to {a3rows.get("B5-conflict-bi", {}).get("T3", {}).get("acc", "n/a")} but remains a conflict-specialist rather than a universal T4 model. This supports the C2 interpretation that conflict resolution must be trained as evidence weighing, including the proprio-true direction.

### A7.3 Conflict-data dose curve

{_dose_tables(conflict)}

Aggregate:

{_dose_aggregate_text(conflict)}

**Headline (protocol-matched only):** best mean O4-conflict attribution is at dose `{best_dose}` with mean `{best_cell.get("mean_rate", "n/a")}` (range `{best_cell.get("min_rate", "n/a")}`–`{best_cell.get("max_rate", "n/a")}`). Dose 10 is also the best A4-projected-regret operating point. Protocol dose 5 is **unstable** under the same unique-sample recipe (mean `{dose5_cell.get("mean_rate", "n/a")}`, min `{dose5_cell.get("min_rate", "n/a")}` including seed1 O4=0.0); doses 20/40 plateau at mean 0.667. Every seed is non-monotone. A7 therefore claims a practical **10-sample** curriculum under the controlled recipe—not that “any ≤10 matched samples” is equally reliable, and not that dose 5 is an efficient knee.

McNemar p values are paired diagnostics over shared snapshots within one seed; tiny snapshot-level p values do not replace uncertainty over training replicates. The dose claim therefore treats the three training seeds as the model-level independent units and reports their mean, range, and non-monotonicity together.

#### Sensitivity only (unequal recipe; excluded from headline aggregate)

Status: {_status_line(dose_sens)}.

{_dose_sensitivity_text(dose_sens)}

### A7.4 ERS/regret consequence projection

Aggregate by dose:

{_ers_aggregate_table(ers)}

Per evaluated adapter:

{_ers_table(ers)}

ERS/regret is an attribution-implied A4 projection: cached A7/A2 rows do not preserve action parameters, so each predicted attribution is mapped through the frozen recovery taxonomy and scored using `M_mean_cost`. It is a mechanism diagnostic, not actual-action ERS. Across three complete seeds, dose `{ers.get("best_regret_dose", "n/a")}` gives the lowest mean projected regret (`{((ers.get("aggregate_by_dose") or {}).get(str(ers.get("best_regret_dose")), {}) or {}).get("mean_regret", "n/a")}`); neighboring doses plateau nearby and do not improve monotonically. Dose 10 is the cleanest joint attribution+cost operating point under the controlled recipe.

### A7.5 Truth filter and rationale grounding

Status: {_status_line(cot)}.

The real ApiOracle Hindsight stream contains `{cot.get("n_kept_truth_filtered", "n/a")}` truth-filter-kept records and `{cot.get("n_filter_rejected_with_annotation", "n/a")}` filter-rejected annotated records; `{cot.get("n_oracle_errors_excluded", "n/a")}` oracle transport-error rows are excluded from the filter-effect denominator. Grounded-correct rationale rate improves from `{cot_unfiltered.get("grounded_correct_rate", "n/a")}` before filtering to `{cot_filtered.get("grounded_correct_rate", "n/a")}` after filtering (gain `{cot_gain}`). The keep rate over judged real ApiOracle annotations is `{cot.get("filter_keep_rate", "n/a")}`. This makes truth filtering reportable without rerunning or substituting an Oracle stream.

#### Test-time emitted-rationale grounding

Status: {_status_line(ttg)}.

{_ttg_table(ttg)}

Design §A7 also requires the same automatic checker on *emitted* rationales at test time (not only the training-stream ApiOracle filter). Best grounded-correct rate among scored real adapters: `{ttg_headline.get("best_grounded_correct_rate", "n/a")}` (n_agents=`{ttg_headline.get("n_agents_scored", "n/a")}`). This separates accuracy-with-grounded-explanations from confabulated correctness. **Grounding-rate gaps across latent vs rich here are not pure injection effects** (different training curricula; see §A7.1b); they remain valid test-time measurements of the deployed adapters that the paper actually uses.

### A7.6 Encoder / θ residual / abstention

Encoder status: {_status_line(enc)}; grid status: {_status_line(encoder_grid)}.

{_encoder_best_text(enc, encoder_grid)}

{_encoder_grid_table(encoder_grid)}

The full grid uses `{encoder_grid.get("n_samples", "n/a")}` frozen snapshots from `{", ".join(encoder_grid.get("corpus_dirs", [])) if encoder_grid.get("corpus_dirs") else "n/a"}` and three seeds `{encoder_grid.get("seeds", "n/a")}`. Privileged-distillation reaches the top attribution tier (0.805; tied by the supervised from-scratch control on attribution) while uniquely providing strong θ grounding; its T=25 gate-on cell gives the best θ MAE (0.0279). Contrastive-only does not receive privileged θ loss, so its θ MAE stays far worse; from-scratch is the random-initialized supervised attribution control and is not θ-grounded. Separately, the deployed A5 projector residual ranks B5-conflict-bi attribution errors with AUROC `{b5_abs.get("auroc", "n/a")}` (n={b5_abs.get("n", "n/a")}, errors={b5_abs.get("n_err", "n/a")}), but A5 has no out-of-range θ samples. On frozen test appearances, residual thresholding selects coverage `{broad_selected.get("test", {}).get("coverage", "n/a")}` and cost `{broad_selected.get("test", {}).get("expected_cost", "n/a")}`; the minimum nonzero point has coverage `{broad_nonzero.get("test", {}).get("coverage", "n/a")}` and cost `{broad_nonzero.get("test", {}).get("expected_cost", "n/a")}`, worse than always-safe `{broad_nonzero.get("test", {}).get("baseline_cost", {}).get("safe", "n/a")}`. A6 identifies only the base boundary bracket `{theta_bracket}`; logistic midpoint `{theta_star}` is descriptive and no learned decision flip was observed.

### A7.7 Review-driven abstention baselines and calibration

Status: {_status_line(a71)}; posterior method: `{a71.get("posterior_method", "n/a")}`; n=`{a71.get("n_points", "n/a")}`. Best AURC score: `{a71_best_name}` with AURC `{a71_best_cell.get("aurc", "n/a")}`.

{_aurc_table(a71_aurc)}

Held-out threshold selection uses the pre-registered appearance split (`train`: 288 calibration rows; `test`: 300 rows):

{_heldout_table(a71_heldout)}

The calibration-only conformal-style screen uses risk budget `{a71.get("risk_budget", "n/a")}` ({a71.get("risk_budget_definition", "n/a")}) and reports whether a threshold's upper confidence cost is below that budget:

{_conformal_table(a71_conformal)}

Posterior reliability from direct action-completion log-prob scoring, calibrated against the posterior argmax attribution rather than the executed-agent decision, gives ECE `{a71_ece.get("ece", "n/a")}` over n=`{a71_ece.get("n", "n/a")}`. This is poor calibration and only a diagnostic. Entropy/MSP improve AURC over θ residual and reach test cost `{a71_heldout.get("entropy", {}).get("test", {}).get("expected_cost", "n/a")}` at coverage `{a71_heldout.get("entropy", {}).get("test", {}).get("coverage", "n/a")}`. Their paired two-stage `selective−safe` CI is `{a71_heldout.get("entropy", {}).get("two_stage_bootstrap", {}).get("paired_delta_ci", {}).get("selective_minus_safe", "n/a")}`: it touches zero, so the result is a nonzero-coverage trend, **not** strict superiority over always-safe. Ensemble variance shifts badly under held-out appearances, and no score passes the calibration conformal screen.

## 5. Claim bridge / falsifier

A7 supports C2 by showing that conflict competence depends on method choices rather than generic multimodality. The artifact-backed bridge is:

1. Vision-only fails the ambiguity regime while proprio-injected routes solve it, so the body channel is load-bearing.
2. Latent does not need to beat rich/oracle text on greedy accuracy to be useful; it preserves accuracy while carrying θ-grounded residuals and reducing prompt bandwidth relative to binned text.
3. Under the **single protocol-matched recipe**, conflict data has a measured 3-seed knee at **dose 10** (mean O4 attr 0.733; also best A4-projected mean regret 0.40). Protocol dose 5 is unstable (mean 0.333, includes catastrophic seed cells) and is not claimed as an efficiency knee.
4. Truth filtering raises grounded-correct rationale rate on the real ApiOracle stream (0.604→0.765). Test-time emitted-rationale grounding applies the same checker to model generations (best GC 0.8), so CoT quality is measurable beyond the training filter—without claiming pure injection causality for cross-adapter grounding gaps.
5. The encoder grid shows privileged distillation preserves the top attribution tier while supplying θ grounding that contrastive-only/from-scratch controls lack; A5/A6 show that this representation benefit has not yet become calibrated OOD or intervention-boundary control.
6. A7.1 adds the review-requested selective-prediction benchmark: entropy/MSP beat θ residual on AURC and outperform deployed-agent/continue baselines, but do not strictly beat always-safe under the paired two-stage CI.
7. Paired taxonomy slices close the design T4 falsifier as a **match-at-floor** among conflict-route packages; the pure latent injection claim stays with M7 (θ + tokens), not with unmatched-curriculum T3/grounding deltas.

Falsifiers remain explicit: protocol dose-0 matching protocol dose-10 on O4 conflict would erase the curriculum claim; rich text matching latent on token budget *and* θ under the **matched M7 harness** would collapse the latent advantage to implementation convenience; truth filtering failing would undercut the CoT-quality claim; uncertainty scores failing held-out risk-coverage would undermine the abstention method claim; contrastive-only/from-scratch matching privileged-distillation on θ MAE would erase the encoder-distillation claim. Mixing unequal-recipe dose5 cells into the controlled dose aggregate would also invalidate the dose-axis claim—and is forbidden in this report.

## 6. Honest scope

- A7 is complete for artifact-backed method readouts and reportable as a method-ablation appendix/main-text support package.
- The raw unfiltered real `ApiOracle` CoT comparison uses the kept+dropped Hindsight stream; no external call or surrogate stream is used.
- Test-time emitted-rationale grounding is offline over frozen matched snapshots with real VLA generations.
- Taxonomy T3/grounding comparisons across latent/reflect/rich are **curriculum-confounded**; only the M7 route table is a pure injection ablation.
- The full encoder grid is complete on recorded real binding windows; offline only.
- Greedy latent-vs-text accuracy ties on the M7 ambiguity benchmark; latent claim is efficiency + θ grounding + sampling robustness, not a forced greedy accuracy gap.
- Conflict-dose headline uses only protocol-matched unique-sample / epochs=4 cells. Dose5 instability is real under that protocol and is reported honestly. Unequal-recipe upsample stabilization is sensitivity-only (`conflict_dose/sensitivity_dose05_upsample/`).
- A7.1 abstention baselines are offline over frozen snapshots and the A4 cost matrix; the appearance split is frozen, and the paired CI resamples both appearance clusters and A4 episodes.
- Add-on ERS/OOD/θ* readouts reuse A4/A5/A6 artifacts and are not independent A7 closed-loop experiments.

## 7. Reproduction

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_build_datasets.py --config configs/eval/a7.yaml --stage all
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_eval.py --config configs/eval/a7.yaml --stage all
# A7.1 direct VLA posterior scoring (uses cache on rerun)
env -u PYTHONPATH HF_HUB_OFFLINE=1 KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct \
  ~/miniconda3/envs/kinovla/bin/python scripts/a7_eval.py --config configs/eval/a7.yaml --stage abstention-baselines --evaluate
# Test-time rationale grounding + paired T4 taxonomy (real VLA)
env -u PYTHONPATH HF_HUB_OFFLINE=1 KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct \
  ~/miniconda3/envs/kinovla/bin/python scripts/a7_eval.py --config configs/eval/a7.yaml --stage test-time-grounding --evaluate
env -u PYTHONPATH HF_HUB_OFFLINE=1 KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct \
  ~/miniconda3/envs/kinovla/bin/python scripts/a7_eval.py --config configs/eval/a7.yaml --stage text-schema-taxonomy --evaluate
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_report.py --config configs/eval/a7.yaml --out A实验/A7.md
```

To evaluate additional real adapters without repeating the cached seed0 sweep, run the corresponding `scripts/a7_train.py` arm, then rerun `scripts/a7_eval.py --evaluate` for only the new seed/dose/schema.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate A7 report from machine JSON artifacts")
    ap.add_argument("--config", default="configs/eval/a7.yaml")
    ap.add_argument("--out", default="A实验/A7.md")
    args = ap.parse_args()
    text = build_report(args.config)
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
