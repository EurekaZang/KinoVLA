#!/usr/bin/env python
"""A4.3 — reduce the interventional matrix into the C3 readouts (offline; Paper-A §4 A4.3).

Consumes ``outputs/eval/a4/matrix.jsonl`` (the forced-label outcomes) + the A2/A3 per-snapshot
agent predictions, and produces:

  1. **M(s, ℓ)** — the causal success (+ catastrophic-event + cost) matrix per (scenario, label).
  2. **Cost asymmetry → safe default** — for the T2 adhesion/mud pair, the expected-cost crossover
     ``p*`` under ``P(adhesion)``; the conservative recovery (Backstep) wins for any p > p* because
     push-through-on-adhesion is catastrophic (immobilization) ≫ back-out-on-mud (benign-slow).
  3. **Two non-interchangeable compositions** — (a) the action actually emitted by the agent and
     (b) the action implied by its predicted attribution.  Their agreement rate is audited before
     either is composed with M.  Regret integrates the complete empirical distribution and is
     measured against the best observed matrix action, with the canonical gap reported separately.

Renders heatmap.png (M success), cost_heatmap.png (mean cost), safe_default.png (the crossover),
ers_regret.png (per-agent). Writes a4_results.json. No Isaac.

Run:  ~/miniconda3/envs/kinovla/bin/python scripts/a4_analyze.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from kino_vla.eval.a4_consequence import (
    DEFAULT_COST_LEVELS,
    LABELS,
    M_matrix,
    Outcome,
    a3_row_to_a4_scenario,
    attribution_to_label,
    compose_agent,
    physical_cost,
    primitive_to_label,
    safe_default_crossover,
)


# A2/A3 per-snapshot (cell, truth, t3_sub) → A4 matrix scenario. The attribution task is identical
# (the A4 two-phase plateau ≡ the matched_O4 the agents were evaluated on, A1.3), so the agents'
# matched_O4 predictions transfer to matched_O4_twophase.
def snap_to_a4(row: dict) -> str | None:
    return a3_row_to_a4_scenario(row)


def load_outcomes(path: Path) -> list[Outcome]:
    outs = []
    for ln in path.read_text().splitlines():
        if ln:
            d = json.loads(ln)
            outs.append(Outcome(**d))
    return outs


def _label_for_row(row: dict, source: str) -> str:
    if source == "attribution_implied":
        return attribution_to_label(row.get("attribution"))
    if source != "actual_action":
        raise ValueError(f"unknown composition source: {source}")
    prim = row.get("primitive")
    if not prim:
        return "continue"
    params = row.get("primitive_params") or {}
    if prim == "Switch_Gait" and not row.get("action_params_observed", False):
        raise ValueError(
            f"legacy A3 row {row.get('sid')} lost Switch_Gait parameters; rerun eval_a3.py"
        )
    return primitive_to_label(prim, params, strict_params=True)


def agent_distribution(
    per_item: list[dict], agent: str, *, source: str = "actual_action"
) -> tuple[dict[str, dict[str, float]], dict]:
    """Per-scenario empirical label distribution plus an attribution/action consistency audit."""
    by_scn: dict[str, list[dict]] = {}
    for r in per_item:
        s = snap_to_a4(r)
        if s is None:
            continue
        by_scn.setdefault(s, []).append(r)
    dist: dict[str, dict[str, float]] = {}
    n_consistent = 0
    n_audited = 0
    for s, rows in by_scn.items():
        n = len(rows)
        counts = {lab: 0 for lab in LABELS}
        for r in rows:
            lab = _label_for_row(r, source)
            counts[lab] = counts.get(lab, 0) + 1
            if source == "actual_action":
                n_audited += 1
                n_consistent += int(lab == _label_for_row(r, "attribution_implied"))
        dist[s] = {lab: round(counts.get(lab, 0) / n, 4) for lab in LABELS}
    audit = {
        "agent": agent,
        "source": source,
        "n_mapped_rows": sum(len(v) for v in by_scn.values()),
        "n_scenarios": len(by_scn),
        "attribution_action_consistency_n": n_audited,
        "attribution_action_consistency": (
            round(n_consistent / n_audited, 4) if n_audited else None
        ),
    }
    return dist, audit


def _composition_payload(dist: dict, m: dict, scenarios: list[str], canon: dict) -> dict:
    comp = compose_agent(dist, m, scenarios, canon)
    canonical_mass = {s: round(float(dist[s].get(canon[s], 0.0)), 4) for s in scenarios}
    return {
        "n_scenarios": len(scenarios),
        "ers_mean_cost": comp.ers_mean,
        "success_mean": comp.success_mean,
        "regret_vs_best_mean": comp.regret_mean,
        "canonical_gap_mean": comp.canonical_gap_mean,
        "canonical_action_rate": round(sum(canonical_mass.values()) / len(scenarios), 4),
        "ers_per_scenario": comp.ers_per_scenario,
        "success_per_scenario": comp.success_per_scenario,
        "regret_vs_best_per_scenario": comp.regret_per_scenario,
        "canonical_gap_per_scenario": comp.canonical_gap_per_scenario,
        "mode_label": comp.mode_label_per_scenario,
        "canonical_mass_per_scenario": canonical_mass,
        "distribution": dist,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_compositions(
    per_items: dict[str, list[dict]],
    outcomes: list[Outcome],
    scenarios: list[str],
    canonical: dict[str, str],
    *,
    reps: int,
    seed: int,
) -> dict:
    """Paired two-stage bootstrap over appearance clusters and A4 matrix episodes.

    All agents share each resampled sample id, so agent differences are paired.  Appearance
    clusters are resampled independently within scenario; physical outcomes are resampled within
    each measured (scenario, label) cell.
    """
    rng = np.random.default_rng(seed)
    first_agent = next(iter(per_items))
    base_rows = {row["sid"]: row for row in per_items[first_agent]}
    aligned = {agent: {row["sid"]: row for row in rows} for agent, rows in per_items.items()}
    clusters: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sid, row in base_rows.items():
        scenario = snap_to_a4(row)
        if scenario not in scenarios:
            continue
        appearance = str(row.get("appearance_id") or sid)
        clusters[scenario][f"{scenario}|{appearance}"].append(sid)
    for agent, rows in aligned.items():
        missing = [
            sid
            for groups in clusters.values()
            for ids in groups.values()
            for sid in ids
            if sid not in rows
        ]
        if missing:
            raise ValueError(f"paired bootstrap alignment failed for {agent}: {missing[:3]}")

    outcome_cells: dict[tuple[str, str], list[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.scenario in scenarios:
            outcome_cells[(outcome.scenario, outcome.label)].append(outcome)

    sources = ("actual_action", "attribution_implied")
    metric_names = ("ers_mean_cost", "regret_vs_best", "success_mean", "canonical_gap")
    draws = {
        source: {agent: {metric: [] for metric in metric_names} for agent in per_items}
        for source in sources
    }
    for _ in range(reps):
        selected: list[tuple[str, str]] = []
        for scenario in scenarios:
            group_map = clusters[scenario]
            keys = list(group_map)
            for index in rng.integers(0, len(keys), size=len(keys)):
                selected.extend((scenario, sid) for sid in group_map[keys[int(index)]])

        boot_m: dict[str, dict[str, dict]] = {scenario: {} for scenario in scenarios}
        for scenario in scenarios:
            for label in LABELS:
                cell = outcome_cells[(scenario, label)]
                sampled = rng.choice(cell, size=len(cell), replace=True)
                boot_m[scenario][label] = {
                    "n": len(cell),
                    "mean_cost": float(np.mean([physical_cost(outcome) for outcome in sampled])),
                    "success_rate": float(np.mean([outcome.success for outcome in sampled])),
                }

        for source in sources:
            for agent, rows_by_sid in aligned.items():
                counts = {scenario: {label: 0 for label in LABELS} for scenario in scenarios}
                totals = {scenario: 0 for scenario in scenarios}
                for scenario, sid in selected:
                    label = _label_for_row(rows_by_sid[sid], source)
                    counts[scenario][label] += 1
                    totals[scenario] += 1
                dist = {
                    scenario: {
                        label: counts[scenario][label] / totals[scenario] for label in LABELS
                    }
                    for scenario in scenarios
                }
                comp = compose_agent(dist, boot_m, scenarios, canonical)
                values = {
                    "ers_mean_cost": comp.ers_mean,
                    "regret_vs_best": comp.regret_mean,
                    "success_mean": comp.success_mean,
                    "canonical_gap": comp.canonical_gap_mean,
                }
                for metric, value in values.items():
                    draws[source][agent][metric].append(value)

    def summary(values: list[float]) -> dict:
        return {
            "mean": round(float(np.mean(values)), 4),
            "ci95": [
                round(float(np.quantile(values, 0.025)), 4),
                round(float(np.quantile(values, 0.975)), 4),
            ],
        }

    result = {
        "reps": reps,
        "seed": seed,
        "unit": "appearance cluster within scenario + episode within A4 matrix cell",
        "n_scenarios": len(scenarios),
        "scenarios": scenarios,
        "by_source": {},
    }
    for source in sources:
        agent_summary = {
            agent: {metric: summary(values) for metric, values in metrics.items()}
            for agent, metrics in draws[source].items()
        }
        paired = {}
        reference = "B5_conflict" if "B5_conflict" in per_items else first_agent
        for comparator in per_items:
            if comparator == reference:
                continue
            paired[comparator] = {}
            for metric in ("ers_mean_cost", "regret_vs_best"):
                delta = [
                    a - b
                    for a, b in zip(
                        draws[source][reference][metric],
                        draws[source][comparator][metric],
                        strict=True,
                    )
                ]
                paired[comparator][f"{reference}_minus_{comparator}_{metric}"] = summary(delta)
        result["by_source"][source] = {
            "agents": agent_summary,
            "paired_reference": reference,
            "paired_differences": paired,
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="A4.3 consequence analysis (offline)")
    ap.add_argument("--matrix", default="outputs/eval/a4/matrix.jsonl")
    ap.add_argument("--a3-dir", default="outputs/eval/a3")
    ap.add_argument("--agents", default="B1,B5_unshaped,B5_conflict,B5_conflict_bi,B_V,B_T,B_F")
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260718)
    ap.add_argument("--out", default="outputs/eval/a4")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    outs = load_outcomes(Path(args.matrix))
    scenarios = sorted({o.scenario for o in outs})
    labels = [lab for lab in LABELS if any(o.label == lab for o in outs)]
    print(
        f"[a4] {len(outs)} outcomes | {len(scenarios)} scenarios | {len(labels)} labels", flush=True
    )
    m = M_matrix(outs, scenarios, tuple(labels))
    # the registry canonical label per scenario (the diagonal)
    from kino_vla.eval.registry import load_registry

    reg = load_registry()
    canon = {}
    _CANON = {
        "Backstep": "backstep_detour",
        "Set_Constraint": "slow_low",
        "Switch_Gait": "high_step",
        "Hold_and_Request": "hold_request",
        "Update_Topology": "detour_replan",
        "continue": "continue",
    }
    for s in scenarios:
        if s in reg.scenarios:
            cprim = reg[s].canonical_recovery.primitive
            mode = (reg[s].canonical_recovery.params or {}).get("mode", "")
            canon[s] = (
                "crawl"
                if cprim == "Switch_Gait" and mode == "crawl"
                else _CANON.get(cprim, "continue")
            )

    # ---- 1. printed M matrix (success rate) ----
    print("\n=== M(s,ℓ) success rate ===")
    print(f"{'scenario':<20}" + "".join(f"{lab:>13}" for lab in labels))
    for s in scenarios:
        print(f"{s:<20}" + "".join(f"{m[s][lab]['success_rate']:>13.2f}" for lab in labels))
    print("\n=== M(s,ℓ) mean physical cost (lower better) ===")
    print(f"{'scenario':<20}" + "".join(f"{lab:>13}" for lab in labels))
    for s in scenarios:
        print(f"{s:<20}" + "".join(f"{m[s][lab]['mean_cost']:>13.2f}" for lab in labels))

    # ---- 2. cost asymmetry → safe default (T2 pair) ----
    asym = None
    if "matched_O4_twophase" in m and "matched_O2" in m:
        # diag = cost of the CORRECT action per truth; off = cost of the WRONG action per truth
        diag = {
            "backstep_detour": m["matched_O4_twophase"]["backstep_detour"]["mean_cost"],
            "high_step": m["matched_O2"]["high_step"]["mean_cost"],
        }
        off = {
            "backstep_detour": m["matched_O2"]["backstep_detour"][
                "mean_cost"
            ],  # back-out on mud (benign)
            "high_step": m["matched_O4_twophase"]["high_step"]["mean_cost"],
        }  # push-through on adhesion (catastrophic)
        asym = safe_default_crossover(diag, off, ("backstep_detour", "high_step"))
        print("\n=== Cost asymmetry → safe default (T2 adhesion/mud pair) ===")
        print(
            f"  backstep on adhesion (diag): {diag['backstep_detour']:.2f}  | "
            f"backstep on mud (off, benign): {off['backstep_detour']:.2f}"
        )
        print(
            f"  high_step on mud (diag):      {diag['high_step']:.2f}  | "
            f"high_step on adhesion (off, CATASTROPHIC): {off['high_step']:.2f}"
        )
        print(f"  crossover p*(P(adhesion))={asym['p_star']} (Backstep safe for p > p*)")
        print(f"  cost-asymmetry ratio off[high]/off[back] = {asym['cost_asymmetry_ratio']}")

    scenario_groups = {
        "all": scenarios,
        "t2_pair": [s for s in ("matched_O4_twophase", "matched_O2") if s in scenarios],
        "clean_unique": [
            s
            for s in (
                "matched_O4_twophase",
                "O8_invisible",
                "O5_payload_B",
                "O10_A_nominal",
                "O1_A_nominal",
                "O2_A_nominal",
            )
            if s in scenarios
        ],
        "non_nominal": [s for s in scenarios if not s.endswith("_A_nominal")],
        "nominal": [s for s in scenarios if s.endswith("_A_nominal")],
        "known_weak": [s for s in ("O7_looks_safe", "O10_decay_B") if s in scenarios],
    }

    matrix_support: dict[str, dict] = {}
    for s in scenarios:
        c = canon[s]
        other_labels = [lab for lab in labels if lab != c]
        c_success = float(m[s][c]["success_rate"])
        best_off_success = max(float(m[s][lab]["success_rate"]) for lab in other_labels)
        c_cost = float(m[s][c]["mean_cost"])
        best_cost = min(float(m[s][lab]["mean_cost"]) for lab in labels)
        c_ci = m[s][c]["success_ci"]
        max_off_ci_hi = max(float(m[s][lab]["success_ci"][1]) for lab in other_labels)
        matrix_support[s] = {
            "canonical_label": c,
            "canonical_success": c_success,
            "best_offdiagonal_success": best_off_success,
            "canonical_unique_success_winner": c_success > best_off_success,
            "canonical_wilson_separated": float(c_ci[0]) > max_off_ci_hi,
            "canonical_cost": c_cost,
            "best_measured_cost": best_cost,
            "canonical_cost_optimal": c_cost <= best_cost + 1e-12,
        }

    # ---- 3. Actual-action and attribution-implied composition (A2/A3 agents) ----
    composition: dict[str, dict] = {}
    per_items_by_agent: dict[str, list[dict]] = {}
    a3dir = Path(args.a3_dir)
    input_hashes = {"matrix_jsonl": _sha256(Path(args.matrix))}
    for agent in [a.strip() for a in args.agents.split(",") if a.strip()]:
        f = a3dir / f"per_item_{agent}.json"
        if not f.exists():
            continue
        per_item = json.loads(f.read_text())
        per_items_by_agent[agent] = per_item
        input_hashes[f"a3_per_item_{agent}"] = _sha256(f)
        actual_dist, audit = agent_distribution(per_item, agent, source="actual_action")
        attr_dist, _ = agent_distribution(per_item, agent, source="attribution_implied")
        bridge_scenarios = [s for s in scenarios if s in actual_dist]
        missing = sorted(set(scenarios) - set(bridge_scenarios))
        if not bridge_scenarios:
            raise ValueError(f"{agent} has no A3 rows overlapping the A4 matrix")
        by_source = {}
        for source, dist in (
            ("actual_action", actual_dist),
            ("attribution_implied", attr_dist),
        ):
            payload = _composition_payload(dist, m, bridge_scenarios, canon)
            payload["strata"] = {
                name: _composition_payload(
                    dist, m, [s for s in scns if s in bridge_scenarios], canon
                )
                for name, scns in scenario_groups.items()
                if any(s in bridge_scenarios for s in scns)
            }
            by_source[source] = payload
        audit["matrix_scenarios_evaluated"] = bridge_scenarios
        audit["matrix_scenarios_without_a3_rows"] = missing
        composition[agent] = {"bridge_audit": audit, **by_source}
        act = by_source["actual_action"]
        print(
            f"[a4] {agent}: action ERS={act['ers_mean_cost']:.2f}  "
            f"regret(best)={act['regret_vs_best_mean']:.2f}  "
            f"attr/action consistency={audit['attribution_action_consistency']:.2f}"
        )

    bootstrap_scenarios = [
        scenario
        for scenario in scenarios
        if all(
            scenario in composition[agent]["bridge_audit"]["matrix_scenarios_evaluated"]
            for agent in composition
        )
    ]
    bootstrap = bootstrap_compositions(
        per_items_by_agent,
        outs,
        bootstrap_scenarios,
        canon,
        reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
    )

    result = {
        "schema_version": 2,
        "estimands": {
            "actual_action": "primitive + observed primitive_params composed with M",
            "attribution_implied": "predicted attribution mapped through frozen recovery taxonomy",
            "regret": "expected cost minus the best measured action cost in each scenario",
            "canonical_gap": "signed expected cost minus pre-registered canonical-action cost",
        },
        "input_sha256": input_hashes,
        "n_outcomes": len(outs),
        "scenarios": scenarios,
        "scenario_groups": scenario_groups,
        "labels": labels,
        "canonical_label": canon,
        "matrix_support": matrix_support,
        "M_success": {s: {lab: m[s][lab]["success_rate"] for lab in labels} for s in scenarios},
        "M_mean_cost": {s: {lab: m[s][lab]["mean_cost"] for lab in labels} for s in scenarios},
        "M_full": m,
        "cost_levels": DEFAULT_COST_LEVELS,
        "safe_default": asym,
        "composition": composition,
        "composition_bootstrap": bootstrap,
    }
    (out / "a4_results.json").write_text(json.dumps(result, indent=2))
    _render(out, scenarios, labels, m, asym, composition)
    msg = (
        f"[OK] wrote {out / 'a4_results.json'} + heatmap.png + cost_heatmap.png + "
        f"safe_default.png + ers_regret.png"
    )
    print(msg)


def _render(out: Path, scenarios, labels, m, asym, composition) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[a4] matplotlib missing; skipping figures")
        return

    # M success heatmap (scenarios × labels)
    mat = np.array([[m[s][lab]["success_rate"] for lab in labels] for s in scenarios])
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 0.5 * len(scenarios) + 2))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)
    for i, s in enumerate(scenarios):
        for j, lab in enumerate(labels):
            ax.text(j, i, f"{m[s][lab]['success_rate']:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("A4 M(s,ℓ) — success rate\n(diagonal = canonical recovery wins)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "heatmap.png", dpi=130)
    plt.close(fig)

    # mean-cost heatmap
    cmat = np.array([[m[s][lab]["mean_cost"] for lab in labels] for s in scenarios])
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 0.5 * len(scenarios) + 2))
    im = ax.imshow(cmat, vmin=0, vmax=max(6.0, cmat.max()), cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)
    for i, _s in enumerate(scenarios):
        for j, _lab in enumerate(labels):
            ax.text(j, i, f"{cmat[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("A4 M(s,ℓ) — mean physical cost (lower better; catastrophic red)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "cost_heatmap.png", dpi=130)
    plt.close(fig)

    # safe-default crossover
    if asym is not None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        da, db = asym["diag_cost"]["backstep_detour"], asym["diag_cost"]["high_step"]
        oa, ob = asym["off_cost"]["backstep_detour"], asym["off_cost"]["high_step"]
        p = np.linspace(0, 1, 50)
        e_back = p * da + (1 - p) * oa  # do backstep: correct on adhesion(p), benign-wrong on mud
        e_high = (
            p * ob + (1 - p) * db
        )  # do high_step: catastrophic-wrong on adhesion(p), correct on mud
        ax.plot(p, e_back, label="do Backstep (conservative)", lw=2)
        ax.plot(p, e_high, label="do High-step (push-through)", lw=2)
        ax.axvline(asym["p_star"], color="k", ls="--", label=f"p*={asym['p_star']}")
        ax.set_xlabel("P(adhesion)  (belief the patch is sticky, not mud)")
        ax.set_ylabel("expected physical cost")
        ax.set_title(
            "A4 cost asymmetry → safe default\nBackstep wins for any non-negligible P(adhesion)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "safe_default.png", dpi=130)
        plt.close(fig)

    # Actual-action ERS / empirical regret per agent.  Attribution-implied values remain in JSON as
    # a mechanism diagnostic and are intentionally not blended into the physical-action bars.
    if composition:
        agents = list(composition.keys())
        ers = [composition[a]["actual_action"]["ers_mean_cost"] for a in agents]
        reg = [composition[a]["actual_action"]["regret_vs_best_mean"] for a in agents]
        x = np.arange(len(agents))
        w = 0.4
        fig, ax = plt.subplots(figsize=(max(6, len(agents) * 0.9), 4.5))
        ax.bar(x - w / 2, ers, w, label="ERS (expected cost)", color="#d62728")
        ax.bar(x + w / 2, reg, w, label="Regret vs best measured", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("physical cost (units)")
        ax.set_title("A4 actual-action composition — physical cost units")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / "ers_regret.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
