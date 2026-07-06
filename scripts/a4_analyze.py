#!/usr/bin/env python
"""A4.3 — reduce the interventional matrix into the C3 readouts (offline; Paper-A §4 A4.3).

Consumes ``outputs/eval/a4/matrix.jsonl`` (the forced-label outcomes) + the A2/A3 per-snapshot
agent predictions, and produces:

  1. **M(s, ℓ)** — the causal success (+ catastrophic-event + cost) matrix per (scenario, label).
  2. **Cost asymmetry → safe default** — for the T2 adhesion/mud pair, the expected-cost crossover
     ``p*`` under ``P(adhesion)``; the conservative recovery (Backstep) wins for any p > p* because
     push-through-on-adhesion is catastrophic (immobilization) ≫ back-out-on-mud (benign-slow).
  3. **ERS / Regret composition** — each A2/A3 agent's predicted label distribution composes with M
     ⇒ expected physical cost + regret vs the canonical action (attribution error measured in
     physical-cost units; the A4.4 closed-loop check validates this composition).

Renders heatmap.png (M success), cost_heatmap.png (mean cost), safe_default.png (the crossover),
ers_regret.png (per-agent). Writes a4_results.json. No Isaac.

Run:  ~/miniconda3/envs/kinovla/bin/python scripts/a4_analyze.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kino_vla.eval.a4_consequence import (
    DEFAULT_COST_LEVELS,
    LABELS,
    M_matrix,
    Outcome,
    compose_agent,
    primitive_to_label,
    safe_default_crossover,
)


# A2/A3 per-snapshot (cell, truth, t3_sub) → A4 matrix scenario. The attribution task is identical
# (the A4 two-phase plateau ≡ the matched_O4 the agents were evaluated on, A1.3), so the agents'
# matched_O4 predictions transfer to matched_O4_twophase.
def snap_to_a4(row: dict) -> str | None:
    cell, truth, sub = row.get("cell"), row.get("truth"), row.get("t3_sub", "")
    if cell == "T2" and truth == "adhesion":
        return "matched_O4_twophase"
    if cell == "T1" and truth == "compliant_terrain":
        return "matched_O2"
    if cell == "T1" and truth == "low_friction":
        return "O1_ice"
    if cell == "T3" and sub == "looks_safe":
        return "O7_looks_safe"
    if cell == "T3" and sub == "reverse":
        return "O7_reverse"
    if cell == "T3" and truth == "invisible_obstacle":
        return "O8_invisible"
    if cell == "T4" and truth == "overload":
        return "O5_payload_B"
    if cell == "T4" and truth == "effort_decay":
        return "O10_decay_B"
    if cell == "T5" and truth == "compliant_terrain":
        return "O2_A_nominal"
    if cell == "T5" and truth == "low_friction":
        return "O1_A_nominal"
    if cell == "T5" and truth == "effort_decay":
        return "O10_A_nominal"
    if cell == "T5" and truth == "external_push":
        return "O6_push_A"
    return None


def load_outcomes(path: Path) -> list[Outcome]:
    outs = []
    for ln in path.read_text().splitlines():
        if ln:
            d = json.loads(ln)
            outs.append(Outcome(**d))
    return outs


def agent_distribution(per_item: list[dict], agent: str) -> dict[str, dict[str, float]]:
    """Per-scenario P̂(label) for one agent, mapping its emitted primitives onto M's columns."""
    by_scn: dict[str, list[dict]] = {}
    for r in per_item:
        s = snap_to_a4(r)
        if s is None:
            continue
        by_scn.setdefault(s, []).append(r)
    dist: dict[str, dict[str, float]] = {}
    for s, rows in by_scn.items():
        n = len(rows)
        counts = {lab: 0 for lab in LABELS}
        for r in rows:
            prim = r.get("primitive")
            params = {}
            lab = primitive_to_label(prim, params) if prim else "continue"
            counts[lab] = counts.get(lab, 0) + 1
        dist[s] = {lab: round(counts.get(lab, 0) / n, 4) for lab in LABELS}
    return dist


def main() -> None:
    ap = argparse.ArgumentParser(description="A4.3 consequence analysis (offline)")
    ap.add_argument("--matrix", default="outputs/eval/a4/matrix.jsonl")
    ap.add_argument("--a3-dir", default="outputs/eval/a3")
    ap.add_argument("--agents", default="B1,B5_unshaped,B5_conflict,B5_conflict_bi,B_V,B_T,B_F")
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
        print(
            f"  crossover p*(P(adhesion))={asym['p_star']} (Backstep safe for p > p*)"
        )
        print(f"  cost-asymmetry ratio off[high]/off[back] = {asym['cost_asymmetry_ratio']}")

    # ---- 3. ERS / Regret composition (A2/A3 agents) ----
    composition: dict[str, dict] = {}
    a3dir = Path(args.a3_dir)
    for agent in [a.strip() for a in args.agents.split(",") if a.strip()]:
        f = a3dir / f"per_item_{agent}.json"
        if not f.exists():
            continue
        per_item = json.loads(f.read_text())
        dist = agent_distribution(per_item, agent)
        comp = compose_agent(dist, m, scenarios, canon)
        composition[agent] = {
            "ers_mean_cost": comp.ers_mean,
            "regret_mean": comp.regret_mean,
            "ers_per_scenario": comp.ers_per_scenario,
            "regret_per_scenario": comp.regret_per_scenario,
            "argmin_label": comp.argmin_label_per_scenario,
            "distribution": dist,
        }
        print(f"[a4] {agent}: ERS(cost)={comp.ers_mean:.2f}  Regret={comp.regret_mean:.2f}")

    result = {
        "n_outcomes": len(outs),
        "scenarios": scenarios,
        "labels": labels,
        "canonical_label": canon,
        "M_success": {s: {lab: m[s][lab]["success_rate"] for lab in labels} for s in scenarios},
        "M_mean_cost": {s: {lab: m[s][lab]["mean_cost"] for lab in labels} for s in scenarios},
        "M_full": m,
        "cost_levels": DEFAULT_COST_LEVELS,
        "safe_default": asym,
        "composition": composition,
    }
    (out / "a4_results.json").write_text(json.dumps(result, indent=2))
    _render(out, scenarios, labels, m, asym, composition)
    msg = (f"[OK] wrote {out / 'a4_results.json'} + heatmap.png + cost_heatmap.png + "
            f"safe_default.png + ers_regret.png")
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

    # ERS / Regret per agent
    if composition:
        agents = list(composition.keys())
        ers = [composition[a]["ers_mean_cost"] for a in agents]
        reg = [composition[a]["regret_mean"] for a in agents]
        x = np.arange(len(agents))
        w = 0.4
        fig, ax = plt.subplots(figsize=(max(6, len(agents) * 0.9), 4.5))
        ax.bar(x - w / 2, ers, w, label="ERS (expected cost)", color="#d62728")
        ax.bar(x + w / 2, reg, w, label="Regret vs canonical", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("physical cost (units)")
        ax.set_title("A4 ERS / Regret — attribution error in cost units")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / "ers_regret.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
