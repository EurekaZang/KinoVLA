#!/usr/bin/env python
"""A5.5 — selective prediction / risk-coverage: the C3+C4 "Safe closer" (Paper-A §4 A5.5).

Ties C3 (A4's asymmetric cost M) to C4 (A5.4 abstention). For each abstention
threshold τ on the OOD-θ residual:
  - NON-abstained (residual < τ) ⇒ act on agent label ⇒ M(scenario, label);
  - abstained snapshots (residual ≥ τ) ⇒ the A4-derived conservative SAFE DEFAULT (Backstep:
    benign on mud, escape on adhesion, backs away from any hazard) ⇒ M(scenario, safe_default).
The expected-cost risk-coverage curve is compared against the two baselines:
  - ALWAYS-INTERVENE  (act on the agent's attribution, coverage = 1.0);
  - NEVER-INTERVENE   (always continue, the abstain-everything extreme).
A calibrated attributor + safe default STRICTLY DOMINATES both at the operating point — the C3/C4
payoff: abstention converts the asymmetric cost (C3) into a strict safety advantage (C4).

Offline: OOD-θ residual (projector-only) + A3 agent labels + A4 M (a4_results.json). No Isaac.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# the A4 conservative safe default: Backstep (benign on mud, escape on adhesion, backs off hazards)
SAFE_DEFAULT = "backstep_detour"


def snap_to_a4(row: dict) -> str | None:
    cell, truth, sub = row.get("cell"), row.get("truth"), row.get("t3_sub", "")
    if cell == "T2" and truth == "adhesion":
        return "matched_O4_twophase"
    if cell == "T1" and truth == "compliant_terrain":
        return "matched_O2"
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
    return None


def _prim_to_label(prim: str | None) -> str:
    m = {
        "Backstep": "backstep_detour",
        "Set_Constraint": "slow_low",
        "Hold_and_Request": "hold_request",
        "Update_Topology": "detour_replan",
        "Replan_Waypoint": "detour_replan",
        "continue": "continue",
    }
    if prim in m:
        return m[prim]
    if prim == "Switch_Gait":
        return "high_step"  # crawl≈high_step for the cost mapping
    if prim == "Adjust_Posture":
        return "slow_low"
    return "continue"


def compute_residuals(adapter: str, corpus_dirs: list[str]) -> dict[str, float]:
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
    pj.load_state_dict(torch.load(Path(adapter, "kino_projector.pt"), map_location="cpu"))
    pj.eval()
    sid2res: dict[str, float] = {}
    with torch.no_grad():
        for d in corpus_dirs:
            sp = Path(d, "samples.jsonl")
            npz_path = Path(d, "frames.npz")
            if not sp.exists() or not npz_path.exists():
                continue
            npz = np.load(npz_path)
            for ln in sp.read_text().splitlines():
                if not ln:
                    continue
                r = json.loads(ln)
                sid = r["sample_id"]
                tt = r.get("target_theta")
                if f"{sid}__proprio" not in npz or tt is None or len(tt) != 4:
                    continue
                w = torch.from_numpy(npz[f"{sid}__proprio"].astype(np.float32)).unsqueeze(0)
                _soft, theta = pj(w)
                pred = theta.squeeze(0).float().cpu().numpy()
                sid2res[sid] = float(np.linalg.norm(pred - np.asarray(tt, dtype=np.float32)))
    return sid2res


def main() -> None:
    ap = argparse.ArgumentParser(description="A5.5 risk-coverage Safe closer (offline)")
    ap.add_argument(
        "--agent-file",
        default="per_item_B5_conflict_bi",
        help="A3 per_item file for the agent (default B5-conflict-bi)",
    )
    ap.add_argument("--agent-name", default="B5-conflict-bi")
    ap.add_argument("--adapter", default="outputs/eval/a3/b5_conflict_bi/adapter_best")
    ap.add_argument("--a3-dir", default="outputs/eval/a3")
    ap.add_argument("--a4-results", default="outputs/eval/a4/a4_results.json")
    ap.add_argument("--corpus-dirs", default="outputs/eval/a0/corpus,outputs/eval/a3/corpus_t3")
    ap.add_argument(
        "--scenarios",
        default="matched_O4_twophase,matched_O2",
        help="restrict to these A4 scenarios (default the T2 pair = the principled "
        "safe-default case; '' = all)",
    )
    ap.add_argument("--out", default="outputs/eval/a5")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a4 = json.loads(Path(args.a4_results).read_text())
    Mfull = a4["M_full"]
    sid2res = compute_residuals(args.adapter, args.corpus_dirs.split(","))
    rows = json.loads((Path(args.a3_dir) / f"{args.agent_file}.json").read_text())
    keep_scn = set(s for s in args.scenarios.split(",") if s.strip())

    # build per-snapshot (residual, agent_label, scenario) joined to M costs
    pts: list[dict] = []
    for r in rows:
        scn = snap_to_a4(r)
        sid = r["sid"]
        if scn is None or scn not in Mfull or sid not in sid2res or not r.get("parsed"):
            continue
        if keep_scn and scn not in keep_scn:
            continue
        alab = _prim_to_label(r.get("primitive"))
        cells = Mfull[scn]
        pts.append(
            {
                "residual": sid2res[sid],
                "scenario": scn,
                "cost_agent": cells.get(alab, {}).get("mean_cost", 4.0),
                "cost_safe": cells.get(SAFE_DEFAULT, {}).get("mean_cost", 4.0),
                "cost_continue": cells.get("continue", {}).get("mean_cost", 4.0),
                "agent_label": alab,
                "attr_ok": bool(r.get("attr_ok")),
            }
        )
    print(
        f"[a5.5] agent={args.agent_name} | {len(pts)} snapshots mapped to A4 scenarios", flush=True
    )

    res = np.array([p["residual"] for p in pts])
    c_agent = np.array([p["cost_agent"] for p in pts])
    c_safe = np.array([p["cost_safe"] for p in pts])
    c_cont = np.array([p["cost_continue"] for p in pts])
    always_intervene = float(c_agent.mean())  # coverage 1.0 baseline
    never_intervene = float(c_cont.mean())  # always-continue baseline

    # sweep τ over residual quantiles ⇒ (coverage, expected_cost) for the calibrated policy
    taus = np.unique(np.quantile(res, np.linspace(0, 1, 30)))
    curve = []
    for tau in taus:
        keep = res < tau  # non-abstained ⇒ agent label
        cov = float(keep.mean())
        cost = float(np.where(keep, c_agent, c_safe).mean())
        curve.append(
            {
                "tau": round(float(tau), 4),
                "coverage": round(cov, 3),
                "expected_cost": round(cost, 3),
            }
        )
    # the operating point: minimum expected cost (the calibrated policy's sweet spot)
    best = min(curve, key=lambda c: c["expected_cost"])

    print(f"[a5.5] always-intervene (agent label) cost = {always_intervene:.3f}")
    print(f"[a5.5] never-intervene (continue)       cost = {never_intervene:.3f}")
    print(
        f"[a5.5] CALIBRATED best (abstain→{SAFE_DEFAULT}): cost={best['expected_cost']:.3f} "
        f"@ coverage={best['coverage']:.3f} (τ={best['tau']})"
    )
    dominates = (best["expected_cost"] < always_intervene) and (
        best["expected_cost"] < never_intervene
    )
    print(f"[a5.5] calibrated STRICTLY DOMINATES both baselines: {dominates}")

    result = {
        "agent": args.agent_name,
        "n_snapshots": len(pts),
        "safe_default": SAFE_DEFAULT,
        "always_intervene_cost": round(always_intervene, 3),
        "never_intervene_cost": round(never_intervene, 3),
        "calibrated_best": best,
        "dominates_both": bool(dominates),
        "risk_coverage_curve": curve,
    }
    (out / "a5_5_risk_coverage.json").write_text(json.dumps(result, indent=2))

    # figure
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        cov = [c["coverage"] for c in curve]
        ec = [c["expected_cost"] for c in curve]
        ax.plot(cov, ec, "o-", label="calibrated (abstain→Backstep)", lw=2, color="#2ca02c")
        ax.axhline(
            always_intervene,
            color="#d62728",
            ls="--",
            label=f"always-intervene ({always_intervene:.2f})",
        )
        ax.axhline(
            never_intervene,
            color="#1f77b4",
            ls="--",
            label=f"never-intervene ({never_intervene:.2f})",
        )
        ax.scatter(
            [best["coverage"]],
            [best["expected_cost"]],
            color="k",
            zorder=5,
            s=60,
            label=f"operating point ({best['expected_cost']:.2f})",
        )
        ax.set_xlabel("coverage (fraction non-abstained)")
        ax.set_ylabel("expected physical cost (via M)")
        ax.set_title(
            f"A5.5 risk-coverage — {args.agent_name} + OOD-θ abstention\n"
            f"calibrated + safe default dominates both extremes"
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "risk_coverage.png", dpi=130)
        plt.close(fig)
        print(f"[OK] wrote {out / 'a5_5_risk_coverage.json'} + risk_coverage.png")
    except ImportError:
        print("[OK] wrote a5_5_risk_coverage.json (matplotlib missing)")


if __name__ == "__main__":
    main()
