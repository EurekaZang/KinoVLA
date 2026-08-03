#!/usr/bin/env python
"""E2 three-row eval: the conflict-resolution ablation on the matched construction (no Isaac).

Open-loop, single-decision attribution on the #49-matched eval set (outputs/eval/e2/matched_eval)
for THREE agents — reports attribution accuracy + feasible-recovery rate (primitive ∈ the true
cause's feasible set) per agent:

  Row 1  B1 (strongest pure-proprio)        chance attribution → non-feasible primitive (C2ST)
  Row 2  B5-unshaped (SFT on unshaped O4)   matched scene OOD → wrong (has vision, never learned
                                            conflict-resolution: "a vision channel ≠ using it")
  Row 3  B5-conflict (SFT + #49-matched)    correct attribution → feasible primitive (Backstep)

Monotone: B1 can't-in-principle < B5-unshaped could-but-didn't-learn < B5-conflict learned-so-can.
Pure model inference + disk snapshots — no AppLauncher, no use_map, no closed loop.

Run:  KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct HF_HUB_OFFLINE=1 \
        python scripts/eval_e2_three_row.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score 95% CI for a binomial proportion (matched-eval items are independent snapshots,
    so a per-item binomial interval is correct — no lane grouping needed, unlike E1's windows)."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, center - half), 3), round(min(1.0, center + half), 3)]


def main() -> None:
    ap = argparse.ArgumentParser(description="E2 three-row conflict-resolution ablation")
    ap.add_argument("--matched-eval", default="outputs/eval/e2/matched_eval")
    ap.add_argument("--b1-model", default="outputs/eval/e2/b1_attributor/monitor")
    ap.add_argument("--b5-unshaped", default="outputs/vla/sft_latent/adapter_best")
    ap.add_argument("--b5-conflict", default="outputs/eval/e2/b5_conflict/adapter_best")
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--out", default="outputs/eval/e2/three_row.json")
    args = ap.parse_args()

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy
    from kino_vla.eval.suite_sem import evaluate_attribution, evaluate_by_regime, load_suite_sem
    from kino_vla.utils.config import load_config

    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    feasible = {c: set(p) for c, p in pcfg.recovery.feasible.to_dict().items()}

    ids = [json.loads(line)["sample_id"]
           for line in Path(args.matched_eval, "samples.jsonl").read_text().splitlines() if line]
    items = load_suite_sem(args.matched_eval, ids, ambiguity_only=True)
    print(f"[e2.3row] matched-eval items: {len(items)} "
          f"(truths: {sorted({it.attribution_truth for it in items})})", flush=True)

    rows: dict[str, dict] = {}

    # Row 1 — B1 (proprio-only). CPU model.
    b1 = ProprioBaselinePolicy.from_deployed(pcfg, tax, model_path=args.b1_model, device="cpu")
    rows["B1_proprio"] = evaluate_by_regime(
        b1, items, ambiguous_apps=set(), feasible_sets=feasible
    )["overall"]
    print(f"[e2.3row] B1: {rows['B1_proprio']}", flush=True)

    # Rows 2/3 — the two VLA agents (GPU). Same backbone, different adapter.
    import torch

    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy

    vcfg = load_config(args.config, {"route": "latent"})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, adapter in (("B5_unshaped", args.b5_unshaped), ("B5_conflict", args.b5_conflict)):
        model = KinoVLA.from_pretrained(vcfg, device=device, adapter_dir=adapter)
        model.eval()
        pol = ModelVlaPolicy(
            model, pcfg, tax, route="latent", n_images=int(vcfg.data.get("n_images", 1)),
            temperature=0.0, proprio_detail=str(vcfg.data.get("proprio_detail", "binned")),
        )
        rows[name] = evaluate_attribution(pol, items, feasible_sets=feasible)
        print(f"[e2.3row] {name}: {rows[name]}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Per-agent summary, with the O4 conflict case as the protagonist + Wilson CIs. The headline
    # number is the JOINT correct_recovery_rate (attribution-gated); feasible_recovery is demoted to
    # a diagnostic because it is NOT attribution-gated (a wrong cause can pick a coincidentally
    # feasible primitive — that is why B1 shows >0 feasible_recovery at 0.0 attribution).
    def _op(row: dict, op: str) -> dict:
        c = row["per_operator_counts"].get(op, {"n": 0, "attr_correct": 0, "joint_correct": 0})
        n_op = c["n"]
        return {
            "n": n_op,
            "attribution_acc": round(c["attr_correct"] / max(1, n_op), 3),
            "attribution_ci": _wilson(c["attr_correct"], n_op),
            "correct_recovery_rate": round(c["joint_correct"] / max(1, n_op), 3),
            "correct_recovery_ci": _wilson(c["joint_correct"], n_op),
        }

    summary = {}
    for name in rows:
        r = rows[name]
        n_all = r["n"]
        attr_correct_all = round(r["attribution_accuracy"] * n_all)
        joint_correct_all = round((r.get("correct_recovery_rate") or 0.0) * n_all)
        summary[name] = {
            "overall": {
                "n": n_all,
                "attribution_acc": round(r["attribution_accuracy"], 3),
                "attribution_ci": _wilson(attr_correct_all, n_all),
                "correct_recovery_rate": round(r.get("correct_recovery_rate") or 0.0, 3),
                "correct_recovery_ci": _wilson(joint_correct_all, n_all),
                "feasible_recovery_rate_UNGATED": round(r["feasible_recovery_rate"], 3),
            },
            "O4_conflict": _op(r, "O4_tether"),
            "O2_control": _op(r, "O2_compliance"),
        }

    out = {
        "matched_eval": args.matched_eval,
        "n_items": len(items),
        "headline": "O4_conflict",  # the load-bearing column (O2 is the non-conflict control)
        "metric_notes": {
            "correct_recovery_rate": "JOINT, attribution-gated: attribution correct AND primitive "
            "in the true cause's feasible set. The recovery number to report.",
            "feasible_recovery_rate_UNGATED": "diagnostic only — NOT attribution-gated; a wrong "
            "cause can emit a coincidentally feasible primitive, so a proprio-blind baseline "
            "scores >0 here. Do not headline.",
        },
        "rows": rows,
        "summary": summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n=== E2 three-row (matched construction) — O4 conflict is the load-bearing column ===")
    hdr = f"{'agent':<14}{'O4 attr (CI)':>22}{'O4 correct-recov (CI)':>26}{'overall attr':>14}"
    print(hdr)
    for name in ("B1_proprio", "B5_unshaped", "B5_conflict"):
        o4 = summary[name]["O4_conflict"]
        ov = summary[name]["overall"]
        aci = o4["attribution_ci"]
        jci = o4["correct_recovery_ci"]
        a = f"{o4['attribution_acc']:.2f} [{aci[0]:.2f},{aci[1]:.2f}]"
        j = f"{o4['correct_recovery_rate']:.2f} [{jci[0]:.2f},{jci[1]:.2f}]"
        print(f"{name:<14}{a:>22}{j:>26}{ov['attribution_acc']:>14.2f}")
    print(f"\nO4 conflict cases: n={summary['B1_proprio']['O4_conflict']['n']}  |  "
          f"O2 control: n={summary['B1_proprio']['O2_control']['n']}")
    print("(feasible_recovery_rate is ungated — reported as a diagnostic only; see metric_notes)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
