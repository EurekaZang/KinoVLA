#!/usr/bin/env python
"""M7 route-fidelity ablation — the spec §3 information-fidelity experiment (CLAUDE.md §6 #35).

Resolves Gap-1: the aggregate Suite-Sem attribution tied (latent 0.974 == text 0.974), so the
paper's central claim — that the *latent* Kino-Tokens route (B5) is the principled choice over
text injection — had no evidence. Root cause (investigated, see #35): the aggregate averages over
14 vision-solvable samples that saturate BOTH routes, and the text route was handed a ~35-number
ORACLE waveform serialization. This experiment isolates the decisive regime and sweeps the text
route's proprioception fidelity, so the latent route's advantage becomes measurable on the axes the
spec actually claims for it.

Arms (all Kino-SFT on the real Qwen3-VL-4B + the 413-sample real-Go2 + gpt-5.5 + filter dataset,
identical seed/epochs/split):
  vision_only  route=text   proprio_detail=none    — the vision floor (is proprio load-bearing?)
  text_scalar  route=text   proprio_detail=scalar  — realistic REFLECT-style text (means only)
  text_binned  route=text   proprio_detail=binned  — the oracle waveform serialization (B4)
  latent       route=latent                         — Kino-Tokens K=6 (B5, ours)

Three axes per arm: (1) attribution accuracy split by appearance regime (ambiguous=proprio-decided
vs solvable=vision-decided); (2) θ-grounding — the latent projector's held-out privileged-θ MAE
(the text arms have NO θ head, a structural capability gap, spec §4a); (3) prompt token cost (the
"低延迟" claim, spec §3 / §6.9).

Usage (GPU box):
    python scripts/m7_route_ablation.py --stage train --arm vision_only
    python scripts/m7_route_ablation.py --stage eval  --arm vision_only [--temp 0.0 --samples 1]
    python scripts/m7_route_ablation.py --stage combine
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

ARMS: dict[str, dict[str, str]] = {
    "vision_only": {"route": "text", "proprio_detail": "none"},
    "text_scalar": {"route": "text", "proprio_detail": "scalar"},
    "text_binned": {"route": "text", "proprio_detail": "binned"},
    "latent": {"route": "latent", "proprio_detail": "binned"},  # detail unused on the latent route
}
ABL_DIR = Path("outputs/vla/ablation")


def _overrides(arm: str) -> dict:
    a = ARMS[arm]
    return {"route": a["route"], "data.proprio_detail": a["proprio_detail"]}


# --------------------------------------------------------------------------- train
def train_arm(arm: str, config: str, dataset: str | None, epochs: int | None) -> None:
    from kino_vla.utils.config import load_config
    from kino_vla.vla.sft import train_sft

    ov = _overrides(arm)
    if epochs is not None:
        ov["train.epochs"] = epochs
    cfg = load_config(config, ov)
    dataset_dir = dataset or str(cfg.data.dataset_dir)
    out = ABL_DIR / arm
    print(f"[ablation] TRAIN arm={arm} {ARMS[arm]} -> {out}", flush=True)
    m = train_sft(cfg, dataset_dir, out)
    print(json.dumps({k: m[k] for k in ("route", "best_val_loss", "n_train", "wall_time_s")}))


# ---------------------------------------------------------------------------- eval
def _theta_mae(model, examples) -> dict | None:
    """Held-out privileged-θ MAE of the latent projector (None for the text arms = no θ head)."""
    import numpy as np
    import torch

    if model.projector is None:
        return None
    from kino_vla.tokens.features import TARGET_SCHEMA

    errs = []
    with torch.no_grad():
        for ex in examples:
            w = torch.tensor(np.asarray(ex.proprio_window), dtype=torch.float32)
            w = w.unsqueeze(0).to(model.device)
            _soft, theta = model.projector(w)
            pred = theta.squeeze(0).float().cpu().numpy()
            errs.append(np.abs(pred - np.asarray(ex.target_theta, dtype=np.float64)))
    mae = np.mean(errs, axis=0)
    return {name: round(float(mae[i]), 4) for i, name in enumerate(TARGET_SCHEMA)}


def _mean_prompt_tokens(model, examples, n_images: int) -> float:
    """Mean #tokens in the (inference) prompt for this arm — the token-cost axis (spec §3).

    Each VlaExample's ``messages`` were already built with this arm's route + proprio fidelity
    (``load_split_for_eval`` reads ``data.proprio_detail``), so the token count is exactly what the
    policy sends. ``build_inputs`` with no target gives the inference-prompt length.
    """
    lengths = []
    for ex in examples:
        images = list(ex.rgb[-n_images:]) if ex.rgb.size else []
        inp = model.build_inputs(ex.messages, images, proprio_window=ex.proprio_window)
        lengths.append(int(inp.input_ids.shape[1]))
    return round(statistics.mean(lengths), 1)


def eval_arm(arm: str, config: str, dataset: str | None, temp: float, samples: int) -> dict:
    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.suite_sem import (
        ambiguous_appearances,
        compare_vla_vs_fsm,
        evaluate_by_regime,
        load_suite_sem,
    )
    from kino_vla.utils.config import load_config
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy
    from kino_vla.vla.sft import load_split_for_eval

    cfg = load_config(config, _overrides(arm))
    pcfg = load_config("data/hindsight.yaml")
    dataset_dir = dataset or str(cfg.data.dataset_dir)
    tax = FailureTaxonomy(pcfg)
    n_images = int(cfg.data.get("n_images", 1))
    route = ARMS[arm]["route"]
    detail = ARMS[arm]["proprio_detail"]

    adapter = ABL_DIR / arm / "adapter_best"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KinoVLA.from_pretrained(cfg, device=device, adapter_dir=str(adapter))
    model.eval()
    policy = ModelVlaPolicy(
        model, pcfg, tax, route=route, n_images=n_images, temperature=temp, proprio_detail=detail
    )

    split = load_split_for_eval(cfg, dataset_dir)
    sem_examples = [e for e in split.test if e.ambiguity_pair is not None]
    ids = [e.sample_id for e in split.test]
    items = load_suite_sem(dataset_dir, ids, ambiguity_only=True)
    amb_apps = ambiguous_appearances(items)
    feasible = {k: set(v) for k, v in tax._feasible.items()}
    train_cats = [e.attribution_truth for e in split.train]

    t0 = time.perf_counter()
    by_regime = evaluate_by_regime(policy, items, ambiguous_apps=amb_apps, feasible_sets=feasible)
    cmp = compare_vla_vs_fsm(
        policy, items, cfg=pcfg, train_categories=train_cats, feasible_sets=feasible
    )
    # Sampled robustness on the decisive (ambiguous) subset.
    sampled = None
    if samples > 1 and temp > 0:
        amb_items = [it for it in items if it.snapshot.appearance_class in amb_apps]
        from kino_vla.eval.suite_sem import evaluate_attribution

        runs = [
            evaluate_attribution(policy, amb_items, feasible_sets=feasible) for _ in range(samples)
        ]
        accs = [r["attribution_accuracy"] for r in runs]
        sampled = {
            "temperature": temp,
            "n_samples": samples,
            "ambiguous_attr_acc_mean": round(statistics.mean(accs), 4),
            "ambiguous_attr_acc_std": round(statistics.pstdev(accs), 4),
            "runs": [round(a, 4) for a in accs],
        }

    result = {
        "arm": arm,
        "route": route,
        "proprio_detail": detail,
        "temperature": temp,
        "fsm_majority_category": cmp["fsm_majority_category"],
        "fsm_attr_acc": round(cmp["fsm_baseline"]["attribution_accuracy"], 4),
        "n_ambiguous": by_regime["ambiguous"]["n"],
        "n_solvable": by_regime["solvable"]["n"],
        "attr_overall": round(by_regime["overall"]["attribution_accuracy"], 4),
        "attr_ambiguous": round(by_regime["ambiguous"]["attribution_accuracy"], 4),
        "attr_solvable": round(by_regime["solvable"]["attribution_accuracy"], 4),
        "feasible_ambiguous": round(by_regime["ambiguous"]["feasible_recovery_rate"], 4),
        "parse_rate": round(by_regime["overall"]["parse_rate"], 4),
        "per_operator_ambiguous": {
            k: round(v, 3) for k, v in by_regime["ambiguous"]["per_operator_accuracy"].items()
        },
        "theta_mae": _theta_mae(model, sem_examples),
        "mean_prompt_tokens": _mean_prompt_tokens(model, sem_examples, n_images),
        "sampled": sampled,
        "eval_wall_s": round(time.perf_counter() - t0, 1),
    }
    out = ABL_DIR / f"{arm}_eval_t{str(temp).replace('.', '')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# ------------------------------------------------------------------------- combine
def combine(temp: float) -> None:
    tag = f"t{str(temp).replace('.', '')}"
    rows = []
    for arm in ARMS:
        p = ABL_DIR / f"{arm}_eval_{tag}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if not rows:
        print(f"no eval files for {tag}")
        return
    vis = next((r for r in rows if r["arm"] == "vision_only"), None)
    base_tokens = vis["mean_prompt_tokens"] if vis else 0.0
    lines = [
        f"# M7 Route-Fidelity Ablation (Gap-1) — temp {temp}",
        "",
        "All arms: Kino-SFT on Qwen3-VL-4B, identical seed/epochs/split, real-Go2 dataset.",
        "`ambiguous` = the appearance-ambiguous regime (ice_sheet/solid_ground; only proprio can "
        "attribute). `solvable` = vision-decided (mud/adhesive).",
        "",
        "| arm | proprio | overall | **ambiguous** | solvable | Δtokens | θ-grounded |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        r = next((x for x in rows if x["arm"] == arm), None)
        if r is None:
            continue
        dtok = r["mean_prompt_tokens"] - base_tokens
        theta = "yes" if r["theta_mae"] else "—"
        lines.append(
            f"| {arm} | {r['proprio_detail'] if r['route'] == 'text' else 'Kino-Tokens(K=6)'} | "
            f"{r['attr_overall']:.3f} | **{r['attr_ambiguous']:.3f}** | {r['attr_solvable']:.3f} | "
            f"+{dtok:.0f} | {theta} |"
        )
    lines += ["", f"FSM majority baseline (constant): {rows[0]['fsm_attr_acc']:.3f}", ""]
    lat = next((r for r in rows if r["arm"] == "latent"), None)
    if lat and lat["theta_mae"]:
        lines.append(f"Latent projector held-out θ-MAE: {lat['theta_mae']}")
    out_md = ABL_DIR / f"summary_{tag}.md"
    out_md.write_text("\n".join(lines))
    (ABL_DIR / f"summary_{tag}.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M7 route-fidelity ablation (Gap-1)")
    ap.add_argument("--stage", required=True, choices=["train", "eval", "combine"])
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=1)
    args = ap.parse_args()

    ABL_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "train":
        train_arm(args.arm, args.config, args.dataset, args.epochs)
    elif args.stage == "eval":
        eval_arm(args.arm, args.config, args.dataset, args.temp, args.samples)
    else:
        combine(args.temp)


if __name__ == "__main__":
    main()
