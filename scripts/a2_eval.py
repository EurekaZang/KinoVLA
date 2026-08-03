#!/usr/bin/env python
"""A2 — headline three-row, hardened + extended roster (Paper-A §4 A2; serves C2, C5).

Open-loop single-decision attribution on the FROZEN A0.3 matched corpus (matched-O4 n=120 +
matched-O2 n=120), sliced by the pre-registered A0.4 appearance split (train vs held-out test).
No Isaac at eval time — the snapshots are frozen (content hash in the manifest). Agents:

  B1            strongest pure-proprio (LearnedMonitor attributor)  [CPU]   C1/C5 negative arm
  B5-unshaped   Qwen3-VL-4B+LoRA, trained on unshaped ops only      [GPU]   "has vision != uses it"
  B5-conflict   + 20 matched conflict samples (E2 headline model)   [GPU]   learned the operation
  B-V           B5-conflict with the Kino-Token channel masked      [GPU]   vision-only mirror arm
  B-T           text-summary-injection VLM (REFLECT scalar schema)  [GPU]   latent-vs-text lever
  B-F           CLIP feats + proprio-encoder feats -> MLP           [CPU]   honest no-VLM baseline
  zero-shot     off-the-shelf Qwen3-VL, no LoRA, text summary        [GPU]   context row

Headline (R6): the BALANCED (both-directions) attribution + attribution-gated correct-recovery over
the matched pair, with Wilson 95% CIs and paired McNemar on the shared snapshots. A proprio-only
agent is capped at 0.5 balanced by construction (matched pair C2ST=0.5).

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct \
        ~/miniconda3/envs/kinovla/bin/python scripts/a2_eval.py --agents B1,B5-unshaped,B5-conflict
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

CORE = ["B1", "B5-unshaped", "B5-conflict", "B-V"]
ROSTER_ORDER = ["B1", "B-F", "zero-shot", "B-T", "B-V", "B5-unshaped", "B5-conflict"]


def main() -> None:
    ap = argparse.ArgumentParser(description="A2 headline three-row + extended roster")
    ap.add_argument("--corpus", default="outputs/eval/a0/corpus")
    ap.add_argument("--agents", default=",".join(CORE), help="comma list of agent ids")
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--b1-model", default="outputs/eval/e2/b1_attributor/monitor")
    ap.add_argument("--b5-unshaped", default="outputs/vla/sft_latent/adapter_best")
    ap.add_argument("--b5-conflict", default="outputs/eval/e2/b5_conflict/adapter_best")
    ap.add_argument("--bt-adapter", default="outputs/eval/a2/b_text/adapter_best")
    ap.add_argument("--bf-model", default="outputs/eval/a2/b_fusion/bf.pt")
    ap.add_argument("--out", default="outputs/eval/a2")
    ap.add_argument("--merge", action="store_true",
                    help="load existing per_item.json and ADD the new agents (incremental roster)")
    args = ap.parse_args()
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.a2_headline import (
        ItemResult,
        aggregate,
        confusion,
        eval_policy,
        load_matched_corpus,
        mcnemar,
    )
    from kino_vla.utils.config import load_config

    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    items = load_matched_corpus(args.corpus)
    n_o4 = sum(1 for it in items if it.operator == "O4_tether")
    n_o2 = sum(1 for it in items if it.operator == "O2_compliance")
    print(f"[a2] matched corpus: {len(items)} items (O4={n_o4}, O2={n_o2})", flush=True)
    print(f"[a2] appearance split: "
          f"train={sum(1 for it in items if it.appearance_split=='train')} "
          f"test={sum(1 for it in items if it.appearance_split=='test')}", flush=True)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    per_item: dict[str, list] = {}
    summary: dict[str, dict] = {}

    def _summarize(name: str, res: list) -> None:
        summary[name] = aggregate(res)
        summary[name]["_confusion_O4"] = confusion(res, "O4_tether")
        summary[name]["_confusion_O2"] = confusion(res, "O2_compliance")

    if args.merge and (outdir / "per_item.json").exists():
        per_item = json.loads((outdir / "per_item.json").read_text())
        for name, rows in per_item.items():  # rebuild summaries for previously-run agents
            _summarize(name, [ItemResult(**r) for r in rows])
        agents = [a for a in agents if a not in per_item]  # only run the newly-requested agents
        print(f"[a2] merge mode: kept {list(per_item)}; running {agents}", flush=True)

    def _run(name: str, policy: object) -> None:
        res = eval_policy(policy, items)
        per_item[name] = [asdict(r) for r in res]
        _summarize(name, res)
        o4 = summary[name]["O4"]["all"]
        bal = summary[name]["balanced"]["all"]
        tb = summary[name]["balanced"]["test_appearance"].get("attribution_balanced")
        print(f"[a2] {name:12s} O4 attr={o4['attribution_acc']:.3f}/O2={bal['o2_attr']:.3f} "
              f"BALANCED attr={bal['attribution_balanced']:.3f}{bal['attribution_balanced_ci']} "
              f"corr-recov={bal['correct_recovery_balanced']:.3f}"
              f"{bal['correct_recovery_balanced_ci']} | test-split bal={tb}", flush=True)

    # ---- B1 (CPU, proprio-only) ------------------------------------------------
    if "B1" in agents:
        from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy

        b1 = ProprioBaselinePolicy.from_deployed(pcfg, tax, model_path=args.b1_model, device="cpu")
        _run("B1", b1)

    # ---- B-F (CPU, CLIP+proprio fusion MLP) ------------------------------------
    if "B-F" in agents:
        from kino_vla.eval.fusion_baseline import FusionBaselinePolicy

        bf = FusionBaselinePolicy.load(args.bf_model, pcfg, tax)
        _run("B-F", bf)

    # ---- VLA agents (GPU) ------------------------------------------------------
    vla_ids = {"B5-unshaped", "B5-conflict", "B-V", "B-T", "zero-shot"}
    vla_agents = [a for a in agents if a in vla_ids]
    if vla_agents:
        import torch

        from kino_vla.vla.model import KinoVLA
        from kino_vla.vla.planner import ModelVlaPolicy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_images = None

        def _load(route: str, adapter: str | None):
            vcfg = load_config(args.config, {"route": route})
            m = KinoVLA.from_pretrained(vcfg, device=device, adapter_dir=adapter)
            m.eval()
            ni = int(vcfg.data.get("n_images", 1))
            return m, ni, str(vcfg.data.get("proprio_detail", "binned"))

        # B5-unshaped
        if "B5-unshaped" in agents:
            model, n_images, pdet = _load("latent", args.b5_unshaped)
            _run("B5-unshaped", ModelVlaPolicy(model, pcfg, tax, route="latent",
                 n_images=n_images, temperature=0.0, proprio_detail=pdet))
            del model
            torch.cuda.empty_cache()

        # B5-conflict + B-V (share one loaded model)
        if "B5-conflict" in agents or "B-V" in agents:
            model, n_images, pdet = _load("latent", args.b5_conflict)
            if "B5-conflict" in agents:
                _run("B5-conflict", ModelVlaPolicy(model, pcfg, tax, route="latent",
                     n_images=n_images, temperature=0.0, proprio_detail=pdet))
            if "B-V" in agents:
                _run("B-V", ModelVlaPolicy(model, pcfg, tax, route="latent", n_images=n_images,
                     temperature=0.0, proprio_detail=pdet, mask_proprio=True))
            del model
            torch.cuda.empty_cache()

        # B-T (text route, REFLECT scalar summary; needs a text-route adapter)
        if "B-T" in agents:
            model, n_images, _ = _load("text", args.bt_adapter)
            _run("B-T", ModelVlaPolicy(model, pcfg, tax, route="text", n_images=n_images,
                 temperature=0.0, proprio_detail="scalar"))
            del model
            torch.cuda.empty_cache()

        # zero-shot off-the-shelf VLM (no adapter, text summary)
        if "zero-shot" in agents:
            model, n_images, _ = _load("text", None)
            _run("zero-shot", ModelVlaPolicy(model, pcfg, tax, route="text", n_images=n_images,
                 temperature=0.0, proprio_detail="scalar"))
            del model
            torch.cuda.empty_cache()

    # ---- paired McNemar between adjacent rows --------------------------------
    # Two panels: (a) the O4-conflict column (the E2 comparison), and (b) the FULL matched set
    # (240 items) on the JOINT correct-recovery — the un-gameable panel where a constant predictor
    # (right on one operator, wrong on the other) cannot inflate the score.
    order = [a for a in ROSTER_ORDER if a in per_item]
    o4_mask = [it.operator == "O4_tether" for it in items]

    def _vec(name: str, field: str, mask: list[bool] | None = None) -> list[bool]:
        rows = per_item[name]
        if mask is None:
            return [bool(r[field]) for r in rows]
        return [bool(r[field]) for r, keep in zip(rows, mask, strict=True) if keep]

    pairwise = {}
    for lo, hi in zip(order, order[1:], strict=False):
        pairwise[f"{lo}__vs__{hi}"] = {
            "O4_attribution": mcnemar(_vec(lo, "attr_ok", o4_mask), _vec(hi, "attr_ok", o4_mask)),
            "matched_correct_recovery": mcnemar(_vec(lo, "joint_ok"), _vec(hi, "joint_ok")),
        }

    manifest = json.loads(Path(args.corpus, "frozen_manifest.json").read_text())
    out = {
        "corpus": args.corpus,
        "corpus_hash_sha256": manifest.get("corpus_hash_sha256"),
        "commit": manifest.get("commit"),
        "n_items": len(items),
        "n_O4_conflict": n_o4,
        "n_O2_control": n_o2,
        "headline_column": "O4",
        "headline_metric": "correct_recovery_rate (attribution-gated, R6)",
        "agents": order,
        "summary": summary,
        "mcnemar_O4_conflict": pairwise,
        "metric_notes": {
            "correct_recovery_rate": "JOINT, attribution-gated: attribution == truth AND primitive "
            "in the true cause's A0.5 admissible set. The headline recovery number.",
            "feasible_recovery_rate_UNGATED": "diagnostic only — not attribution-gated.",
            "appearance_split": "train/test are the A0.4 pre-registered appearance split; the "
            "test column is held-out appearances the conflict model never trained on "
            "(the shortcut-killer).",
        },
    }
    (outdir / "headline.json").write_text(json.dumps(out, indent=2))
    (outdir / "per_item.json").write_text(json.dumps(per_item, indent=2))

    # ---- printed table ---------------------------------------------------------
    print("\n=== A2 headline — BALANCED (both-directions) attribution over the matched pair ===")
    print("(a proprio-only agent is capped at 0.50 by construction — matched pair C2ST=0.5)")
    hdr = (f"{'agent':<13}{'O4':>6}{'O2':>6}{'balanced attr [CI]':>24}"
           f"{'bal corr-recov [CI]':>24}{'test-split bal':>16}")
    print(hdr)
    for name in order:
        b = summary[name]["balanced"]["all"]
        bt = summary[name]["balanced"]["test_appearance"]
        ba = f"{b['attribution_balanced']:.2f}{b['attribution_balanced_ci']}"
        bc = f"{b['correct_recovery_balanced']:.2f}{b['correct_recovery_balanced_ci']}"
        tb = f"{bt.get('attribution_balanced', float('nan')):.2f}"
        print(f"{name:<13}{b['o4_attr']:>6.2f}{b['o2_attr']:>6.2f}{ba:>24}{bc:>24}{tb:>16}")
    print("\nMcNemar (adjacent rows):")
    print(f"  {'pair':<26}{'O4-attr (b,c,p)':>26}{'matched corr-recov (b,c,p)':>32}")
    for k, v in pairwise.items():
        m4, mm = v["O4_attribution"], v["matched_correct_recovery"]
        s4 = f"{m4['b_lo_right_hi_wrong']},{m4['c_lo_wrong_hi_right']},p={m4['p_exact_two_sided']}"
        sm = (f"{mm['b_lo_right_hi_wrong']},{mm['c_lo_wrong_hi_right']},"
              f"p={mm['p_exact_two_sided']} sig={mm['significant_05']}")
        print(f"  {k:<26}{s4:>26}{sm:>32}")
    print(f"\nwrote {outdir/'headline.json'} and {outdir/'per_item.json'}")


if __name__ == "__main__":
    main()
