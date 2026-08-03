#!/usr/bin/env python
"""A5.3 — compositional stacking eval (Paper-A §4 A5.3, serves C4).

Evaluates the agent on the A5.3 composed snapshots (two concurrent causes each). The agent emits ONE
attribution; on a composition the metrics are:
  - PER-FACTOR RECALL: did the attribution name EITHER of the two true factors?
  - COMPOSED-ADMISSIBLE: is the chosen primitive ∈ the UNION admissible set of the two factors?
  - exact 2-factor match is impossible for a single-attribution agent (reported as 0).

A positive (names ≥1 factor + admissible primitive) ⇒ attribution generalizes across
operator COMPOSITIONS (the agent need not solve both factors; recognizing one + an admissible
primitive is the generalization claim). Offline (model forward on the 16 frozen composed snapshots).

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 KINOVLA_MODEL_ID=… \\
        ~/miniconda3/envs/kinovla/bin/python scripts/a5_3_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="A5.3 composition eval")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--adapter", default="outputs/eval/a3/b5_conflict_bi/adapter_best")
    ap.add_argument("--corpus", default="outputs/eval/a5/corpus_composition")
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--out", default="outputs/eval/a5/a5_3_composition.json")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla.dataset_build import _snapshot_from_record
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy

    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    vcfg = load_config(args.config)
    ni = int(vcfg.data.get("n_images", 1))
    pdet = str(vcfg.data.get("proprio_detail", "binned"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = KinoVLA.from_pretrained(vcfg, device=dev, adapter_dir=args.adapter)
    model.eval()
    pol = ModelVlaPolicy(
        model, pcfg, tax, route="latent", n_images=ni, temperature=0.0, proprio_detail=pdet
    )

    from pathlib import Path

    cdir = Path(args.corpus)
    lines = (cdir / "samples.jsonl").read_text().splitlines()
    rows = [json.loads(ln) for ln in lines if ln]
    npz = np.load(cdir / "frames.npz")
    out_rows = []
    for r in rows:
        sid = r["sample_id"]
        if f"{sid}__rgb" not in npz:
            continue
        frames = {k: npz[f"{sid}__{k}"] for k in ("rgb", "depth", "proprio")}
        snap = _snapshot_from_record(r, frames)
        dec = pol.decide(snap)
        parsed = bool(dec.ok and dec.annotation is not None)
        attr = dec.attribution if parsed else None
        prim = dec.primitive_name if parsed else None
        factors = set(r["factors"])
        admissible = set(r["admissible_recovery_set"])
        out_rows.append(
            {
                "sid": sid,
                "stack": r["operator_name_composed"],
                "factors": sorted(factors),
                "attribution": attr,
                "primitive": prim,
                "parsed": parsed,
                "named_a_factor": parsed and attr in factors,
                "prim_in_composed_admissible": parsed and prim in admissible,
            }
        )
    n = len(out_rows)
    nf = sum(1 for x in out_rows if x["named_a_factor"])
    np_ = sum(1 for x in out_rows if x["prim_in_composed_admissible"])
    result = {
        "adapter": args.adapter,
        "n": n,
        "per_factor_recall": round(nf / max(1, n), 3),  # named ≥1 of the 2 factors
        "composed_admissible_rate": round(np_ / max(1, n), 3),
        "rows": out_rows,
    }
    (REPO_ROOT / args.out).write_text(json.dumps(result, indent=2))
    print(
        f"[a5.3] {n} composed snapshots | per-factor recall {nf}/{n} ({nf / max(1, n):.2f}) | "
        f"primitive ∈ composed-admissible {np_}/{n} ({np_ / max(1, n):.2f})",
        flush=True,
    )
    for x in out_rows:
        hit = int(x["named_a_factor"])
        adm = int(x["prim_in_composed_admissible"])
        print(
            f"  {x['stack']:20} factors={x['factors']} attr={x['attribution']} "
            f"prim={x['primitive']} hit={hit} adm={adm}"
        )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
