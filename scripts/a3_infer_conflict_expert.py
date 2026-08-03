#!/usr/bin/env python
"""Run the frozen bidirectional-conflict VLA on an additional, non-test A3 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_key(snapshot: Any, *, cfg: Any, n_images: int, proprio_detail: str) -> str:
    """Hash every deterministic input consumed by the latent ModelVlaPolicy."""
    from kino_vla.vla.prompt import build_messages, context_from_snapshot

    context = context_from_snapshot(
        snapshot,
        route="latent",
        reveal_appearance=False,
        proprio_detail=proprio_detail,
        map_note="",
    )
    messages = build_messages(context, cfg, route="latent", n_images=n_images)
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True, default=str).encode())
    for image in list(snapshot.rgb[-n_images:]) if snapshot.rgb.size else []:
        array = np.ascontiguousarray(image)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    proprio = np.ascontiguousarray(snapshot.proprio_window)
    digest.update(str((proprio.shape, proprio.dtype)).encode())
    digest.update(proprio.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen conflict-expert inference")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--adapter", default="outputs/eval/a3/b5_conflict_bi/adapter_best")
    parser.add_argument("--config", default="vla/sft.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.utils.config import load_config
    from kino_vla.vla.dataset_build import _snapshot_from_record
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy

    corpus = Path(args.corpus)
    records = [
        json.loads(line)
        for line in (corpus / "samples.jsonl").read_text().splitlines()
        if line
    ]
    frames = np.load(corpus / "frames.npz")
    prompt_cfg = load_config("data/hindsight.yaml")
    vla_cfg = load_config(args.config, {"route": "latent"})
    taxonomy = FailureTaxonomy(prompt_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KinoVLA.from_pretrained(vla_cfg, device=device, adapter_dir=args.adapter)
    model.eval()
    n_images = int(vla_cfg.data.get("n_images", 1))
    proprio_detail = str(vla_cfg.data.get("proprio_detail", "binned"))
    policy = ModelVlaPolicy(
        model,
        prompt_cfg,
        taxonomy,
        route="latent",
        n_images=n_images,
        temperature=0.0,
        proprio_detail=proprio_detail,
    )
    results = []
    decision_cache: dict[str, Any] = {}
    for index, rec in enumerate(records, start=1):
        sid = str(rec["sample_id"])
        snapshot = _snapshot_from_record(
            rec,
            {kind: frames[f"{sid}__{kind}"] for kind in ("rgb", "depth", "proprio")},
        )
        key = _input_key(
            snapshot,
            cfg=prompt_cfg,
            n_images=n_images,
            proprio_detail=proprio_detail,
        )
        if key not in decision_cache:
            decision_cache[key] = policy.decide(snapshot)
        decision = decision_cache[key]
        parsed = bool(decision.ok and decision.annotation is not None)
        results.append(
            {
                "sid": sid,
                "attribution": decision.attribution if parsed else None,
                "primitive": decision.primitive_name if parsed else None,
                "parsed": parsed,
            }
        )
        print(
            f"[conflict-expert] {index}/{len(records)} {sid}: "
            f"{results[-1]['attribution']}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    meta = {
        "schema_version": 1,
        "frozen_expert": str(args.adapter),
        "corpus": str(corpus),
        "n": len(results),
        "n_unique_model_inputs": len(decision_cache),
        "device": device,
        "input_sha256": {
            "samples": _sha256(corpus / "samples.jsonl"),
            "frames": _sha256(corpus / "frames.npz"),
        },
        "output_sha256": _sha256(out),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
