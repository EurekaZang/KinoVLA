#!/usr/bin/env python
"""Evaluate frozen structured-router v2 checkpoints on the second confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.eval.fusion_baseline import FusionBaselinePolicy
from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy
from kino_vla.utils.config import load_config
from kino_vla.vla.dataset_build import _snapshot_from_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    for value, expected in manifest["file_sha256"].items():
        path = Path(value)
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"frozen v2 method changed: {path}")
    return manifest


def _sub(record: dict[str, Any]) -> str:
    if record.get("a3_direction") in {"looks_safe", "reverse"}:
        return str(record["a3_direction"])
    if record["snapshot"]["operator_name"] == "O8_invisible_collider":
        return "O8"
    return "other"


def _truth(record: dict[str, Any]) -> str:
    if record["taxonomy_cell"] == "T5" or record.get("a3_direction") == "reverse":
        return "nominal"
    return str(record["ground_truth"]["category"])


def _row(record: dict[str, Any], attribution: str | None, **extra: Any) -> dict[str, Any]:
    truth = _truth(record)
    return {
        "sid": str(record["sample_id"]),
        "cell": str(record["taxonomy_cell"]),
        "t3_sub": _sub(record),
        "truth": truth,
        "appearance_id": str(record["appearance_id"]),
        "appearance_split": str(record["appearance_split"]),
        "attribution": attribution,
        "attr_ok": attribution == truth,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen structured-router v2 evaluation")
    parser.add_argument(
        "--corpora",
        default=(
            "outputs/eval/a3_successor_v2/confirmatory_main,"
            "outputs/eval/a3_successor_v2/confirmatory_t3"
        ),
    )
    parser.add_argument(
        "--manifest", default="outputs/eval/a3_successor_v2/frozen_method_manifest.json"
    )
    parser.add_argument(
        "--out", default="outputs/eval/a3_successor_v2/confirmatory_eval"
    )
    parser.add_argument("--b1-model", default="outputs/eval/e2/b1_attributor/monitor")
    parser.add_argument("--bf-model", default="outputs/eval/a2/b_fusion/bf.pt")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = _verify(manifest_path)
    records = []
    arrays = {}
    corpus_evidence = {}
    for value in args.corpora.split(","):
        directory = Path(value)
        source = [
            json.loads(line)
            for line in (directory / "samples.jsonl").read_text().splitlines()
            if line
        ]
        archive = np.load(directory / "frames.npz")
        records.extend(source)
        arrays.update({key: np.asarray(archive[key]) for key in archive.files})
        corpus_evidence[str(directory)] = {
            "n": len(source),
            "samples_sha256": _sha256(directory / "samples.jsonl"),
            "frames_sha256": _sha256(directory / "frames.npz"),
        }
    rgbs = [arrays[f"{record['sample_id']}__rgb"][-1] for record in records]
    proprios = [arrays[f"{record['sample_id']}__proprio"] for record in records]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for seed in manifest["training_seeds"]:
        model_path = Path(f"outputs/eval/a3_successor_v2/seed{seed}/router_v2.pkl")
        with model_path.open("rb") as handle:
            policy = pickle.load(handle)
        predictions, p_intervene, p_material = policy.predict_arrays(rgbs, proprios)
        scored = [
            _row(
                record,
                prediction,
                p_intervene=float(intervention),
                p_material_regime=float(material),
            )
            for record, prediction, intervention, material in zip(
                records, predictions, p_intervene, p_material, strict=True
            )
        ]
        (output / f"per_item_seed{seed}.json").write_text(json.dumps(scored, indent=2) + "\n")

    prompt_cfg = load_config("data/hindsight.yaml")
    b1 = ProprioBaselinePolicy.from_deployed(
        prompt_cfg,
        FailureTaxonomy(prompt_cfg),
        model_path=args.b1_model,
        device="cpu",
    )
    b1_rows = []
    snapshots = []
    for record in records:
        sid = str(record["sample_id"])
        snapshot = _snapshot_from_record(
            record,
            {kind: arrays[f"{sid}__{kind}"] for kind in ("rgb", "depth", "proprio")},
        )
        snapshots.append(snapshot)
        decision = b1.decide(snapshot)
        b1_rows.append(_row(record, decision.attribution if decision.ok else None))
    (output / "per_item_B1.json").write_text(json.dumps(b1_rows, indent=2) + "\n")
    bf = FusionBaselinePolicy.load(
        args.bf_model, prompt_cfg, FailureTaxonomy(prompt_cfg)
    )
    bf_rows = []
    for record, snapshot in zip(records, snapshots, strict=True):
        decision = bf.decide(snapshot)
        bf_rows.append(_row(record, decision.attribution if decision.ok else None))
    (output / "per_item_B_F.json").write_text(json.dumps(bf_rows, indent=2) + "\n")
    evaluation = {
        "schema_version": 2,
        "status": "untouched_second_confirmatory_evaluation",
        "method_sha256": manifest["method_sha256"],
        "method_unchanged": True,
        "n": len(records),
        "n_training_seeds": len(manifest["training_seeds"]),
        "corpus_evidence": corpus_evidence,
        "manifest_sha256": _sha256(manifest_path),
        "no_fit_or_threshold_selection_on_confirmatory": True,
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    print(json.dumps({"n": len(records), "seeds": manifest["training_seeds"]}, indent=2))


if __name__ == "__main__":
    main()
