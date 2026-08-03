#!/usr/bin/env python
"""Freeze the exact A3 successor method before confirmatory corpus collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kino_vla.eval.a7_ablation import load_yaml, repo_path, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze A3 successor method")
    parser.add_argument("--config", default="configs/eval/a3_successor.yaml")
    parser.add_argument("--audit", default="outputs/eval/a3_successor/publication_audit.json")
    parser.add_argument(
        "--out", default="outputs/eval/a3_successor/frozen_method_manifest.json"
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text())
    if not bool(audit["acceptance"]["passed"]):
        raise ValueError("cannot freeze a method that failed the every-seed development gates")
    if audit["status"] != "inspected_method_development_not_confirmatory":
        raise ValueError("freeze expects an explicitly non-confirmatory development audit")

    method_files = [
        Path(args.config),
        Path("kino_vla/eval/evidence_routed.py"),
        Path("kino_vla/eval/material_support.py"),
        Path("scripts/a3_train_evidence_router.py"),
        repo_path(cfg["feature_cache"]),
        repo_path(cfg["feature_cache"]).with_suffix(".json"),
        audit_path,
    ]
    checkpoint_files = [
        repo_path(cfg["output_dir"]) / f"seed{seed}" / "router.pt"
        for seed in cfg["training"]["seeds"]
    ]
    source_files = [
        repo_path(cfg["sources"]["successor_reverse_train"]) / "samples.jsonl",
        repo_path(cfg["sources"]["successor_reverse_train"]) / "frames.npz",
        repo_path(cfg["sources"]["successor_reverse_conflict_predictions"]),
    ]
    paths = [*method_files, *checkpoint_files, *source_files]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"freeze inputs missing: {missing}")
    hashes = {str(path): _sha256(path) for path in paths}
    aggregate = hashlib.sha256(
        "\n".join(f"{path}:{digest}" for path, digest in sorted(hashes.items())).encode()
    ).hexdigest()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": cfg["experiment"],
        "status": "frozen_before_confirmatory_collection",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "method_sha256": aggregate,
        "file_sha256": hashes,
        "training_seeds": [int(seed) for seed in cfg["training"]["seeds"]],
        "development_acceptance_passed": True,
        "acceptance": cfg["acceptance"],
        "deployment_inputs": [
            "RGB-derived frozen CLIP and train-prototype material features",
            "standardized proprioception window",
            "frozen bidirectional-conflict VLA logits",
        ],
        "forbidden_deployment_inputs": cfg["protocol"]["forbidden_deployment_inputs"],
        "confirmatory_rule": "no method, checkpoint, prototype, or threshold changes after freeze",
    }
    write_json(Path(args.out), result)
    print(json.dumps({"method_sha256": aggregate, "n_files": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
