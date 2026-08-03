#!/usr/bin/env python
"""Freeze structured A3 successor v2 before its second untouched confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from kino_vla.eval.a3_successor_stats import acceptance_audit
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
    parser = argparse.ArgumentParser(description="Freeze A3 successor v2")
    parser.add_argument("--config", default="configs/eval/a3_successor_v2.yaml")
    parser.add_argument(
        "--out", default="outputs/eval/a3_successor_v2/frozen_method_manifest.json"
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    output = repo_path(cfg["output_dir"])
    seeds = [int(value) for value in cfg["training"]["seeds"]]
    seed_rows = {
        seed: json.loads((output / f"seed{seed}" / "per_item_cluster_dev.json").read_text())
        for seed in seeds
    }
    audit = acceptance_audit(seed_rows, cfg["acceptance"])
    if not audit["passed"]:
        raise ValueError("v2 failed its every-seed cluster-development gates")
    files = [
        Path(args.config),
        Path("kino_vla/eval/evidence_routed_v2.py"),
        Path("kino_vla/eval/material_support.py"),
        Path("scripts/a3_train_structured_router_v2.py"),
        output / "training_summary.json",
        Path("outputs/eval/a3_successor/confirmatory_eval/evaluation_manifest.json"),
        *(output / f"seed{seed}" / "router_v2.pkl" for seed in seeds),
        *(output / f"seed{seed}" / "metrics.json" for seed in seeds),
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"v2 freeze inputs missing: {missing}")
    hashes = {str(path): _sha256(path) for path in files}
    aggregate = hashlib.sha256(
        "\n".join(f"{path}:{value}" for path, value in sorted(hashes.items())).encode()
    ).hexdigest()
    result = {
        "schema_version": 2,
        "experiment": cfg["experiment"],
        "status": "frozen_before_second_confirmatory_collection",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "method_sha256": aggregate,
        "file_sha256": hashes,
        "training_seeds": seeds,
        "cluster_development_acceptance": audit,
        "acceptance": cfg["acceptance"],
        "preserved_first_confirmation_status": "failed_and_repurposed_as_v2_development",
        "preserved_first_confirmation_manifest": (
            "outputs/eval/a3_successor/confirmatory_eval/evaluation_manifest.json"
        ),
        "deployment_inputs": [
            "proprio temporal distribution summary",
            "train-fitted material prototype distances",
            "observable colour chroma",
        ],
        "forbidden_deployment_inputs": cfg["protocol"]["forbidden_deployment_inputs"],
        "confirmatory_rule": "no v2 code, model, prototype, or threshold changes after freeze",
    }
    write_json(Path(args.out), result)
    print(json.dumps({"method_sha256": aggregate, "n_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
