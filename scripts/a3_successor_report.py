#!/usr/bin/env python
"""Build cluster-aware A3 successor tables, tests, and acceptance audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.eval.a3_successor_stats import (
    CELLS,
    acceptance_audit,
    accuracy_metrics,
    appearance_cluster_bootstrap,
    paired_cluster_test,
)
from kino_vla.eval.a7_ablation import load_yaml, repo_path, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def _mean_sd(seed_metrics: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float]:
    values = []
    for metrics in seed_metrics:
        value: Any = metrics
        for key in path:
            value = value[key]
        values.append(float(value))
    return {
        "mean": round(float(np.mean(values)), 6),
        "sd": round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0,
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# A3 Successor — cluster-aware development audit",
        "",
        f"Status: **{result['status']}**. The current appearance-test split was inspected during "
        "method development and is not labelled confirmatory.",
        "",
        "## Frozen-battery attribution accuracy",
        "",
        "| Method | T1 | T2 | T3 | T4 | T5 | Macro |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ours = result["five_seed_summary"]
    values = [ours["per_cell"][cell]["mean"] for cell in CELLS]
    lines.append(
        "| Ours (5-seed mean) | "
        + " | ".join(f"{value:.3f}" for value in values)
        + f" | {ours['macro']['mean']:.3f} |"
    )
    for name, item in result["baselines"].items():
        metrics = item["metrics"]
        values = [metrics["per_cell"][cell] for cell in CELLS]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{value:.3f}" for value in values)
            + f" | {metrics['macro']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Hard gates",
            "",
            f"Every-seed acceptance: **{result['acceptance']['passed']}** "
            f"({result['acceptance']['n_seeds']} seeds).",
            "",
            "## Cluster-aware inference",
            "",
            f"Bootstrap unit: appearance cluster within cell; "
            f"{result['cluster_bootstrap']['n_clusters']} clusters, "
            f"{result['cluster_bootstrap']['reps']} replicates. Snapshot-level independence is "
            "not assumed.",
            "",
            "| Comparator | Δ cluster-balanced | 95% paired bootstrap CI | exact sign p |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, item in result["baselines"].items():
        paired = item["paired_vs_ours"]
        ci = paired["paired_bootstrap_ci95"]
        lines.append(
            f"| {name} | {paired['ours_minus_baseline_cluster_balanced']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {paired['exact_sign_p_two_sided']:.4g} |"
        )
    lines.extend(
        [
            "",
            "The exact sign test operates on paired appearance-cluster accuracies and discards "
            "ties; it does not treat repeated deterministic seed snapshots as independent trials.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 successor cluster-aware report")
    parser.add_argument("--config", default="configs/eval/a3_successor.yaml")
    parser.add_argument("--split", default="appearance_test")
    parser.add_argument("--out", default="outputs/eval/a3_successor/publication_audit.json")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seeds = [int(value) for value in cfg["training"]["seeds"]]
    output_dir = repo_path(cfg["output_dir"])
    seed_paths = {
        seed: output_dir / f"seed{seed}" / f"per_item_{args.split}.json" for seed in seeds
    }
    seed_rows = {seed: _load(path) for seed, path in seed_paths.items()}
    seed_metrics = [accuracy_metrics(seed_rows[seed]) for seed in seeds]
    acceptance = acceptance_audit(seed_rows, cfg["acceptance"])
    bootstrap = appearance_cluster_bootstrap(
        seed_rows,
        reps=int(cfg["statistics"]["bootstrap_reps"]),
        seed=int(cfg["statistics"]["bootstrap_seed"]),
    )
    baseline_results = {}
    source_hashes = {str(path): _sha256(path) for path in seed_paths.values()}
    for offset, (name, value) in enumerate(cfg["sources"]["baselines"].items()):
        path = repo_path(value)
        rows = [row for row in _load(path) if row.get("appearance_split") == "test"]
        baseline_results[name] = {
            "metrics": accuracy_metrics(rows),
            "paired_vs_ours": paired_cluster_test(
                seed_rows,
                rows,
                reps=int(cfg["statistics"]["bootstrap_reps"]),
                seed=int(cfg["statistics"]["bootstrap_seed"]) + offset + 1,
            ),
        }
        source_hashes[str(path)] = _sha256(path)
    summary = {
        "per_cell": {
            cell: _mean_sd(seed_metrics, ("per_cell", cell)) for cell in CELLS
        },
        "t3_subcells": {
            sub: _mean_sd(seed_metrics, ("t3_subcells", sub))
            for sub in ("looks_safe", "O8", "reverse")
        },
        "worst_cell": _mean_sd(seed_metrics, ("worst_cell",)),
        "macro": _mean_sd(seed_metrics, ("macro",)),
    }
    result = {
        "schema_version": 1,
        "experiment": cfg["experiment"],
        "status": cfg["statistics"]["current_split_status"],
        "split": args.split,
        "estimand": "attribution accuracy; snapshot rates for gates, cluster-balanced inference",
        "n_training_seeds": len(seeds),
        "five_seed_summary": summary,
        "acceptance": acceptance,
        "cluster_bootstrap": bootstrap,
        "baselines": baseline_results,
        "source_sha256": source_hashes,
        "anti_pseudoreplication": {
            "snapshot_independence_assumed": False,
            "bootstrap_unit": cfg["statistics"]["bootstrap_unit"],
            "paired_test": cfg["statistics"]["paired_test"],
        },
    }
    out = Path(args.out)
    write_json(out, result)
    out.with_suffix(".md").write_text(_markdown(result) + "\n")
    print(json.dumps({"passed": acceptance["passed"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
