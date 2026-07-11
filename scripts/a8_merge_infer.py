#!/usr/bin/env python
"""Merge real a8_infer JSON results into outputs/eval/a8/a8a/summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, repo_path, write_json
from kino_vla.eval.a8_metrics import accuracy_ci, macro_f1, per_category_recall

GUARDIAN_PAPER_REFERENCE = {
    "status": "reference_only",
    "source": "Pacaud et al. Guardian arXiv:2512.01946 Table II",
    "RoboFail": {"exec": 0.86, "plan": 0.70},
    "UR5-Fail": {"exec": 0.77, "plan": 0.89},
    "RoboVQA": {"exec": 0.85, "plan": None},
}


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/a8.yaml")
    ap.add_argument("--infer-dir", default="outputs/eval/a8/a8a/infer")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    infer_dir = repo_path(args.infer_dir)
    out_path = repo_path(cfg["output_dir"]) / "a8a" / "summary.json"

    real_splits = {}
    sources = {}
    for p in sorted(infer_dir.glob("zero_shot_*.json")):
        if p.name.endswith("_summary.json") or "smoke" in p.name:
            continue
        data = _load(p)
        if not data or "rows" not in data:
            continue
        rows = data["rows"]
        if not rows:
            continue
        y_true = [r["truth"] for r in rows]
        y_pred = [r["pred"] for r in rows]
        # derive split key from filename
        name = p.stem.replace("zero_shot_", "")
        key = f"{name}__execution"
        real_splits[key] = {
            "accuracy": accuracy_ci(rows, field="correct"),
            "macro_f1": round(macro_f1(y_true, y_pred), 3),
            "per_category_recall": per_category_recall(y_true, y_pred),
            "n": len(rows),
            "status": "real_qwen3vl_zero_shot",
            "source": str(p),
        }
        sources[name] = str(p)

    payload = {
        **artifact_meta(args.config, sources=sources),
        "status": "partial_real" if real_splits else "requires_infer",
        "note": (
            "Real rows are Qwen3-VL-4B zero-shot generations on official images. "
            "Guardian-8B is paper reference only. FailCoT-SFT / Kino-VLA-General rows appear "
            "after training adapters and re-running a8_infer with --adapter."
        ),
        "arms": {
            "zero_shot": {
                "arm": "zero_shot",
                "predictor": "qwen3vl4b_zero_shot",
                "splits": real_splits,
            },
            "guardian_8b_paper": GUARDIAN_PAPER_REFERENCE,
        },
        "real_splits": sorted(real_splits),
    }
    if "ur5fail_test__execution" in real_splits:
        cell = real_splits["ur5fail_test__execution"]
        rec = cell["per_category_recall"]
        payload["headline_ur5_zero_shot"] = {
            "acc": cell["accuracy"]["rate"],
            "ci": cell["accuracy"]["ci"],
            "macro_f1": cell["macro_f1"],
            "failure_recall": rec.get("failure", {}).get("recall"),
            "success_recall": rec.get("success", {}).get("recall"),
            "n": cell["n"],
        }
    write_json(out_path, payload)
    print(json.dumps({"out": str(out_path), "real_splits": sorted(real_splits)}, indent=2))


if __name__ == "__main__":
    main()
