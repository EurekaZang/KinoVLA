#!/usr/bin/env python
"""Build leakage-safe A8a/A8b dataset cards from downloaded Guardian/REFLECT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, repo_path, write_json
from kino_vla.eval.a8_datasets import (
    build_a8a_cards_from_split,
    build_a8b_cards_from_reflect,
    write_a8a_dataset,
    write_a8b_dataset,
)


def _find_split_dirs(guardian_root: Path) -> dict[str, Path]:
    """Map split name -> directory containing jsonl/records."""
    out: dict[str, Path] = {}
    if not guardian_root.exists():
        return out
    for p in sorted(guardian_root.iterdir()):
        if p.is_dir():
            # direct split dir
            if any(p.glob("*.jsonl")) or (p / "records").exists():
                out[p.name] = p
            # OOD bundle nested
            for child in p.iterdir() if p.is_dir() else []:
                if child.is_dir() and (any(child.glob("*.jsonl")) or (child / "records").exists()):
                    out[f"{p.name}/{child.name}"] = child
    return out


def build_a8a(cfg: dict[str, Any], config_path: str, *, limit: int | None = None) -> dict[str, str]:
    g_root = repo_path(cfg["guardian"]["root"])
    out_root = repo_path(cfg["output_dir"]) / "datasets" / "a8a"
    splits = _find_split_dirs(g_root)
    written: dict[str, str] = {}
    policy = str(cfg.get("prompt_policy", "execution_vanilla"))
    # policy like execution_vanilla
    if "_" in policy:
        stage, ppol = policy.split("_", 1)
    else:
        stage, ppol = "execution", "vanilla"
    for name, path in splits.items():
        cards = build_a8a_cards_from_split(
            path, split_name=name, stage=stage, prompt_policy=ppol, limit=limit
        )
        if not cards:
            # try planning too for coverage
            cards = build_a8a_cards_from_split(
                path, split_name=name, stage="planning", prompt_policy=ppol, limit=limit
            )
        if not cards:
            continue
        dest = out_root / name.replace("/", "__")
        write_a8a_dataset(
            dest,
            cards,
            card_meta={
                **artifact_meta(config_path, sources={"split": str(path)}),
                "split": name,
                "prompt_policy": policy,
            },
        )
        written[name] = str(dest)
    write_json(out_root / "index.json", {"splits": written, "n_splits": len(written)})
    return written


def build_a8b(cfg: dict[str, Any], config_path: str, *, limit: int | None = None) -> dict[str, Any]:
    out_root = repo_path(cfg["output_dir"]) / "datasets" / "a8b"
    audit_path = repo_path(cfg["a8b"]["admission_gate"])
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
    else:
        audit = {"pass": False, "reason": "missing audit file"}
    if not audit.get("pass"):
        blocked = {
            **artifact_meta(config_path),
            "status": "blocked",
            "reason": audit.get("reason", "a8b admission gate failed"),
            "audit": audit,
        }
        write_json(out_root / "blocked.json", blocked)
        return blocked

    reflect_root = repo_path(cfg["reflect"]["root"])
    # prefer real_data then sim_data
    candidates = [reflect_root / "real_data", reflect_root / "sim_data", reflect_root]
    cards = []
    used = None
    for c in candidates:
        cards = build_a8b_cards_from_reflect(c, limit=limit)
        if cards:
            used = c
            break
    if not cards:
        blocked = {
            **artifact_meta(config_path),
            "status": "blocked",
            "reason": "no a8b cards discovered under reflect root",
            "audit": audit,
        }
        write_json(out_root / "blocked.json", blocked)
        return blocked
    dest = out_root / "reflect_multisensory"
    write_a8b_dataset(
        dest,
        cards,
        card_meta={**artifact_meta(config_path, sources={"reflect": str(used)}), "source": str(used)},
    )
    index = {"status": "available", "dataset": str(dest), "n": len(cards), "source": str(used)}
    write_json(out_root / "index.json", index)
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/a8.yaml")
    ap.add_argument("--track", choices=["a8a", "a8b", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="optional per-split card cap for smoke")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out: dict[str, Any] = {}
    if args.track in ("a8a", "all"):
        out["a8a"] = build_a8a(cfg, args.config, limit=args.limit)
    if args.track in ("a8b", "all"):
        out["a8b"] = build_a8b(cfg, args.config, limit=args.limit)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
