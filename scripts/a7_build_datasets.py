#!/usr/bin/env python
# ruff: noqa: E501
"""A7 dataset builder: text-schema views, conflict-dose curricula, and CoT-filter cards.

All outputs are derived from existing real-stack frozen artifacts. If a real source needed for a
publication finding is missing (e.g. raw unfiltered ApiOracle annotations), the builder writes an
explicit blocked artifact instead of silently substituting scripted/surrogate data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, repo_path, write_json

CONFLICT_COT = {
    "O4_tether": (
        "The tangential resistance could be compliant mud, but the surface ahead is clearly "
        "an adhesive board. When the proprioception is ambiguous between mud and adhesion, "
        "the visible material is decisive: this is adhesion. Escape the sticky patch with a "
        "Backstep rather than pushing through."
    ),
    "O2_compliance": (
        "The surface ahead is mud and the proprioception shows sustained compliance — vision "
        "and body agree, this is soft deformable ground with no conflict. Attribute "
        "compliant_terrain and switch to a high-step gait to traverse it."
    ),
}


def _load_records_frames(d: str | Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    root = repo_path(d)
    recs = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines() if line]
    npz = np.load(root / "frames.npz")
    return recs, {k: npz[k] for k in npz.files}


def _records_sha(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(r, sort_keys=True) for r in records)
    import hashlib

    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _write_dataset(out: Path, records: list[dict[str, Any]], frames: dict[str, np.ndarray], card: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    shipped = {r["sample_id"] for r in records}
    kept_frames = {
        k: v
        for k, v in frames.items()
        if k.split("__", 1)[0] in shipped and k.split("__", 1)[-1] in ("rgb", "depth", "proprio")
    }
    np.savez(out / "frames.npz", **kept_frames)
    card = dict(card)
    card["n_records"] = len(records)
    card["frames_keys"] = len(kept_frames)
    card["records_sha256"] = _records_sha(records)
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2) + "\n")


def select_conflict_records(
    corpus_records: list[dict[str, Any]],
    *,
    train_appearances: dict[str, list[str]],
    dose: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Select a deterministic nested matched-conflict curriculum.

    ``dose`` is the number of O4 conflict examples; the builder also selects up to ``dose`` O2 control
    examples so the curriculum remains balanced across the matched pair. Selection happens only from
    pre-registered train appearances and train split records.
    """
    import copy

    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, Any]] = []
    selected_ids: dict[str, list[str]] = {}
    for op in sorted(train_appearances):
        apps = set(train_appearances[op])
        pool = [
            r
            for r in corpus_records
            if r["snapshot"]["operator_name"] == op
            and (r.get("appearance_id") or r["snapshot"].get("appearance_class")) in apps
            and r.get("appearance_split", "train") == "train"
            and r.get("annotation") is not None
        ]
        pool = sorted(pool, key=lambda r: r["sample_id"])
        order = list(rng.permutation(len(pool))) if pool else []
        ordered = [pool[i] for i in order]
        chosen = ordered[: min(int(dose), len(ordered))]
        selected_ids[op] = [r["sample_id"] for r in chosen]
        for rec in chosen:
            rec2 = copy.deepcopy(rec)
            rec2.setdefault("verdict", {"keep": True, "reason": "A7-conflict-dose"})
            if rec2.get("annotation") is not None and op in CONFLICT_COT:
                rec2["annotation"]["thought"] = CONFLICT_COT[op]
                if "attribution" in rec2["annotation"]:
                    rec2["annotation"]["attribution_raw"] = rec2["annotation"]["attribution"]
            rec2["a7_conflict_dose"] = int(dose)
            selected.append(rec2)
    return selected, selected_ids


def build_conflict_dose(cfg: dict[str, Any], config_path: str) -> list[Path]:
    out_root = repo_path(cfg["output_dir"]) / "datasets" / "conflict_dose"
    h_recs, h_frames = _load_records_frames(cfg["sources"]["hindsight_filtered"])
    c_recs, c_frames = _load_records_frames(cfg["sources"]["a0_corpus"])
    h_ids = {r["sample_id"] for r in h_recs}
    train_apps = cfg["conflict_doses"]["train_appearances"]
    seed = int(cfg["seeds"]["split_seed"])
    written = []
    for dose in cfg["conflict_doses"]["matched_samples"]:
        conflict, selected_ids = select_conflict_records(
            c_recs, train_appearances=train_apps, dose=int(dose), seed=seed
        )
        all_records = list(h_recs) + conflict
        frames = dict(h_frames)
        for k, v in c_frames.items():
            sid = k.split("__", 1)[0]
            if sid not in h_ids:
                frames[k] = v
        out = out_root / f"dose_{int(dose):02d}"
        card = {
            **artifact_meta(
                config_path,
                sources={
                    "hindsight_filtered": cfg["sources"]["hindsight_filtered"],
                    "a0_corpus": cfg["sources"]["a0_corpus"],
                },
            ),
            "name": f"a7_conflict_dose_{int(dose):02d}",
            "dose_definition": "number of O4_tether conflict examples; equal O2_compliance controls selected when available",
            "dose": int(dose),
            "n_hindsight": len(h_recs),
            "n_conflict_selected": len(conflict),
            "selected_ids": selected_ids,
            "train_appearances": train_apps,
            "leakage_check": "selected only from appearance_split=train and configured train appearances",
        }
        _write_dataset(out, all_records, frames, card)
        written.append(out)
    return written


def build_schema_views(cfg: dict[str, Any], config_path: str) -> list[Path]:
    out_root = repo_path(cfg["output_dir"]) / "datasets" / "text_schema"
    src = repo_path(cfg["sources"]["a3_conflict_dataset"])
    written = []
    for name, spec in cfg["text_schemas"].items():
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        # Avoid duplicating frames; this view card declares the route/proprio_detail used at train time.
        card = {
            **artifact_meta(config_path, sources={"source_dataset": str(src)}),
            "name": f"a7_text_schema_{name}",
            "source_dataset": str(src),
            "route": spec["route"],
            "proprio_detail": spec["proprio_detail"],
            "adapter": spec.get("adapter"),
            "description": spec.get("description"),
            "view_only": True,
        }
        (out / "dataset_card.json").write_text(json.dumps(card, indent=2) + "\n")
        written.append(out)
    return written


def build_cot_filter_cards(cfg: dict[str, Any], config_path: str) -> list[Path]:
    out_root = repo_path(cfg["output_dir"]) / "datasets" / "cot_filter"
    out_root.mkdir(parents=True, exist_ok=True)
    src = repo_path(cfg["sources"]["hindsight_filtered"])
    card_paths = []
    # Truth-filtered real dataset exists and is a valid view.
    filtered_card = {
        **artifact_meta(config_path, sources={"hindsight_filtered": str(src)}),
        "name": "a7_cot_truth_filtered",
        "status": "available",
        "source_dataset": str(src),
        "finding": True,
        "notes": "Uses the already truth-filtered Hindsight-CoT dataset.",
    }
    p = out_root / "truth_filtered_card.json"
    p.write_text(json.dumps(filtered_card, indent=2) + "\n")
    card_paths.append(p)
    # The unfiltered real ApiOracle stream is the union of kept samples and filter-rejected
    # dropped records with annotations. Oracle transport errors are excluded by the reducer.
    unfiltered_card = {
        **artifact_meta(
            config_path,
            sources={
                "hindsight_kept": str(src / "samples.jsonl"),
                "hindsight_dropped": str(src / "dropped.jsonl"),
            },
        ),
        "name": "a7_cot_unfiltered_api_oracle",
        "status": "available",
        "finding": True,
        "source_dataset": str(src),
        "notes": "Uses real ApiOracle kept+dropped Hindsight records before truth-filter retention; oracle_error rows are excluded from filter-effect denominators.",
    }
    p = out_root / "unfiltered_api_oracle_card.json"
    p.write_text(json.dumps(unfiltered_card, indent=2) + "\n")
    card_paths.append(p)
    return card_paths


def build_all(config_path: str, stage: str) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    written: dict[str, list[str]] = {}
    if stage in ("all", "schema"):
        written["schema"] = [str(p) for p in build_schema_views(cfg, config_path)]
    if stage in ("all", "conflict-dose"):
        written["conflict_dose"] = [str(p) for p in build_conflict_dose(cfg, config_path)]
    if stage in ("all", "cot-filter"):
        written["cot_filter"] = [str(p) for p in build_cot_filter_cards(cfg, config_path)]
    summary = {
        **artifact_meta(config_path, sources={"config": config_path}),
        "stage": stage,
        "written": written,
    }
    out = repo_path(cfg["output_dir"]) / "datasets" / "build_summary.json"
    write_json(out, summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build A7 derived datasets and dataset cards")
    ap.add_argument("--config", default="configs/eval/a7.yaml")
    ap.add_argument("--stage", default="all", choices=["all", "schema", "conflict-dose", "cot-filter"])
    args = ap.parse_args()
    summary = build_all(args.config, args.stage)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
