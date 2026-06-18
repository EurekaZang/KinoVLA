"""Hindsight-CoT dataset format + auto-generated dataset card (spec §10; M6 exit criterion 3).

Format: a JSONL manifest of the lightweight per-sample records (scenario, ground truth, the
Oracle annotation, the verdict, the privileged-θ distillation target) plus a sidecar ``.npz``
holding the heavy modalities (RGB / depth / proprio window) keyed by ``sample_id`` — so no
large binaries land in git (QA 5.4; the dataset lives under ``outputs/``). Kept samples go to
``samples.jsonl`` (the SFT set, M7); dropped samples go to ``dropped.jsonl`` (audit + the
"去真值过滤" ablation training set, spec §12). The dataset card (markdown + JSON) is generated
from the pipeline stats — per-operator counts, A/B balance, ambiguity-pair coverage, reject-rate
by reason, throughput — with full provenance (seed, config hash, git commit).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.data.pipeline import PipelineResult
from kino_vla.data.schema import DataSample, Snapshot
from kino_vla.tokens.features import TARGET_SCHEMA
from kino_vla.utils.config import Config


def config_hash(cfg: Config) -> str:
    """Stable short hash of the config (provenance; QA 5.4 — numbers trace to a config hash)."""
    blob = json.dumps(cfg.to_dict(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def write_dataset(
    result: PipelineResult,
    cfg: Config,
    out_dir: str | Path,
    *,
    seed: int,
    oracle_name: str,
    git_commit: str = "unknown",
    with_frames: bool = True,
) -> dict[str, Any]:
    """Write the manifest, frames, dataset card, and review annotations; return the card dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept = [s for s in result.samples if s.verdict.keep]
    dropped = [s for s in result.samples if not s.verdict.keep]

    _write_jsonl(out / "samples.jsonl", kept)
    _write_jsonl(out / "dropped.jsonl", dropped)
    if with_frames:
        _write_frames(out / "frames.npz", kept)

    card = build_card(result, cfg, seed=seed, oracle_name=oracle_name, git_commit=git_commit)
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2))
    (out / "dataset_card.md").write_text(render_card_md(card))
    n_review = int(cfg.report.n_review_annotations)
    (out / "sample_annotations.md").write_text(render_review_annotations(result.samples, n_review))
    return card


def read_dataset(out_dir: str | Path) -> tuple[list[dict], dict]:
    """Read back the kept-sample manifest records and the dataset card (round-trip helper)."""
    out = Path(out_dir)
    records = [
        json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines() if line
    ]
    card = json.loads((out / "dataset_card.json").read_text())
    return records, card


def load_frames(out_dir: str | Path, sample_id: str) -> dict[str, np.ndarray]:
    """Load one sample's heavy modalities from the sidecar ``.npz``."""
    data = np.load(Path(out_dir) / "frames.npz")
    return {
        "rgb": data[f"{sample_id}__rgb"],
        "depth": data[f"{sample_id}__depth"],
        "proprio": data[f"{sample_id}__proprio"],
    }


# ----------------------------------------------- raw snapshot store (decoupled GPU→CPU pipeline)
def save_snapshots(path: str | Path, items: list[tuple[Snapshot, dict[str, float]]]) -> Path:
    """Persist collected snapshots (+ each operator's θ for the taxonomy) to an ``.npz`` + json.

    Decouples GPU collection from CPU annotation (CLAUDE.md #23): the real-Go2 snapshots are
    collected once on the GPU (scripts/isaac_hindsight_collect.py) and saved here; the external
    Oracle + filter then run offline (build_hindsight_dataset.py --from-snapshots).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    for i, (snap, op_theta) in enumerate(items):
        arrays[f"{i}__rgb"] = snap.rgb.astype(np.float32)
        arrays[f"{i}__depth"] = snap.depth.astype(np.float32)
        arrays[f"{i}__proprio"] = snap.proprio_window.astype(np.float32)
        manifest.append({**snap.to_meta(), "op_theta": {k: float(v) for k, v in op_theta.items()}})
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return path


def merge_snapshots(shard_paths: list[str | Path], out_path: str | Path) -> int:
    """Concatenate per-shard snapshot ``.npz`` files (scale collection); return the count."""
    items: list[tuple[Snapshot, dict[str, float]]] = []
    for p in shard_paths:
        items.extend(load_snapshots(p))
    save_snapshots(out_path, items)
    return len(items)


def load_snapshots(path: str | Path) -> list[tuple[Snapshot, dict[str, float]]]:
    """Reload snapshots saved by :func:`save_snapshots` as ``(Snapshot, op_theta)`` pairs."""
    path = Path(path)
    data = np.load(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    out: list[tuple[Snapshot, dict[str, float]]] = []
    for i, meta in enumerate(manifest):
        snap = Snapshot(
            operator_name=meta["operator_name"],
            appearance_class=meta["appearance_class"],
            t=float(meta["t"]),
            pose_xy=np.asarray(meta["pose_xy"], dtype=np.float64),
            heading=float(meta["heading"]),
            rgb=data[f"{i}__rgb"],
            depth=data[f"{i}__depth"],
            proprio_window=data[f"{i}__proprio"],
            prior_outputs=list(meta["prior_outputs"]),
            privileged_theta={k: float(v) for k, v in meta["privileged_theta"].items()},
            monitor_channel=meta["monitor_channel"],
        )
        out.append((snap, {k: float(v) for k, v in meta["op_theta"].items()}))
    return out


# ---------------------------------------------------------------------- dataset card
def build_card(
    result: PipelineResult,
    cfg: Config,
    *,
    seed: int,
    oracle_name: str,
    git_commit: str,
) -> dict[str, Any]:
    """Assemble the dataset card dict from the pipeline stats + provenance."""
    stats = result.stats
    assert stats is not None
    target = float(cfg.report.target_samples_per_hour)
    return {
        "name": "kino_fail_hindsight_cot",
        "spec": "§10 Privileged-Grounded Hindsight CoT",
        "provenance": {
            "seed": int(seed),
            "config_hash": config_hash(cfg),
            "git_commit": git_commit,
            "oracle": oracle_name,
            "target_schema": list(TARGET_SCHEMA),
        },
        "stats": stats.to_dict(),
        "throughput_ok": stats.samples_per_hour >= target,
        "target_samples_per_hour": target,
    }


def render_card_md(card: dict[str, Any]) -> str:
    """Render the dataset card as markdown (the human-readable artifact)."""
    s = card["stats"]
    prov = card["provenance"]
    lines: list[str] = [
        f"# Dataset Card — {card['name']}",
        "",
        f"_{card['spec']}_ · auto-generated (do not hand-edit; QA 5.4)",
        "",
        "## Summary",
        f"- maze cells: **{s['n_cells']}** · interceptions: **{s['n_interceptions']}** · "
        f"no-interception: **{s['n_no_interception']}**",
        f"- kept: **{s['kept']}** · dropped: **{s['dropped']}** · "
        f"reject-rate: **{s['reject_rate']:.1%}**",
        f"- throughput: **{s['samples_per_hour']:.0f}** samples/hour "
        f"(kept **{s['kept_per_hour']:.0f}**/hour) · target ≥ "
        f"{card['target_samples_per_hour']:.0f} "
        f"→ {'PASS' if card['throughput_ok'] else 'FAIL'}",
        "",
        "## Reject-rate by reason (truth-consistency filter, spec §10 PHASE 3)",
        "| reason | count |",
        "| --- | --- |",
    ]
    lines += [f"| {reason} | {count} |" for reason, count in sorted(s["by_reason"].items())]
    lines += [
        "",
        "## Per-operator (kept / total interceptions)",
        "| operator | kept | total |",
        "| --- | --- | --- |",
    ]
    for op in sorted(s["per_operator_total"]):
        lines.append(
            f"| {op} | {s['per_operator_kept'].get(op, 0)} | {s['per_operator_total'][op]} |"
        )
    lines += [
        "",
        "## A/B balance (kept; spec §2.5 recoverability dichotomy)",
        f"- A (low-level recoverable): **{s['ab_balance_kept'].get('A', 0)}**",
        f"- B (semantic-intervention required): **{s['ab_balance_kept'].get('B', 0)}**",
        "",
        "## Ambiguity-pair coverage (Suite-Sem; spec §8.1 P4)",
        "| pair | kept (each sibling) | both present |",
        "| --- | --- | --- |",
    ]
    for label, cov in sorted(card["stats"]["ambiguity_coverage"].items()):
        per = ", ".join(f"{k}={v}" for k, v in cov.items() if k != "both_present")
        lines.append(f"| {label} | {per} | {'yes' if cov['both_present'] else 'NO'} |")
    lines += [
        "",
        "## Provenance",
        f"- seed: `{prov['seed']}` · config hash: `{prov['config_hash']}` · "
        f"git: `{prov['git_commit']}`",
        f"- oracle: `{prov['oracle']}` · distillation target schema: `{prov['target_schema']}`",
        "",
    ]
    return "\n".join(lines)


def render_review_annotations(samples: list[DataSample], n: int) -> str:
    """Render the first ``n`` annotations for human review (QA 5.2 prompt-change gate)."""
    lines = [
        "# Sample annotations for review (QA 5.2)",
        "",
        "Review these after any Oracle prompt-template change (10 samples by default).",
        "",
    ]
    for sample in samples[:n]:
        ann = sample.annotation
        lines += [
            f"## {sample.sample_id}  ({sample.snapshot.operator_name})",
            f"- appearance: `{sample.snapshot.appearance_class}` · monitor: "
            f"`{sample.snapshot.monitor_channel}` · θ-truth: "
            f"`{sample.snapshot.privileged_theta}`",
            f"- ground-truth attribution: `{sample.ground_truth.category}` "
            f"(class {sample.ground_truth.ab_class}); feasible: "
            f"`{sorted(sample.ground_truth.feasible)}`",
            f"- oracle attribution: `{ann.attribution if ann else None}` · primitive: "
            f"`{ann.primitive.name if ann else None}`",
            f"- **verdict: `{sample.verdict.reason}`** — {sample.verdict.detail}",
            "",
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------------- writers
def _write_jsonl(path: Path, samples: list[DataSample]) -> None:
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_record()) + "\n")


def _write_frames(path: Path, samples: list[DataSample]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for sample in samples:
        snap = sample.snapshot
        arrays[f"{sample.sample_id}__rgb"] = snap.rgb.astype(np.float32)
        arrays[f"{sample.sample_id}__depth"] = snap.depth.astype(np.float32)
        arrays[f"{sample.sample_id}__proprio"] = snap.proprio_window.astype(np.float32)
    np.savez_compressed(path, **arrays)
