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
from kino_vla.data.schema import DROP_SCHEMA, ORACLE_ERROR, DataSample, Snapshot
from kino_vla.tokens.features import TARGET_SCHEMA
from kino_vla.utils.config import Config


def config_hash(cfg: Config) -> str:
    """Stable short hash of the config (provenance; QA 5.4 — numbers trace to a config hash)."""
    blob = json.dumps(cfg.to_dict(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_commit_sha() -> str:
    """Short HEAD SHA for card provenance (QA 5.4 — every number traces to a commit). ``unknown``
    if not a git checkout. Used by every dataset writer so the card never stamps a placeholder."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


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


def _card_wall(d: Path) -> float:
    """Recorded annotation wall-clock from a dataset card (0 if absent) — for merge throughput."""
    try:
        return float(json.loads((d / "dataset_card.json").read_text())["stats"]["wall_time_s"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0.0


def merge_datasets(
    base_dir: str | Path,
    add_dir: str | Path,
    ops: list[str],
    *,
    cfg: Config,
    seed: int,
    oracle_name: str,
    git_commit: str = "merge",
) -> dict[str, Any]:
    """Fold the ``ops`` records of ``add_dir`` into ``base_dir`` IN PLACE; return the card.

    Adds O5/O10 to the canonical dataset without re-annotating the (expensive) other operators
    (#32). The base must carry no kept records for ``ops`` (else they'd duplicate; the base's
    region-op ids are disjoint from the added ``{i}_O5/O10`` ids, so frame keys never collide).
    Old ``oracle call failed`` records (pre-M1) are normalized to ORACLE_ERROR so the merged
    reject-rate excludes them; throughput is the summed component wall-clocks (m1).
    """
    from kino_vla.data.pipeline import stats_from_records

    base, add = Path(base_dir), Path(add_dir)
    add_ops = set(ops)

    def read(p: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def normalize(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in recs:
            v = r.get("verdict", {})
            if v.get("reason") == DROP_SCHEMA and "oracle call failed" in v.get("detail", ""):
                v["reason"] = ORACLE_ERROR  # M1: a failed API call is NOT a filter rejection
        return recs

    def op(r: dict[str, Any]) -> str:
        return r["snapshot"]["operator_name"]

    base_kept = normalize(read(base / "samples.jsonl"))
    base_drop = normalize(read(base / "dropped.jsonl"))
    add_kept = [r for r in normalize(read(add / "samples.jsonl")) if op(r) in add_ops]
    add_drop = [r for r in normalize(read(add / "dropped.jsonl")) if op(r) in add_ops]
    kept = [r for r in base_kept if op(r) not in add_ops] + add_kept
    drop = [r for r in base_drop if op(r) not in add_ops] + add_drop

    # Union the kept-sample frame sidecars (base region keys + new op keys are disjoint).
    base_frames = np.load(base / "frames.npz")
    add_frames = np.load(add / "frames.npz")
    merged: dict[str, np.ndarray] = {}
    for r in kept:
        sid = r["sample_id"]
        src = add_frames if op(r) in add_ops else base_frames
        for suffix in ("rgb", "depth", "proprio"):
            key = f"{sid}__{suffix}"
            if key in src:
                merged[key] = src[key]

    stats = stats_from_records(cfg, kept, drop, wall_time_s=_card_wall(base) + _card_wall(add))
    card = build_card(
        PipelineResult(samples=[], stats=stats),
        cfg,
        seed=seed,
        oracle_name=oracle_name,
        git_commit=git_commit,
    )
    (base / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    (base / "dropped.jsonl").write_text("".join(json.dumps(r) + "\n" for r in drop))
    np.savez_compressed(base / "frames.npz", **merged)
    (base / "dataset_card.json").write_text(json.dumps(card, indent=2))
    (base / "dataset_card.md").write_text(render_card_md(card))
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
def _meta_path(path: Path) -> Path:
    """Sidecar collection-meta path for a snapshots ``.npz`` (e.g. ``snapshots.meta.json``)."""
    return path.parent / (path.stem + ".meta.json")


def save_snapshots(
    path: str | Path,
    items: list[tuple[Snapshot, dict[str, float]]],
    *,
    n_attempted: int | None = None,
) -> Path:
    """Persist collected snapshots (+ each operator's θ for the taxonomy) to an ``.npz`` + json.

    Decouples GPU collection from CPU annotation (CLAUDE.md #23): the real-Go2 snapshots are
    collected once on the GPU (scripts/isaac_hindsight_collect.py) and saved here; the external
    Oracle + filter then run offline (build_hindsight_dataset.py --from-snapshots).

    ``n_attempted`` (the number of lanes driven, >= len(items)) is recorded in a sidecar so the
    dataset card can report the true interception rate instead of hardcoding 0 misses (m5).
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
    if n_attempted is not None:
        _meta_path(path).write_text(
            json.dumps({"n_attempted": int(n_attempted), "n_saved": len(items)})
        )
    return path


def load_snapshot_meta(path: str | Path) -> dict[str, int]:
    """Read the collection sidecar (``n_attempted``/``n_saved``); ``{}`` if absent (m5)."""
    mp = _meta_path(Path(path))
    return json.loads(mp.read_text()) if mp.exists() else {}


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
    rep = cfg.report
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
        # Release metadata for the card's standard sections (m2/m3). License is a release decision
        # (config); sft_scale_floor is the count below which the set is flagged proof-of-pipeline.
        "release": {
            "license": str(rep.get("license", "see repository LICENSE")),
            "sft_scale_floor": int(rep.get("sft_scale_floor", 1000)),
        },
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
        f"- kept: **{s['kept']}** · filter-dropped: **{s['dropped']}** · "
        f"reject-rate: **{s['reject_rate']:.1%}** "
        f"(of the {s['kept'] + s['dropped']} the filter judged; oracle errors excluded)",
        f"- oracle errors (API/transport — NOT a filter rejection): **{s.get('oracle_error', 0)}**",
        f"- throughput: **{s['samples_per_hour']:.0f}** samples/hour "
        f"(kept **{s['kept_per_hour']:.0f}**/hour) · target ≥ "
        f"{card['target_samples_per_hour']:.0f} "
        f"→ {'PASS' if card['throughput_ok'] else 'FAIL'}",
        "",
        "## Reject-rate by reason (truth-consistency filter, spec §10 PHASE 3)",
        "_`oracle_error` = the Oracle API call failed; NOT a filter rejection, excluded from the "
        "reject-rate (recover via scripts/retry_failed_annotations.py)._",
        "",
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
    rel = card.get("release", {})
    per_kept = s["per_operator_kept"]
    skew = (
        f"{max(per_kept.values())}/{min(per_kept.values())} (max/min across operators)"
        if per_kept
        else "n/a"
    )
    floor = int(rel.get("sft_scale_floor", 1000))
    scale_note = (
        f"**{s['kept']} kept** is proof-of-pipeline, not SFT-scale (< {floor}); scale via more "
        "Isaac shards before training."
        if s["kept"] < floor
        else f"{s['kept']} kept samples."
    )
    limitations = [
        f"- **Scale**: {scale_note}",
        f"- **Per-operator skew**: kept counts vary {skew}; O7/region ops dominate while O5/O10 "
        "(embodiment) are scarcer — the observability floor (§6 #15/#32).",
        "- **Oracle != spec**: annotated by gpt-5.5 via an OpenAI-compatible gateway; spec §10 "
        "names Gemini 3.1 Pro. The truth-consistency filter makes the KEPT set θ-grounded "
        "regardless of the annotator.",
        "- **Perception**: RGB-D is the camera-free §7 pinhole render (real geometry), not a live "
        "RTX feed; the appearance encoder is a colour histogram, not open-vocab CLIP.",
        "- **Reproducibility**: the surrogate (ScriptedOracle) path is bit-reproducible (CI); the "
        "real-LLM dataset is NOT (external non-deterministic API) — archive the artifact + a hash.",
    ]
    if s.get("oracle_error", 0):
        limitations.append(
            f"- **Oracle errors**: {s['oracle_error']} snapshot(s) lost to transient gateway "
            "failures, recoverable via scripts/retry_failed_annotations.py."
        )
    lines += [
        "",
        "## Provenance",
        f"- seed: `{prov['seed']}` · config hash: `{prov['config_hash']}` · "
        f"git: `{prov['git_commit']}`",
        f"- oracle: `{prov['oracle']}` · distillation target schema: `{prov['target_schema']}`",
        "",
        "## Intended use",
        "- **Training** the VLA failure-recovery planner (M7): `samples.jsonl` is the Kino-SFT set "
        "(privileged-grounded CoT = physical attribution + one atomic §5 recovery primitive).",
        "- **Ablation**: `dropped.jsonl` is the truth-consistency-filter ablation set (the "
        '"去真值过滤" / no-filter baseline, spec §12) — Oracle CoTs that failed the filter.',
        "- **Not** an evaluation benchmark: the Cal/Sem/Comp/Bound/OOD suites are generated "
        "separately by the M8 eval harness, and train/val/test splits are defined at M7.",
        "",
        "## Collection (spec §10 PHASE 1-2)",
        "- Real-Go2 Isaac rollouts drive each operator into its failure; the Kino-Monitor "
        "intercepts and packages [5xRGB-D + 500 ms proprio window + privileged θ]. The external "
        "Oracle annotates (PHASE 3); the truth-consistency filter keeps only θ-grounded CoTs.",
        "",
        "## Limitations & biases",
        *limitations,
        "",
        "## License",
        f"- {rel.get('license', 'see repository LICENSE')}",
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
