"""Build VLA training examples from the M6 Hindsight-CoT dataset + seeded train/val/test split.

Turns the filtered M6 dataset (``samples.jsonl`` kept records + ``frames.npz`` modalities, spec
§10) into :class:`VlaExample`s — the (prompt messages, RGB frames, Kino-Token window, target
completion) tuples the Kino-SFT trainer consumes (spec §11 Stage 1) — and partitions them into
train / val / test.

The split (the M7 deliverable the M6 card deferred — "train/val/test splits are defined at M7"):
operator-stratified, seeded, sample-disjoint, and reproducible from one ``(seed, fractions)``.
Suite-Sem (the ambiguity-pair held-out evaluation, spec §8.3 / M7 exit criterion 1) is the
test-split subset whose operator belongs to an ambiguity pair.

Pure-Python/numpy: no torch — the split + formatting run on the CI machine; the trainer
(:mod:`kino_vla.vla.model`) does the tokenization.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive, Snapshot
from kino_vla.utils.config import Config
from kino_vla.vla.prompt import build_messages, context_from_snapshot, format_target


@dataclass(frozen=True)
class VlaExample:
    """One Kino-SFT training example (spec §11 Stage 1)."""

    sample_id: str
    operator_name: str
    appearance_class: str
    ambiguity_pair: str | None
    ab_class: str
    attribution_truth: str  # privileged ground-truth category (the filter's anchor)
    primitive_truth: str  # the kept annotation's primitive (the SFT target action)
    messages: list[dict]  # prompt (system + user, content-list form)
    target_text: str  # <Thought>…</Thought><Action>{json}</Action>
    rgb: np.ndarray  # (n_frames, H, W, 3) float — the model attaches the last n_images
    proprio_window: np.ndarray  # (W, F) the Kino-Tokens precursor (latent route)
    target_theta: list[float]  # privileged-θ distillation target (TARGET_SCHEMA order)


@dataclass(frozen=True)
class DatasetSplit:
    """Seeded operator-stratified partition of the kept samples."""

    train: list[VlaExample]
    val: list[VlaExample]
    test: list[VlaExample]

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _snapshot_from_record(rec: dict[str, Any], frames: dict[str, np.ndarray]) -> Snapshot:
    """Rebuild a :class:`Snapshot` from a manifest record + its frame sidecar arrays."""
    m = rec["snapshot"]
    return Snapshot(
        operator_name=m["operator_name"],
        appearance_class=m["appearance_class"],
        t=float(m["t"]),
        pose_xy=np.asarray(m["pose_xy"], dtype=np.float64),
        heading=float(m["heading"]),
        rgb=frames["rgb"],
        depth=frames["depth"],
        proprio_window=frames["proprio"],
        prior_outputs=list(m["prior_outputs"]),
        privileged_theta={k: float(v) for k, v in m["privileged_theta"].items()},
        monitor_channel=m["monitor_channel"],
    )


def _annotation_from_record(rec: dict[str, Any]) -> CoTAnnotation:
    """Rebuild the kept :class:`CoTAnnotation` from a manifest record (for the SFT target)."""
    a = rec["annotation"]
    act = a["action"]
    primitive = RecoveryPrimitive(name=act["primitive"], params=dict(act.get("params", {})))
    return CoTAnnotation(
        thought=str(a.get("thought", "")),
        attribution=a["attribution"],
        primitive=primitive,
        attribution_raw=a.get("attribution_raw", a["attribution"]),
        raw_text="",
    )


def load_examples(
    out_dir: str | Path,
    cfg: Config,
    *,
    route: str = "latent",
    reveal_appearance: bool = False,
    n_images: int = 1,
) -> list[VlaExample]:
    """Load every kept M6 sample as a :class:`VlaExample` (prompt + frames + target).

    ``route`` selects the proprioception representation (``"text"`` numbers vs ``"latent"``
    Kino-Tokens, spec §3); ``reveal_appearance`` is the text-only ablation that names the surface
    instead of attaching frames.
    """
    out = Path(out_dir)
    records = [
        json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines() if line
    ]
    npz = np.load(out / "frames.npz")
    examples: list[VlaExample] = []
    for rec in records:
        if rec.get("annotation") is None:
            continue  # defensive: kept records always carry an annotation
        sid = rec["sample_id"]
        frames = {
            "rgb": npz[f"{sid}__rgb"],
            "depth": npz[f"{sid}__depth"],
            "proprio": npz[f"{sid}__proprio"],
        }
        snapshot = _snapshot_from_record(rec, frames)
        ctx = context_from_snapshot(snapshot, route=route, reveal_appearance=reveal_appearance)
        annotation = _annotation_from_record(rec)
        gt = rec["ground_truth"]
        examples.append(
            VlaExample(
                sample_id=sid,
                operator_name=snapshot.operator_name,
                appearance_class=snapshot.appearance_class,
                ambiguity_pair=rec.get("ambiguity_pair"),
                ab_class=gt["ab_class"],
                attribution_truth=gt["category"],
                primitive_truth=annotation.primitive.name,
                messages=build_messages(ctx, cfg, route=route, n_images=n_images),
                target_text=format_target(annotation),
                rgb=frames["rgb"],
                proprio_window=frames["proprio"],
                target_theta=[float(x) for x in rec["target_theta"]],
            )
        )
    return examples


def split_examples(
    examples: list[VlaExample],
    *,
    seed: int,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> DatasetSplit:
    """Operator-stratified, seeded, sample-disjoint train/val/test split (reproducible).

    Within each operator the examples are sorted by ``sample_id`` (stable) then shuffled with a
    per-seed RNG, so the same ``(seed, fractions)`` always yields the same partition. Each
    operator contributes proportionally to val and test (so every operator — and every ambiguity
    sibling — is represented in the held-out sets even when counts are uneven).
    """
    if not 0.0 <= val_frac < 1.0 or not 0.0 <= test_frac < 1.0 or val_frac + test_frac >= 1.0:
        raise ValueError("val_frac, test_frac must be in [0,1) and sum to < 1")
    rng = np.random.default_rng(int(seed))
    by_op: dict[str, list[VlaExample]] = defaultdict(list)
    for ex in examples:
        by_op[ex.operator_name].append(ex)

    train: list[VlaExample] = []
    val: list[VlaExample] = []
    test: list[VlaExample] = []
    for op in sorted(by_op):
        group = sorted(by_op[op], key=lambda e: e.sample_id)
        order = rng.permutation(len(group))
        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        # Guarantee at least one held-out example per operator when the group is big enough,
        # so Suite-Sem and the val gate always see every operator (no empty stratum).
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        n_val = min(n_val, max(0, n - n_test - 1))  # always leave ≥1 for train
        shuffled = [group[i] for i in order]
        test.extend(shuffled[:n_test])
        val.extend(shuffled[n_test : n_test + n_val])
        train.extend(shuffled[n_test + n_val :])
    return DatasetSplit(train=train, val=val, test=test)


def suite_sem_examples(examples: list[VlaExample]) -> list[VlaExample]:
    """The ambiguity-pair subset (Suite-Sem, spec §8.3): examples whose operator is in a pair."""
    return [e for e in examples if e.ambiguity_pair is not None]
