"""B-F — the honest "no-VLM" closed-set fusion baseline (Paper-A §4 A2 / §5 roster).

The predictable reviewer question on A2/A3 is "do you actually need a vision-language model, or
would a plain fusion classifier do?" B-F answers it head-on: real CLIP image features (the same
``ClipAppearanceEncoder`` the §7 map uses) concatenated with a proprio-encoder summary, fed to a
small MLP over the closed category set, trained on the SAME snapshots B5 saw. On the matched pair
the proprioception is C2ST-null (A2 validity gate), so B-F's discrimination comes entirely from CLIP
exactly the point: it is *expected to be competitive in-distribution* and to COLLAPSE on the
appearance-held-out split (A2) and the OOD axes (A5), because a closed-set CLIP-feature MLP keys on
the training colours, whereas the VLM carries a transferable semantic prior ("looks like a sticky
board" vs "muddy ground"). It also cannot emit rationales, recovery parameters, or open-vocabulary
labels — so the method claims live on the generalisation axes, said out loud (design §4 A2).

Drop-in ``VlaPolicy`` (``decide`` → category → canonical recovery), so it plugs into the identical
:mod:`kino_vla.eval.a2_headline` path as every other agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive, Snapshot
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.eval.suite_sem import _default_params
from kino_vla.utils.config import Config
from kino_vla.vla.output import REJECT_SCHEMA, ParsedDecision


def proprio_stats(window: np.ndarray) -> np.ndarray:
    """A fixed proprio-encoder summary: per-channel mean + std over the window (2F-dim).

    Deterministic and cheap — a realistic proprio front-end for a non-VLM classifier. On the matched
    pair this branch is uninformative by construction (the pair is C2ST-indistinguishable), which is
    the honest point of the baseline; on the fine-structure cells (A7) it can be swapped for a
    learned temporal encoder without changing the interface.
    """
    w = np.asarray(window, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] == 0:
        raise ValueError(f"expected a (T, F) proprio window, got {w.shape}")
    return np.concatenate([w.mean(axis=0), w.std(axis=0)]).astype(np.float64)


@dataclass
class _Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.where(self.std < 1e-8, 1.0, self.std)


class FusionBaselinePolicy:
    """B-F: real CLIP(RGB) ⊕ proprio-stats → MLP over the category set, as a ``VlaPolicy``."""

    def __init__(
        self,
        model: object,  # torch.nn.Module (the trained fusion MLP)
        encoder: object,  # ClipAppearanceEncoder
        standardizer: _Standardizer,
        categories: list[str],
        *,
        canonical: dict[str, str],
    ) -> None:
        self._model = model
        self._encoder = encoder
        self._std = standardizer
        self._categories = list(categories)
        self._canonical = dict(canonical)

    # ------------------------------------------------------------------ features
    def _features(self, snapshot: Snapshot) -> np.ndarray:
        rgb = snapshot.rgb[-1] if snapshot.rgb.size else None
        if rgb is None:
            raise ValueError("B-F needs an RGB frame")
        clip = self._encoder.embed(np.asarray(rgb, dtype=np.float32))  # (512,)
        prop = self._std.apply(proprio_stats(snapshot.proprio_window))  # (2F,)
        return np.concatenate([clip, prop]).astype(np.float32)

    def decide(self, snapshot: Snapshot, map_note: str = "") -> ParsedDecision:  # noqa: ARG002
        import torch

        try:
            feat = self._features(snapshot)
        except ValueError as err:
            return ParsedDecision(ok=False, raw_text="", reject_code=f"{REJECT_SCHEMA}: {err}")
        with torch.no_grad():
            logits = self._model(torch.from_numpy(feat).unsqueeze(0))
            idx = int(torch.argmax(logits, dim=-1).item())
        category = self._categories[idx]
        primitive = self._canonical.get(category, "Set_Constraint")
        ann = CoTAnnotation(
            thought=f"fusion classifier (CLIP+proprio) → {category}; canonical recovery.",
            attribution=category,
            primitive=RecoveryPrimitive(primitive, _default_params(primitive)),
            attribution_raw=category,
            raw_text="",
        )
        return ParsedDecision(ok=True, raw_text="", annotation=ann)

    # ------------------------------------------------------------------ checkpoint
    @classmethod
    def load(
        cls, path: str | Path, pcfg: Config, taxonomy: FailureTaxonomy
    ) -> FusionBaselinePolicy:
        import torch

        from kino_vla.map.clip_appearance import ClipAppearanceEncoder

        blob = torch.load(path, map_location="cpu", weights_only=False)
        categories = list(blob["categories"])
        model = _build_mlp(int(blob["in_dim"]), len(categories), int(blob["hidden"]))
        model.load_state_dict(blob["state_dict"])
        model.eval()
        std = _Standardizer(np.asarray(blob["prop_mean"]), np.asarray(blob["prop_std"]))
        return cls(
            model,
            ClipAppearanceEncoder(),
            std,
            categories,
            canonical=pcfg.recovery.canonical.to_dict(),
        )


def _build_mlp(in_dim: int, n_classes: int, hidden: int) -> torch.nn.Module:  # noqa: F821
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden, hidden // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden // 2, n_classes),
    )


def train_fusion(
    train_dir: str | Path,
    out_path: str | Path,
    *,
    hidden: int = 256,
    epochs: int = 60,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Train B-F on a VLA dataset dir (samples.jsonl + frames.npz): CLIP(RGB)⊕proprio → category."""
    import torch
    from torch import nn

    from kino_vla.map.clip_appearance import ClipAppearanceEncoder

    train_dir = Path(train_dir)
    records = [
        json.loads(x) for x in (train_dir / "samples.jsonl").read_text().splitlines() if x
    ]
    npz = np.load(train_dir / "frames.npz")
    encoder = ClipAppearanceEncoder()

    rgbs, props, cats = [], [], []
    for r in records:
        sid = r["sample_id"]
        key = f"{sid}__rgb"
        if key not in npz.files:
            continue
        rgbs.append(np.asarray(npz[key][-1], dtype=np.float32))
        props.append(proprio_stats(npz[f"{sid}__proprio"]))
        cats.append(r["ground_truth"]["category"])
    clip_feats = encoder.embed_batch(rgbs)  # (N, 512)
    prop_arr = np.asarray(props, dtype=np.float64)  # (N, 2F)
    categories = sorted(set(cats))
    cat_idx = {c: i for i, c in enumerate(categories)}
    y = np.asarray([cat_idx[c] for c in cats], dtype=np.int64)

    prop_mean = prop_arr.mean(axis=0)
    prop_std = prop_arr.std(axis=0)
    prop_z = (prop_arr - prop_mean) / np.where(prop_std < 1e-8, 1.0, prop_std)
    x = np.concatenate([clip_feats, prop_z], axis=1).astype(np.float32)

    torch.manual_seed(seed)
    model = _build_mlp(x.shape[1], len(categories), hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # class-balanced CE (the base ops dominate; the matched adhesion/compliant must not be swamped)
    counts = np.bincount(y, minlength=len(categories)).astype(np.float64)
    weight = torch.tensor(
        counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=device
    )
    lossf = nn.CrossEntropyLoss(weight=weight)
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        train_acc = float((model(xt).argmax(-1) == yt).float().mean().item())

    blob = {
        "state_dict": model.state_dict(),
        "categories": categories,
        "in_dim": int(x.shape[1]),
        "hidden": int(hidden),
        "prop_mean": prop_mean,
        "prop_std": prop_std,
        "n_train": int(x.shape[0]),
        "train_acc": train_acc,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out_path)
    return {"n_train": int(x.shape[0]), "categories": categories, "train_acc": train_acc}
