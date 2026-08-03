"""Embodied DPO: physical-outcome preference pairs + DPO training (spec §11 Stage 2).

Stage 2 of the alignment paradigm: the SFT planner is placed in the closed loop, multiple rollouts
are sampled at each failure node, and the *physical outcome* (escaped to goal vs fell / dead-loop)
labels them — successes become Chosen, failures Rejected (spec §11). The v2 supplement is built in:
the ambiguity-pair wrong-strategy rollout (mud answered with Backstep, a glue board answered with
"push through") is exactly a high-quality Rejected, so DPO directly optimizes *attribution
correctness* (spec §5/§11).

Two halves: :func:`build_preference_pairs` (pure logic, golden-tested — turn rollout outcomes into
(prompt, chosen, rejected) triples) and :func:`train_dpo` (the real GPU step — the standard DPO
loss with an adapter-toggle reference, CLAUDE.md §6 #33). The preference triples are the falsifiable
artifact; training just consumes them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from kino_vla.data.schema import Snapshot
from kino_vla.utils.config import Config, load_config
from kino_vla.utils.seeding import seed_everything
from kino_vla.vla.prompt import build_messages, context_from_snapshot, format_target
from kino_vla.vla.rollout import RolloutResult

if TYPE_CHECKING:
    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.vla.model import VlaInputs


@dataclass
class PreferencePair:
    """One DPO triple: a failure-node prompt (snapshot) + a winning and a losing completion."""

    snapshot: Snapshot
    chosen_text: str
    rejected_text: str
    chosen_attr: str | None
    rejected_attr: str | None
    reason: str  # "outcome" (success vs failure) | "ambiguity" (right vs sibling strategy)

    def to_meta(self) -> dict:
        return {
            "operator": self.snapshot.operator_name,
            "appearance": self.snapshot.appearance_class,
            "chosen_attr": self.chosen_attr,
            "rejected_attr": self.rejected_attr,
            "reason": self.reason,
            "chosen": self.chosen_text,
            "rejected": self.rejected_text,
        }


def _completion_of(result: RolloutResult) -> str | None:
    """The ``<Thought>/<Action>`` completion of a rollout's first decision (None if it rejected)."""
    if not result.decisions:
        return None
    ann = result.decisions[0].decision.annotation
    return format_target(ann) if ann is not None else None


def build_preference_pairs(
    node_results: list[RolloutResult],
    *,
    max_pairs: int = 4,
) -> list[PreferencePair]:
    """Pair successful rollouts (Chosen) against failed ones (Rejected) at one node (spec §11).

    All rollouts in ``node_results`` are from the SAME failure node (same snapshot), so the prompt
    is shared and only the completion differs. A pair is emitted for each (success, failure) cross
    product up to ``max_pairs``; the snapshot is taken from a rollout that captured one.
    """
    snapshot = next((r.first_snapshot for r in node_results if r.first_snapshot is not None), None)
    if snapshot is None:
        return []
    wins = [r for r in node_results if r.success and _completion_of(r)]
    losses = [r for r in node_results if not r.success and _completion_of(r)]
    pairs: list[PreferencePair] = []
    for win, loss in product(wins, losses):
        if (
            win.first_primitive == loss.first_primitive
            and win.first_attribution == loss.first_attribution
        ):
            continue  # identical first decision, different luck — not an informative pair
        pairs.append(
            PreferencePair(
                snapshot=snapshot,
                chosen_text=_completion_of(win),
                rejected_text=_completion_of(loss),
                chosen_attr=win.first_attribution,
                rejected_attr=loss.first_attribution,
                reason="ambiguity"
                if win.first_attribution != loss.first_attribution
                else "outcome",
            )
        )
        if len(pairs) >= max_pairs:
            break
    return pairs


def build_ambiguity_pairs(items: list, tax: FailureTaxonomy, cfg: Config) -> list[PreferencePair]:
    """§11 ambiguity-pair preference pairs from in-distribution dataset nodes (the v2 supplement).

    For each Suite-Sem failure node, Chosen = the correct recovery (the node's privileged category +
    its canonical §5 primitive); Rejected = the ambiguity *sibling's* wrong-strategy (the sibling
    category + the sibling's canonical primitive). By the M6 disjoint-primitive construction the
    sibling's primitive lies OUTSIDE the correct category's feasible set, so it is the θ-grounded
    "选错策略" Rejected the truth filter would drop — DPO on these directly optimizes attribution
    correctness (spec §11). Deterministic (no sampling) and in-distribution (real dataset snapshots
    + their privileged truth), so it sidesteps the bang-bang→smooth deployment proprio shift.

    ``items`` are :class:`kino_vla.eval.suite_sem.SemItem`s (snapshot + attribution_truth).
    """
    sib = _sibling_categories(cfg, tax)
    canon = cfg.recovery.canonical
    pairs: list[PreferencePair] = []
    for it in items:
        cat = it.attribution_truth
        sibling = sib.get(cat)
        if sibling is None:
            continue
        chosen = _completion(cat, str(canon.get(cat, "Set_Constraint")), correct=True)
        rejected = _completion(sibling, str(canon.get(sibling, "Set_Constraint")), correct=False)
        pairs.append(
            PreferencePair(
                snapshot=it.snapshot,
                chosen_text=chosen,
                rejected_text=rejected,
                chosen_attr=cat,
                rejected_attr=sibling,
                reason="ambiguity",
            )
        )
    return pairs


def _sibling_categories(cfg: Config, tax: FailureTaxonomy) -> dict[str, str]:
    """Map each ambiguity-pair member's category to its sibling's (from the maze pairs)."""
    op_cat = cfg.attribution.operator_category
    out: dict[str, str] = {}
    for pair in cfg.maze.ambiguity_pairs:
        a, b = str(pair[0]), str(pair[1])
        ca, cb = str(op_cat.get(a)), str(op_cat.get(b))
        if ca and cb and ca != cb:
            out[ca], out[cb] = cb, ca
    return out


_DPO_PARAMS = {
    "Backstep": {"distance_m": 0.5},
    "Switch_Gait": {"mode": "high_step"},
    "Set_Constraint": {"max_speed": 0.4, "stiffness": 0.5},
    "Update_Topology": {"region_xy": [3.0, 0.0], "radius_m": 0.6, "status": "untraversable"},
    "Hold_and_Request": {"reason": "actuator torque saturated; cannot proceed safely"},
    "Adjust_Posture": {"body_height_m": 0.25, "pitch_deg": 0.0},
    "Replan_Waypoint": {"point_px": [480, 360]},
}


def _completion(category: str, primitive: str, *, correct: bool) -> str:
    """Render a ``<Thought>/<Action>`` completion for a (category, primitive) decision."""
    from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive

    note = "the physical evidence indicates" if correct else "a plausible but mistaken reading is"
    ann = CoTAnnotation(
        thought=f"Reflecting on the anomaly, {note} {category}; recover with {primitive}.",
        attribution=category,
        primitive=RecoveryPrimitive(primitive, dict(_DPO_PARAMS.get(primitive, {}))),
        attribution_raw=category,
        raw_text="",
    )
    return format_target(ann)


def save_pairs(pairs: list[PreferencePair], out_dir: str | Path) -> Path:
    """Persist preference pairs (snapshots → npz, triples → jsonl) for reproducible DPO training."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest: list[dict] = []
    for i, p in enumerate(pairs):
        arrays[f"{i}__rgb"] = p.snapshot.rgb.astype(np.float32)
        arrays[f"{i}__proprio"] = p.snapshot.proprio_window.astype(np.float32)
        manifest.append({**p.to_meta(), "snapshot": p.snapshot.to_meta()})
    np.savez_compressed(out / "pairs.npz", **arrays)
    (out / "pairs.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    return out


def pair_stats(pairs: list[PreferencePair]) -> dict:
    """Summary for the DPO data card: counts by operator + by pairing reason."""
    by_op: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for p in pairs:
        by_op[p.snapshot.operator_name] = by_op.get(p.snapshot.operator_name, 0) + 1
        by_reason[p.reason] = by_reason.get(p.reason, 0) + 1
    return {"n_pairs": len(pairs), "by_operator": by_op, "by_reason": by_reason}


# ------------------------------------------------------------------------- training
def dpo_loss(
    lp_chosen_pol: torch.Tensor,
    lp_rejected_pol: torch.Tensor,
    lp_chosen_ref: torch.Tensor,
    lp_rejected_ref: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard DPO loss (Rafailov et al.): −log σ(β·(Δπ_chosen − Δπ_rejected)), Δπ = πθ − πref."""
    logits = beta * ((lp_chosen_pol - lp_chosen_ref) - (lp_rejected_pol - lp_rejected_ref))
    return -torch.nn.functional.logsigmoid(logits)


def train_dpo(
    cfg: Config,
    pairs: list[PreferencePair],
    sft_adapter_dir: str | Path,
    out_dir: str | Path,
) -> dict:
    """DPO-train the SFT planner on the preference pairs; persist + return metrics (spec §11)."""
    from kino_vla.vla.model import KinoVLA

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.train.seed)
    seed_everything(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = KinoVLA.from_pretrained(cfg, device=device, dtype=dtype, adapter_dir=sft_adapter_dir)
    beta = float(cfg.train.beta)
    n_images = int(cfg.data.get("n_images", 1))
    route = str(cfg.get("route", "latent"))
    # loss_span="action" (the §6 #36 fix) makes the DPO preference contrast the <Action> decision
    # span only, not the confounded free-form Thought — essential for on-policy pairs.
    loss_span = str(cfg.train.get("loss_span", "completion"))
    prompt_cfg = load_config("data/hindsight.yaml")  # §5 vocab for the prompt (not the vla cfg)

    def make_inputs(snapshot: Snapshot, completion: str) -> VlaInputs:
        ctx = context_from_snapshot(snapshot, route=route)
        messages = build_messages(ctx, prompt_cfg, route=route, n_images=n_images)
        images = list(snapshot.rgb[-n_images:]) if snapshot.rgb.size else []
        return model.build_inputs(
            messages,
            images,
            target_text=completion,
            proprio_window=snapshot.proprio_window,
            loss_span=loss_span,
        )

    # Precompute each pair's tokenized inputs AND the (fixed) reference logprobs once: the DPO
    # reference is the LoRA-OFF base model, constant throughout training, so its logprobs never
    # change — caching them avoids re-running the processor + the reference forward every epoch.
    cached = []
    for p in pairs:
        ch, rj = make_inputs(p.snapshot, p.chosen_text), make_inputs(p.snapshot, p.rejected_text)
        lp_ch_ref = model.completion_logprob(ch, reference=True).detach()
        lp_rj_ref = model.completion_logprob(rj, reference=True).detach()
        cached.append((ch, rj, lp_ch_ref, lp_rj_ref))

    opt = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    accum = int(cfg.train.grad_accum)
    epochs = int(cfg.train.epochs)
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(pairs))
        running, running_acc = 0.0, 0.0
        opt.zero_grad()
        for i, idx in enumerate(order):
            ch, rj, lp_ch_ref, lp_rj_ref = cached[idx]
            lp_ch_pol = model.completion_logprob(ch, reference=False)
            lp_rj_pol = model.completion_logprob(rj, reference=False)
            loss = dpo_loss(lp_ch_pol, lp_rj_pol, lp_ch_ref, lp_rj_ref, beta)
            (loss / accum).backward()
            running += float(loss)
            running_acc += float((lp_ch_pol - lp_ch_ref) > (lp_rj_pol - lp_rj_ref))
            if (i + 1) % accum == 0 or (i + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), float(cfg.train.grad_clip)
                )
                opt.step()
                opt.zero_grad()
        rec = {
            "epoch": epoch,
            "dpo_loss": running / max(1, len(order)),
            "pref_acc": running_acc / max(1, len(order)),
        }
        history.append(rec)
        print(
            f"[dpo] epoch {epoch} loss {rec['dpo_loss']:.4f} pref_acc {rec['pref_acc']:.3f}",
            flush=True,
        )

    model.save_adapter(out / "adapter_dpo")
    metrics = {
        "seed": seed,
        "beta": beta,
        "n_pairs": len(pairs),
        "epochs": epochs,
        "history": history,
        "wall_time_s": time.perf_counter() - t0,
    }
    (out / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
