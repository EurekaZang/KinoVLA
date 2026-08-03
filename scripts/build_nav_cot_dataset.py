#!/usr/bin/env python
"""Reasoning-grounded nav-CoT dataset from the DAgger raw states (#43) — real Oracle + §10 filter.

Takes the on-policy raw states from scripts/collect_nav_dagger.py (each: the visited nav state, its
RGB+proprio, and the privileged geometric teacher's action) and turns them into the trainable
nav-SFT dataset that kino_vla.vla.dataset_build.load_nav_examples consumes. The privileged geometric
teacher's action is GROUND TRUTH (the DAgger expert) and is the target. For each state a REAL Oracle
(the M6 ApiOracle: gpt-5.5 / Claude / Gemini, env-keyed) is shown the captured RGB + that privileged
action and writes the grounded first-person ``<Thought>`` that justifies it (it does NOT re-decide:
a zero-shot LLM never turns, which would drop every Turn). The truth-consistency filter then keeps
the sample unless the Oracle's REASONING contradicts the action (kino_vla.vla.nav_cot.thought_
coherent). The kept target = the Oracle's reasoning thought + the teacher's geometrically-exact
action — so the VLA learns *why* it turns (active perception) with a perfectly grounded label.

CPU/network (no GPU/Isaac): run after collect_nav_dagger.py. --dry-run skips the API (templated
thought, every-sample-kept) to validate the plumbing/format offline.

    python scripts/build_nav_cot_dataset.py --raw <raw_dir> --out outputs/vla/nav_dagger
    # ^ prefix with OPENAI_API_KEY=... for the real Oracle run; add --dry-run to skip the API.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from collections import Counter

import numpy as np

from kino_vla.data.oracle import ApiOracle
from kino_vla.utils.config import REPO_ROOT, load_config
from kino_vla.vla.nav_cot import (
    KEEP,
    extract_thought,
    format_nav_target,
    teacher_action_brief,
    thought_coherent,
)
from kino_vla.vla.prompt import PlannerContext, nav_user_text

# The privileged geometric teacher's action is GROUND TRUTH (the DAgger expert); the Oracle's job is
# to write the grounded REASONING that justifies it (we do NOT let a zero-shot LLM re-pick the
# action: it never turns, which would drop every Turn). Reasoning-grounded distillation of a label.
_NAV_ORACLE_SYSTEM = (
    "You are annotating a Unitree Go2's nominal-navigation dataset. You are shown the body-camera "
    "RGB (a forward-down view of the ground ahead), the goal direction, the map context, and the "
    "PRIVILEGED correct next navigation action computed by a route-around oracle that can see the "
    "full map. Write ONE first-person sentence of navigation reasoning that justifies THAT action "
    "from what is visible (name the hazard/clear ground you see and why this action makes progress "
    "around it toward the goal). Output EXACTLY: <Thought>one sentence</Thought> and nothing else."
)


def _balance_by_operator(records: list[dict], cap: int) -> list[dict]:
    """Per-operator balanced subset (deterministic): within each operator, round-robin across the
    teacher KINDS (turn/waypoint) by sorted sample_id until ``cap`` is reached, so each operator
    contributes ~uniformly and Turn is never starved. Operators with < cap states are kept whole."""
    import collections

    by_op: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_op[r["operator_name"]].append(r)
    out: list[dict] = []
    for op in sorted(by_op):
        recs = by_op[op]
        if len(recs) <= cap:
            out.extend(sorted(recs, key=lambda r: r["sample_id"]))
            continue
        by_kind: dict[str, list[dict]] = collections.defaultdict(list)
        for r in sorted(recs, key=lambda r: r["sample_id"]):
            by_kind[r["teacher"]["kind"]].append(r)
        picked: list[dict] = []
        idx = {k: 0 for k in by_kind}
        kinds = sorted(by_kind)
        while len(picked) < cap and any(idx[k] < len(by_kind[k]) for k in kinds):
            for k in kinds:
                if idx[k] < len(by_kind[k]) and len(picked) < cap:
                    picked.append(by_kind[k][idx[k]])
                    idx[k] += 1
        out.extend(picked)
    return out


def _dry_thought(teacher: dict) -> str:
    if teacher["kind"] == "waypoint":
        return "Clear ground continues toward the goal on the open side; steer to it."
    return "No clear ground straight ahead; rotate in place to bring the route into view."


def main() -> int:
    ap = argparse.ArgumentParser(description="Reasoning-grounded nav-CoT builder (#43)")
    ap.add_argument("--raw", default="outputs/vla/nav_dagger_raw", help="collect_nav_dagger.py out")
    ap.add_argument("--out", default="outputs/vla/nav_dagger")
    ap.add_argument("--n-images", type=int, default=1, help="RGB frames attached to the Oracle")
    ap.add_argument("--dry-run", action="store_true", help="no API: templated thought, all kept")
    ap.add_argument("--limit", type=int, default=None, help="cap #states (smoke: first N)")
    ap.add_argument(
        "--concurrency", type=int, default=8, help="parallel Oracle calls (I/O-bound; ~Nx faster)"
    )
    ap.add_argument(
        "--per-operator-cap",
        type=int,
        default=None,
        help="balanced subset: cap states PER operator (kind-balanced, deterministic) to bound the "
        "xhigh Oracle calls while keeping uniform operator coverage; small operators kept whole",
    )
    args = ap.parse_args()

    pcfg = load_config("data/hindsight.yaml")  # for ApiOracle.from_config (oracle.api block)
    raw = REPO_ROOT / args.raw
    records = [json.loads(ln) for ln in (raw / "raw_meta.jsonl").read_text().splitlines() if ln]
    npz = np.load(raw / "raw_frames.npz")
    if args.per_operator_cap is not None:
        records = _balance_by_operator(records, int(args.per_operator_cap))
    if args.limit is not None:
        records = records[: int(args.limit)]

    # Run the (independent, I/O-bound) Oracle calls CONCURRENTLY — quality-neutral, ~Nx wall-clock.
    texts: dict[str, str | None] = {}
    if not args.dry_run:
        complete = ApiOracle.from_config(pcfg)._complete  # transport: (messages, images) -> str

        def _oracle_call(rec: dict) -> tuple[str, str | None]:
            sid = rec["sample_id"]
            ctx = PlannerContext(
                monitor_channel="clock",
                pose_xy=tuple(rec["pose_xy"]),
                prior_outputs=list(rec.get("prior_outputs", [])),
                map_note=rec.get("map_note", ""),
            )
            # String-content prompt (Oracle transport) + the captured RGB; text route ⇒ no
            # <kino_tokens> placeholder. The privileged teacher's action is given; the Oracle
            # justifies it (it does NOT re-decide ⇒ Turns are kept, not dropped).
            base = nav_user_text(ctx, float(rec["goal_bearing_deg"]), route="text")
            user = (
                base + f"\n- PRIVILEGED correct next action: {teacher_action_brief(rec['teacher'])}"
            )
            msgs = [
                {"role": "system", "content": _NAV_ORACLE_SYSTEM},
                {"role": "user", "content": user},
            ]
            rgb = npz[f"{sid}__rgb"]
            images = [rgb[i] for i in range(-min(args.n_images, len(rgb)), 0)]
            try:
                return sid, complete(msgs, images)
            except Exception as e:  # noqa: BLE001 - a failed call ⇒ oracle_error, never crash the run
                print(f"[nav-cot] oracle_error {sid}: {type(e).__name__}: {e}", flush=True)
                return sid, None

        with cf.ThreadPoolExecutor(max_workers=int(args.concurrency)) as ex:
            done = 0
            for sid, text in ex.map(_oracle_call, records):
                texts[sid] = text
                done += 1
                if done % 25 == 0:
                    print(f"[nav-cot] oracle {done}/{len(records)}", flush=True)

    kept: list[dict] = []
    frames: dict[str, np.ndarray] = {}
    reasons: Counter = Counter()
    for rec in records:
        sid = rec["sample_id"]
        teacher = rec["teacher"]
        rgb = npz[f"{sid}__rgb"]  # (1, H, W, 3)
        if args.dry_run:
            thought = _dry_thought(teacher)
            reasons[KEEP] += 1
        else:
            oracle_text = texts.get(sid)
            if oracle_text is None:
                reasons["oracle_error"] += 1
                continue
            thought = extract_thought(oracle_text)
            # The ACTION is the privileged teacher's (ground truth, kept); the filter only drops
            # REASONING that contradicts it (the nav §10 truth-consistency check, on the rationale).
            if not thought_coherent(thought, teacher["kind"]):
                reasons["drop_incoherent"] += 1
                continue
            reasons[KEEP] += 1
        kept.append(
            {
                "sample_id": sid,
                "messages": rec["messages"],  # the deployed LATENT nav prompt (train/serve match)
                "target_text": format_nav_target(thought, teacher),
                "kind": teacher["kind"],  # "turn" | "waypoint" (load_nav_examples primitive_truth)
                "scenario": rec.get("scenario"),
                "source": rec.get("source"),
                "verdict_tag": rec.get("verdict_tag"),
            }
        )
        frames[f"{sid}__rgb"] = rgb.astype(np.float32)
        frames[f"{sid}__proprio"] = npz[f"{sid}__proprio"].astype(np.float32)

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "nav_meta.jsonl").write_text("\n".join(json.dumps(r) for r in kept))
    np.savez_compressed(out / "nav_frames.npz", **frames)
    n_turn = sum(r["kind"] == "turn" for r in kept)
    n_err = reasons.get("oracle_error", 0)
    n_dropped = len(records) - len(kept) - n_err  # filter rejections only (exclude API failures)
    card = {
        "n_raw": len(records),
        "n_kept": len(kept),
        "n_turn": int(n_turn),
        "n_waypoint": int(len(kept) - n_turn),
        "per_operator_kept": dict(Counter(r.get("scenario") for r in kept)),
        "reject_by_reason": dict(reasons),
        "n_oracle_error": int(n_err),
        "reject_rate": round(n_dropped / max(1, len(kept) + n_dropped), 4),  # M6 convention
        "oracle": "dry_run" if args.dry_run else "api",
    }
    (out / "nav_card.json").write_text(json.dumps(card, indent=2))
    print(f"[nav-cot] DONE -> {out}: {json.dumps(card)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
