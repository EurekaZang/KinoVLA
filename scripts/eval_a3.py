#!/usr/bin/env python
"""A3 — bidirectional conflict battery: the taxonomy × agents heatmap + override dose-response.

THE paper's central figure (experiments_design.md §4 A3). Merges the frozen A0.3 corpus (T1/T2/T4/T5
+ O8) with the A3 T3 extension (O7 looks_safe μ-sweep + O7 reverse probe) and scores every agent
across all cells, then renders two figures:

  1. HEATMAP — attribution accuracy per (agent × taxonomy cell). The predicted COMPLEMENTARY BLOCK
     failures: B1 (proprio-only) dies on T2 (vision-decidable); B-V (vision-only) dies on T3
     (vision-misleading); only the fusion agents cover all cells. "Feel It, See It" in one image.
  2. DOSE-RESPONSE — O7 looks_safe P(agent overrides the benign 'solid ground' visual prior ⇒
     low_friction) vs μ. A sigmoid threshold = evidence-weighing; a flat line = modality dominance.

Metric: attribution accuracy = P(attribution == privileged truth). For nominal rows (T5, O7 reverse)
the truth is `nominal` ⇒ the agent must ABSTAIN (attribute nominal) to score — the abstention cell.

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 KINOVLA_MODEL_ID=... python scripts/eval_a3.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CELLS = ["T1", "T2", "T3", "T4", "T5"]
CELL_NAMES = {
    "T1": "T1 agree",
    "T2": "T2 conflict\n(vision-true)",
    "T3": "T3 conflict\n(proprio-true)",
    "T4": "T4 fine-struct",
    "T5": "T5 nominal\n(continue)",
}
# T3 sub-cells (different privileged truths); used for the detail table + the dose-response.
T3_SUB = {"looks_safe": "O7 looks_safe", "O8": "O8 invisible", "reverse": "O7 reverse"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


@dataclass(frozen=True)
class Snap:
    sid: str
    cell: str
    t3_sub: str
    truth: str
    operator: str
    admissible: frozenset[str]
    appearance_id: str
    appearance_split: str
    pair_id: str
    mu: float | None
    snapshot: object  # kino_vla.data.schema.Snapshot


def _t3_sub(rec) -> str:
    if rec.get("a3_direction") == "looks_safe":
        return "looks_safe"
    if rec.get("a3_direction") == "reverse":
        return "reverse"
    if rec["snapshot"]["operator_name"] == "O8_invisible_collider":
        return "O8"
    return "other"


def load_merged(a03_dir: str, t3_dir: str) -> list[Snap]:
    from kino_vla.vla.dataset_build import _snapshot_from_record

    snaps: list[Snap] = []
    for d in (a03_dir, t3_dir):
        lines = Path(d, "samples.jsonl").read_text().splitlines()
        recs = [json.loads(ln) for ln in lines if ln]
        npz = np.load(Path(d, "frames.npz"))
        for rec in recs:
            sid = rec["sample_id"]
            if f"{sid}__rgb" not in npz:
                continue
            frames = {k: npz[f"{sid}__{k}"] for k in ("rgb", "depth", "proprio")}
            cell = rec.get("taxonomy_cell", "?")
            # T5 is the explicitly pre-registered A3 semantic-abstention task.  The underlying
            # simulator and A0 registry retain the mild physical operator category and its
            # low-level recovery semantics, while A3's decision truth is ``nominal`` and its only
            # admissible semantic action is ``continue``.  Keep that decision overlay explicit.
            truth = "nominal" if cell == "T5" else rec["ground_truth"]["category"]
            admissible = (
                frozenset({"continue"})
                if cell == "T5"
                else frozenset(rec.get("admissible_recovery_set") or [])
            )
            snaps.append(
                Snap(
                    sid=sid,
                    cell=cell,
                    t3_sub=_t3_sub(rec),
                    truth=truth,
                    operator=str(rec["snapshot"]["operator_name"]),
                    admissible=admissible,
                    appearance_id=str(rec.get("appearance_id", "")),
                    appearance_split=str(rec.get("appearance_split", "")),
                    pair_id=str(rec.get("pair_id") or rec.get("ambiguity_pair") or ""),
                    mu=rec.get("a3_mu"),
                    snapshot=_snapshot_from_record(rec, frames),
                )
            )
    return snaps


def _scored_row(s: Snap, dec) -> dict:
    parsed = bool(dec.ok and dec.annotation is not None)
    attr = dec.attribution if parsed else None
    prim = dec.primitive_name if parsed else None
    params = dict(dec.annotation.primitive.params) if parsed else {}
    attr_ok = parsed and attr == s.truth
    prim_feasible = parsed and prim in s.admissible
    return {
        "sid": s.sid,
        "cell": s.cell,
        "t3_sub": s.t3_sub,
        "truth": s.truth,
        "operator": s.operator,
        "appearance_id": s.appearance_id,
        "appearance_split": s.appearance_split,
        "pair_id": s.pair_id,
        "admissible_recovery_set": sorted(s.admissible),
        "attribution": attr,
        "primitive": prim,
        "primitive_params": params,
        "action_params_observed": bool(parsed),
        "attr_ok": attr_ok,
        "prim_feasible": prim_feasible,
        "joint_ok": attr_ok and prim_feasible,
        "mu": s.mu,
        "parsed": parsed,
    }


def score_agent(policy, snaps: list[Snap], *, input_key=None) -> list[dict]:
    """Score snapshots, optionally reusing a deterministic decision for byte-identical inputs."""
    rows = []
    decisions = {}
    for s in snaps:
        key = input_key(s.snapshot) if input_key is not None else None
        if key is None or key not in decisions:
            dec = policy.decide(s.snapshot)
            if key is not None:
                decisions[key] = dec
        else:
            dec = decisions[key]
        rows.append(_scored_row(s, dec))
    if input_key is not None:
        print(
            f"[a3] evaluated {len(decisions)} unique model inputs for {len(snaps)} rows", flush=True
        )
    return rows


def vla_input_key(
    snapshot, *, cfg, route: str, n_images: int, proprio_detail: str, mask: bool
) -> str:
    """Hash the exact deterministic ModelVlaPolicy inputs for safe duplicate elimination."""
    from kino_vla.vla.prompt import build_messages, context_from_snapshot

    detail = "none" if (mask and route == "text") else proprio_detail
    ctx = context_from_snapshot(
        snapshot,
        route=route,
        reveal_appearance=False,
        proprio_detail=detail,
        map_note="",
    )
    messages = build_messages(ctx, cfg, route=route, n_images=n_images)
    h = hashlib.sha256(json.dumps(messages, sort_keys=True, default=str).encode())
    images = list(snapshot.rgb[-n_images:]) if snapshot.rgb.size else []
    for image in images:
        array = np.ascontiguousarray(image)
        h.update(str((array.shape, array.dtype)).encode())
        h.update(array.tobytes())
    if route == "latent" and not mask:
        proprio = np.ascontiguousarray(snapshot.proprio_window)
        h.update(str((proprio.shape, proprio.dtype)).encode())
        h.update(proprio.tobytes())
    return h.hexdigest()


def upgrade_cached_rows(rows: list[dict], snaps: list[Snap]) -> list[dict]:
    """Upgrade legacy A3 rows with frozen-corpus metadata and the current scoring truth.

    This deliberately does *not* invent action parameters that old files failed to persist.  A
    legacy ``Switch_Gait`` therefore remains marked ``action_params_observed=false`` and A4's
    physical-action composition must reject it until the model is re-evaluated.
    """
    by_sid = {s.sid: s for s in snaps}
    upgraded: list[dict] = []
    for old in rows:
        row = dict(old)
        s = by_sid.get(str(row.get("sid", "")))
        if s is None:
            raise KeyError(f"cached A3 row is not present in the frozen corpus: {row.get('sid')}")
        row.update(
            {
                "cell": s.cell,
                "t3_sub": s.t3_sub,
                "truth": s.truth,
                "operator": s.operator,
                "appearance_id": s.appearance_id,
                "appearance_split": s.appearance_split,
                "pair_id": s.pair_id,
                "admissible_recovery_set": sorted(s.admissible),
                "mu": s.mu,
            }
        )
        parsed = bool(row.get("parsed") and row.get("attribution") is not None)
        prim = row.get("primitive")
        row["attr_ok"] = bool(parsed and row.get("attribution") == s.truth)
        row["prim_feasible"] = bool(parsed and prim in s.admissible)
        row["joint_ok"] = bool(row["attr_ok"] and row["prim_feasible"])
        if "primitive_params" in row:
            row["primitive_params"] = dict(row.get("primitive_params") or {})
            row["action_params_observed"] = bool(row.get("action_params_observed", parsed))
        else:
            row["primitive_params"] = {}
            row["action_params_observed"] = False
        upgraded.append(row)
    return upgraded


def _acc(rows: list[dict]) -> dict:
    """Attribution accuracy + Wilson CI over an ALREADY-filtered row list."""
    n = len(rows)
    k = sum(1 for r in rows if r["attr_ok"])
    lo, hi = wilson(k, n)
    return {"n": n, "acc": round(k / max(1, n), 3), "ci": [lo, hi], "k": k}


def cell_acc(rows: list[dict], cell: str) -> dict:
    return _acc([r for r in rows if r["cell"] == cell])


def main() -> None:
    ap = argparse.ArgumentParser(description="A3 bidirectional conflict battery (heatmap + dose)")
    ap.add_argument("--a03", default="outputs/eval/a0/corpus")
    ap.add_argument("--t3", default="outputs/eval/a3/corpus_t3")
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--b1-model", default="outputs/eval/e2/b1_attributor/monitor")
    ap.add_argument("--b5-unshaped", default="outputs/vla/sft_latent/adapter_best")
    ap.add_argument("--b5-conflict", default="outputs/eval/a2/b5_conflict_heldout/adapter_best")
    ap.add_argument("--b5-conflict-bi", default="outputs/eval/a3/b5_conflict_bi/adapter_best")
    ap.add_argument("--bt-adapter", default="outputs/eval/a2/b_text/adapter_best")
    ap.add_argument("--bf-model", default="outputs/eval/a2/b_fusion/bf.pt")
    ap.add_argument(
        "--agents",
        default="B1,B-F,zero-shot,B-T,B-V,B5-unshaped,B5-conflict,B5-conflict-bi",
    )
    ap.add_argument("--out", default="outputs/eval/a3")
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="skip agent runs; rebuild the heatmap/dose from existing per_item_*.json",
    )
    ap.add_argument(
        "--refresh-legacy-gaits",
        action="store_true",
        help="re-evaluate only cached Switch_Gait rows whose mode was not persisted, then merge",
    )
    ap.add_argument("--only-cell", default="", help="only eval this cell's snapshots (e.g. T3)")
    args = ap.parse_args()
    if args.recompute and args.refresh_legacy_gaits:
        ap.error("--recompute and --refresh-legacy-gaits are mutually exclusive")
    if args.only_cell and args.refresh_legacy_gaits:
        ap.error("--only-cell and --refresh-legacy-gaits are mutually exclusive")
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.utils.config import load_config

    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    snaps = load_merged(args.a03, args.t3)
    if args.only_cell:  # restrict to one cell (the others are frozen/unchanged on disk)
        snaps = [s for s in snaps if s.cell == args.only_cell]
        print(f"[a3] --only-cell {args.only_cell}: {len(snaps)} snapshots", flush=True)
    by_cell = {c: sum(1 for s in snaps if s.cell == c) for c in CELLS}
    print(f"[a3] merged corpus: {len(snaps)} snapshots | per cell: {by_cell}", flush=True)
    sub_n = {s: sum(1 for x in snaps if x.t3_sub == s) for s in T3_SUB if s != "other"}
    print(f"[a3] T3 sub-cells: {sub_n}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_agent: dict[str, list[dict]] = {}
    legacy_by_agent: dict[str, list[dict]] = {}

    if args.refresh_legacy_gaits:
        for agent in agents:
            path = out / f"per_item_{agent.replace('-', '_')}.json"
            if not path.exists():
                raise FileNotFoundError(f"legacy refresh requires {path}")
            legacy_by_agent[agent] = upgrade_cached_rows(json.loads(path.read_text()), snaps)

    def eval_snaps(agent: str) -> list[Snap]:
        if not args.refresh_legacy_gaits:
            return snaps
        legacy = legacy_by_agent[agent]
        refresh_ids = {
            row["sid"]
            for row in legacy
            if row.get("primitive") == "Switch_Gait"
            and not row.get("action_params_observed", False)
        }
        selected = [snap for snap in snaps if snap.sid in refresh_ids]
        print(f"[a3] {agent}: refreshing {len(selected)} legacy gait rows", flush=True)
        return selected

    def merge_and_checkpoint(agent: str, fresh: list[dict]) -> None:
        if args.refresh_legacy_gaits:
            merged = {row["sid"]: row for row in legacy_by_agent[agent]}
            merged.update({row["sid"]: row for row in fresh})
            per_agent[agent] = [merged[snap.sid] for snap in snaps]
        else:
            per_agent[agent] = fresh
        path = out / f"per_item_{agent.replace('-', '_')}.json"
        path.write_text(json.dumps(per_agent[agent], indent=2))
        print(f"[a3] checkpointed {path} ({len(per_agent[agent])} rows)", flush=True)

    if args.recompute:  # rebuild the heatmap/dose from cached per_item_*.json (no agent runs)
        for a in agents:
            f = out / f"per_item_{a.replace('-', '_')}.json"
            if f.exists():
                per_agent[a] = upgrade_cached_rows(json.loads(f.read_text()), snaps)
                print(f"[a3] loaded cached {a} ({len(per_agent[a])} items)", flush=True)
            else:
                print(f"[a3] WARNING: no cached per_item for {a}; skipping", flush=True)
        agents = [a for a in agents if a in per_agent]

    # ---- B1 (CPU) ----
    if not args.recompute and "B1" in agents:
        selected = eval_snaps("B1")
        from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy

        b1 = ProprioBaselinePolicy.from_deployed(pcfg, tax, model_path=args.b1_model, device="cpu")
        merge_and_checkpoint("B1", score_agent(b1, selected))
        print("[a3] B1 done", flush=True)

    # ---- B-F (CPU, CLIP+proprio) ----
    if not args.recompute and "B-F" in agents:
        selected = eval_snaps("B-F")
        from kino_vla.eval.fusion_baseline import FusionBaselinePolicy

        bf = FusionBaselinePolicy.load(args.bf_model, pcfg, tax)
        merge_and_checkpoint("B-F", score_agent(bf, selected))
        print("[a3] B-F done", flush=True)

    # ---- VLA agents (GPU) ----
    vla_ids = {"B5-unshaped", "B5-conflict", "B-V", "B-T", "B5-conflict-bi", "zero-shot"}
    vla = [a for a in agents if a in vla_ids]
    if not args.recompute and vla:
        import torch

        from kino_vla.vla.model import KinoVLA
        from kino_vla.vla.planner import ModelVlaPolicy

        dev = "cuda" if torch.cuda.is_available() else "cpu"

        def load(route, adapter):
            vcfg = load_config(args.config, {"route": route})
            m = KinoVLA.from_pretrained(vcfg, device=dev, adapter_dir=adapter)
            m.eval()
            ni = int(vcfg.data.get("n_images", 1))
            return m, ni, str(vcfg.data.get("proprio_detail", "binned"))

        runs = [
            ("B5-unshaped", "latent", args.b5_unshaped, False),
            ("B5-conflict", "latent", args.b5_conflict, False),
            ("B-V", "latent", args.b5_conflict, True),  # mask_proprio on the conflict model
            ("B5-conflict-bi", "latent", args.b5_conflict_bi, False),
            ("B-T", "text", args.bt_adapter, False),
            ("zero-shot", "text", None, False),  # fresh LoRA (B=0) = base model
        ]
        loaded_key = None
        m = None
        ni = 1
        pdet = "binned"
        for name, route, adapter, mask in runs:
            if name not in agents:
                continue
            selected = eval_snaps(name)
            if not selected:
                per_agent[name] = legacy_by_agent[name]
                continue
            key = (route, adapter)
            if key != loaded_key:
                if m is not None:
                    del m
                    torch.cuda.empty_cache()
                m, ni, pdet = load(route, adapter)
                loaded_key = key
            pol = ModelVlaPolicy(
                m,
                pcfg,
                tax,
                route=route,
                n_images=ni,
                temperature=0.0,
                proprio_detail=pdet,
                mask_proprio=mask,
            )

            def key_fn(
                snapshot,
                route=route,
                ni=ni,
                pdet=pdet,
                mask=mask,
            ):
                return vla_input_key(
                    snapshot,
                    cfg=pcfg,
                    route=route,
                    n_images=ni,
                    proprio_detail=pdet,
                    mask=mask,
                )

            merge_and_checkpoint(name, score_agent(pol, selected, input_key=key_fn))
            print(f"[a3] {name} done", flush=True)
        if m is not None:
            del m
            torch.cuda.empty_cache()

    # ---- if --only-cell, merge the fresh cell rows with the cached non-cell rows on disk ----
    if args.only_cell:
        for a in list(per_agent):
            cached = out / f"per_item_{a.replace('-', '_')}.json"
            if cached.exists():
                old = upgrade_cached_rows(
                    json.loads(cached.read_text()), load_merged(args.a03, args.t3)
                )
                keep = [r for r in old if r["cell"] != args.only_cell]
                per_agent[a] = keep + per_agent[a]
                print(
                    f"[a3] merged {a}: {len(keep)} cached non-{args.only_cell} + "
                    f"{len(per_agent[a]) - len(keep)} fresh {args.only_cell}",
                    flush=True,
                )

    # ---- heatmap ----
    heatmap = {a: {c: cell_acc(per_agent[a], c) for c in CELLS} for a in agents}
    t3_detail = {
        a: {
            s: _acc([r for r in per_agent[a] if r["t3_sub"] == s])
            for s in ("looks_safe", "O8", "reverse")
        }
        for a in agents
    }

    # ---- dose-response: O7 looks_safe P(low_friction) vs μ ----
    dose = {}
    for a in agents:
        per_mu = {}
        mu_set = {r["mu"] for r in per_agent[a] if r["t3_sub"] == "looks_safe"}
        for mu in sorted(mu_set, reverse=True):
            rs = [r for r in per_agent[a] if r["t3_sub"] == "looks_safe" and r["mu"] == mu]
            k = sum(1 for r in rs if r["attribution"] == "low_friction")
            per_mu[str(mu)] = {
                "n": len(rs),
                "p_low_friction": round(k / max(1, len(rs)), 3),
                "ci": wilson(k, len(rs)),
            }
        dose[a] = per_mu

    manifest_a03 = json.loads(Path(args.a03, "frozen_manifest.json").read_text())
    result = {
        "a03_corpus_hash": manifest_a03.get("corpus_hash_sha256"),
        "t3_dir": args.t3,
        "commit": manifest_a03.get("commit"),
        "n_snaps": len(snaps),
        "per_cell_n": by_cell,
        "agents": agents,
        "heatmap": heatmap,
        "t3_detail": t3_detail,
        "dose_response": dose,
        "metric": "attribution accuracy = P(attribution == privileged truth); nominal rows require "
        "'nominal' (abstain/continue)",
    }
    (out / "a3_battery.json").write_text(json.dumps(result, indent=2))
    for a in agents:
        (out / f"per_item_{a.replace('-', '_')}.json").write_text(
            json.dumps(per_agent[a], indent=2)
        )

    # ---- printed heatmap ----
    print("\n=== A3 heatmap — attribution accuracy (P attribution == truth) ===")
    hdr = f"{'agent':<16}" + "".join(f"{CELL_NAMES[c].splitlines()[0]:>14}" for c in CELLS)
    print(hdr)
    for a in agents:
        row = f"{a:<16}" + "".join(f"{heatmap[a][c]['acc']:>14.2f}" for c in CELLS)
        print(row)
    print("\nT3 sub-cells (looks_safe/O8/reverse):")
    for a in agents:
        d = t3_detail[a]
        print(f"  {a:<16} " + " ".join(f"{s}={d[s]['acc']:.2f}(n{d[s]['n']})" for s in d))
    print("\nDose-response (O7 looks_safe P(low_friction) vs μ):")
    mus_all = sorted({m for d in dose.values() for m in d}, key=float, reverse=True)
    print(f"  {'agent':<16}" + "".join(f"{'μ=' + m:>10}" for m in mus_all))
    for a in agents:
        mus = sorted(dose[a], key=float, reverse=True)
        print(f"  {a:<16}" + "".join(f"{dose[a][m]['p_low_friction']:>10.2f}" for m in mus))

    _render_figures(out, agents, heatmap, dose, per_agent)
    print(f"\nwrote {out / 'a3_battery.json'} + heatmap.png + dose_response.png")


def _render_figures(out: Path, agents, heatmap, dose, per_agent) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[a3] matplotlib missing; skipping figures", flush=True)
        return

    # heatmap: rows=agents, cols=cells
    mat = np.array([[heatmap[a][c]["acc"] for c in CELLS] for a in agents])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(agents) + 2))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([CELL_NAMES[c] for c in CELLS], fontsize=9)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=9)
    for i, a in enumerate(agents):
        for j, c in enumerate(CELLS):
            cell = heatmap[a][c]
            ax.text(
                j,
                i,
                f"{cell['acc']:.2f}\nn={cell['n']}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    ax.set_title(
        "A3 taxonomy × agents — attribution accuracy\n"
        "(B1 dies T2; B-V dies T3; fusion covers all = Feel It, See It)"
    )
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(out / "heatmap.png", dpi=130)
    plt.close(fig)

    # dose-response: P(low_friction) vs μ, one line per agent
    mus = sorted({float(m) for d in dose.values() for m in d}, reverse=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for a in agents:
        ys = [dose[a].get(str(m), {}).get("p_low_friction", float("nan")) for m in mus]
        ax.plot(mus, ys, marker="o", label=a)
    ax.set_xlabel("O7 μ (proprio evidence strength — lower = more slippery)")
    ax.set_ylabel("P(agent attributes low_friction)")
    ax.set_title(
        "A3.4 override dose-response (O7 looks_safe)\n"
        "sigmoid threshold = evidence-weighing; flat = modality dominance"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "dose_response.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
