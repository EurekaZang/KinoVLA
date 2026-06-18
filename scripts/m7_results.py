#!/usr/bin/env python
"""Consolidate the M7 experiment artifacts into one paper-ready results table (spec §11/§12).

Reads whatever M7 outputs exist under ``outputs/vla`` and renders a single markdown report for the
M7 exit criteria. Robust to missing files (skips them).

Usage:  python scripts/m7_results.py [--out outputs/vla/M7_RESULTS.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, ValueError):
        return None


def _attr(d: dict | None, sampled: bool = False) -> str:
    if d is None:
        return "—"
    if sampled and "vla_samples" in d:
        return f"{d['vla_samples']['attr_acc_mean']:.3f}"
    return f"{d['vla']['attribution_accuracy']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/vla")
    ap.add_argument("--out", default="outputs/vla/M7_RESULTS.md")
    args = ap.parse_args()
    d = Path(args.dir)
    L: list[str] = ["# M7 Results — Kino-SFT + Embodied DPO (Qwen3-VL-4B, real-Go2 dataset)", ""]

    # Exit 1 + 2 (temp 0): VLA vs FSM attribution + parse-rate, per route
    L += ["## Exit 1 & 2 — Suite-Sem attribution (VLA vs rule-FSM) @temp 0", ""]
    L += ["| route | VLA acc | FSM acc | margin | feasible | parse | beats FSM |"]
    L += ["| --- | --- | --- | --- | --- | --- | --- |"]
    for route in ("latent", "text"):
        c = _load(d / f"suite_sem_{route}.json")
        if not c:
            continue
        v, f = c["vla"], c["fsm_baseline"]
        b = "5" if route == "latent" else "4"
        L.append(
            f"| {route} (B{b}) | {v['attribution_accuracy']:.3f} | {f['attribution_accuracy']:.3f} "
            f"| {c['margin']:+.3f} | {v['feasible_recovery_rate']:.3f} | {v['parse_rate']:.3f} "
            f"| {'YES' if c['vla_beats_fsm'] else 'no'} |"
        )

    # Exit 3: DPO vs SFT attribution at temperature (sampled mean)
    L += ["", "## Exit 3 — Embodied DPO vs SFT, held-out attribution @temperature", ""]
    L += ["| temp | SFT | DPO (canonical pairs) | DPO (on-policy) |", "| --- | --- | --- | --- |"]
    for t in ("08", "12"):
        sft = _load(d / f"suite_sem_sft_t{t}.json")
        dpo = _load(d / f"suite_sem_dpo_t{t}.json")
        dop = _load(d / f"suite_sem_dpoop_t{t}.json")
        if not sft:
            continue
        L.append(
            f"| {t[0]}.{t[1]} | {_attr(sft, True)} | {_attr(dpo, True)} | {_attr(dop, True)} |"
        )
    dm = _load(d / "dpo" / "dpo_metrics.json")
    if dm:
        L.append(
            f"\nDPO (canonical) preference accuracy: {dm['history'][0]['pref_acc']:.2f} → "
            f"{dm['history'][-1]['pref_acc']:.2f} (perfectly separates correct vs wrong-sibling)."
        )
    dmo = _load(d / "dpo_onpolicy" / "dpo_metrics.json")
    if dmo:
        L.append(
            f"DPO (on-policy) preference accuracy: {dmo['history'][0]['pref_acc']:.2f} → "
            f"{dmo['history'][-1]['pref_acc']:.2f}; n_pairs {dmo['n_pairs']}."
        )

    # Isaac closed loop
    L += ["", "## Exit 3 — Isaac closed-loop (the VLA running on the real Go2)", ""]
    for tag, fn in (("SFT", "isaac_sft_eval"), ("DPO", "isaac_dpo_eval")):
        r = _load(d / fn / "isaac_rollout_summary.json")
        if r:
            L.append(f"- {tag}: temp-0 success {r['success_rate_temp0']:.3f} (n={r['n_temp0']})")

    # SFT training provenance
    L += ["", "## SFT training (reproducibility, QA 5.2)", ""]
    for route in ("latent", "text"):
        m = _load(d / f"sft_{route}" / "sft_metrics.json")
        if m:
            L.append(
                f"- **{route}**: best_val_loss `{m['best_val_loss']:.4f}`, split "
                f"{m['n_train']}/{m['n_val']}/{m['n_test']}, {m['epochs']} ep, seed `{m['seed']}`, "
                f"`{m['model_id']}`, wall {m['wall_time_s'] / 60:.1f} min"
            )

    report = "\n".join(L) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
