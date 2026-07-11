#!/usr/bin/env python
# ruff: noqa: E501
"""Generate A实验/A8.md from machine-readable A8 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kino_vla.eval.a7_ablation import config_hash, git_commit, load_yaml, repo_path


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt_cell(cell: dict[str, Any] | None) -> str:
    if not cell:
        return "—"
    return f"{cell.get('rate', float('nan')):.3f} [{cell.get('ci', ['?', '?'])[0]}, {cell.get('ci', ['?', '?'])[1]}] (n={cell.get('n', 0)})"


def render(cfg: dict[str, Any], config_path: str) -> str:
    out_root = repo_path(cfg["output_dir"])
    a8a = _load(out_root / "a8a" / "summary.json")
    a8b = _load(out_root / "a8b" / "summary.json") or _load(out_root / "a8b" / "blocked.json")
    audit = _load(out_root / "a8b_state_audit.json")
    manifest = _load(out_root / "download_manifest.json")
    ch = config_hash(config_path)
    commit = git_commit()

    lines: list[str] = []
    lines.append("# A8 — External Failure-Reasoning Benchmark")
    lines.append("")
    lines.append(f"- Config: `{config_path}` hash `{ch}`")
    lines.append(f"- Commit: `{commit}`")
    lines.append(f"- Output root: `{cfg['output_dir']}`")
    lines.append("")
    lines.append("## 1. Why")
    lines.append("")
    lines.append(
        "A8 tests external validity of cross-modal failure attribution. "
        "It does not re-prove C1–C3 on Kino-Fail. A8a provides citable public Guardian/FailCoT scores; "
        "A8b tests whether fusion/conflict training still helps when real robot-state is reattached on REFLECT multi-sensory episodes."
    )
    lines.append("")
    lines.append("## 2. Claim role")
    lines.append("")
    lines.append("| Claim | Role |")
    lines.append("|---|---|")
    lines.append("| C2 | Primary — Δ_fusion / Δ_conflict on A8b conflict strata |")
    lines.append("| C4 | Secondary — official OOD splits + optional calibration diagnostics |")
    lines.append("| C1 | Structural analogue via E2/E3 only |")
    lines.append("| C3 | Out of scope (owned by A4) |")
    lines.append("")
    lines.append("## 3. Method")
    lines.append("")
    lines.append("- Backbone for method rows: Qwen3-VL-4B + fixed LoRA budget in `configs/eval/a8.yaml`.")
    lines.append("- Guardian-8B: reference-only paper numbers, not a same-method ablation.")
    lines.append("- Leakage firewall: model inputs exclude failure_mode/reason/captions/rewards.")
    lines.append("- A8b admission gate: REFLECT multi-sensory state audit before headlines.")
    if manifest:
        lines.append(f"- Download manifest present: a8b_audit_pass={manifest.get('a8b_audit_pass')}")
    if audit:
        lines.append(
            f"- State audit: pass={audit.get('pass')} reason={audit.get('reason')} "
            f"multisensory_eps={audit.get('n_multisensory_episodes')}"
        )
    lines.append("")
    lines.append("## 4. Results")
    lines.append("")
    lines.append("### 4.1 A8a — Public Guardian / FailCoT")
    lines.append("")
    if not a8a:
        lines.append("_A8a summary missing — run `scripts/a8_eval.py --stage a8a` after datasets are built._")
    else:
        lines.append(f"- Status: **{a8a.get('status')}**")
        if a8a.get("note"):
            lines.append(f"- Note: {a8a['note']}")
        arms = a8a.get("arms") or {}
        # Guardian reference
        ref = arms.get("guardian_8b_paper") or a8a.get("reference")
        if ref:
            lines.append("")
            lines.append("#### Reference (Guardian-8B paper)")
            lines.append("")
            lines.append("| Benchmark | Exec | Plan |")
            lines.append("|---|---:|---:|")
            for name in ("RoboFail", "UR5-Fail", "RoboVQA"):
                block = ref.get(name) or {}
                if isinstance(block, dict):
                    lines.append(
                        f"| {name} | {block.get('exec', '—')} | {block.get('plan', '—')} |"
                    )
        zs = arms.get("zero_shot") or {}
        sft = arms.get("failcot_sft") or {}
        lines.append("")
        lines.append("#### Same-backbone rows (Qwen3-VL-4B)")
        lines.append("")
        lines.append("| Split | Arm | Status | Accuracy (Wilson) | macro-F1 |")
        lines.append("|---|---|---|---|---:|")
        for arm_name, arm in (("zero_shot", zs), ("failcot_sft", sft)):
            for split, cell in sorted((arm.get("splits") or {}).items()):
                status = cell.get("status", "unknown")
                if status == "heuristic_placeholder":
                    continue
                lines.append(
                    f"| {split} | {arm_name} | {status} | {_fmt_cell(cell.get('accuracy'))} | {cell.get('macro_f1', '—')} |"
                )
        if not (zs.get("splits")) and not (sft.get("splits")):
            lines.append("| — | — | — | no split rows yet | — |")
        if a8a.get("sft_minus_zs"):
            lines.append("")
            lines.append("#### FailCoT-SFT − zero-shot (paired same splits)")
            lines.append("")
            lines.append("| Split | ZS acc | SFT acc | Δacc | ZS F1 | SFT F1 | ΔF1 |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for split, d in sorted(a8a["sft_minus_zs"].items()):
                lines.append(
                    f"| {split} | {d.get('zs')} | {d.get('sft')} | {d.get('acc_delta')} | "
                    f"{d.get('zs_f1', '—')} | {d.get('sft_f1', '—')} | {d.get('f1_delta')} |"
                )
            lines.append("")
            lines.append(
                "- FailCoT-SFT was trained on stratified BDV2-Fail train (2000). "
                "Near-zero/negative OOD deltas are reported honestly; this is not sold as a free win."
            )
        if a8a.get("headline_ur5_zero_shot"):
            h = a8a["headline_ur5_zero_shot"]
            lines.append("")
            lines.append(
                f"- **Headline real ZS (UR5-Fail exec):** acc={h.get('acc')} "
                f"CI={h.get('ci')} macro-F1={h.get('macro_f1')} "
                f"failure_recall={h.get('failure_recall')} success_recall={h.get('success_recall')} n={h.get('n')}"
            )
    lines.append("")
    lines.append("### 4.2 A8b — Multi-sensory conflict transfer")
    lines.append("")
    if not a8b:
        lines.append("_A8b summary missing._")
    elif a8b.get("status") == "blocked":
        lines.append(f"- Status: **blocked**")
        lines.append(f"- Reason: {a8b.get('reason')}")
        lines.append("- No surrogate proprio headlines are reported.")
    else:
        lines.append(f"- Status: **{a8b.get('status')}**")
        if a8b.get("note"):
            lines.append(f"- Note: {a8b['note']}")
        summary = a8b.get("summary") or {}
        arms = summary.get("arms") or {}
        lines.append("")
        lines.append("| Arm | CBA | Acc E2 | Acc E3 | Overall |")
        lines.append("|---|---:|---:|---:|---|")
        for arm, st in sorted(arms.items()):
            lines.append(
                f"| {arm} | {st.get('cba', float('nan')):.3f} | {st.get('acc_e2', float('nan')):.3f} | "
                f"{st.get('acc_e3', float('nan')):.3f} | {_fmt_cell(st.get('overall'))} |"
            )
        if "delta_fusion_latent" in summary:
            lines.append(f"- Δ_fusion (latent): **{summary['delta_fusion_latent']:.3f}**")
        if "delta_fusion_concat" in summary:
            lines.append(f"- Δ_fusion (concat): **{summary['delta_fusion_concat']:.3f}**")
        if "delta_conflict" in summary:
            lines.append(f"- Δ_conflict: **{summary['delta_conflict']:.3f}**")
        if summary.get("e3_acc"):
            lines.append(f"- Acc_E3 by arm: `{summary['e3_acc']}`")
        if "delta_proprio_over_vision_e3" in summary:
            lines.append(f"- Δ proprio-over-vision (E3): **{summary['delta_proprio_over_vision_e3']:.3f}**")
        if "delta_fusion_e3" in summary:
            lines.append(f"- Δ_fusion on E3: **{summary['delta_fusion_e3']:.3f}**")
        if summary.get("mcnemar_conflict_vs_v"):
            lines.append(f"- McNemar latent_conflict vs V: `{summary['mcnemar_conflict_vs_v']}`")
        if summary.get("mcnemar_p_vs_v"):
            lines.append(f"- McNemar P vs V: `{summary['mcnemar_p_vs_v']}`")
        if summary.get("corpus_note"):
            lines.append(f"- Corpus note: {summary['corpus_note']}")
    lines.append("")
    lines.append("## 5. Claim bridge / falsifier")
    lines.append("")
    lines.append(
        "- **Supports C2** if A8b admission passes and Δ_fusion > 0 and/or Δ_conflict > 0 "
        "with McNemar/CI support concentrated on E2/E3."
    )
    lines.append(
        "- **Supports C4** if same-backbone A8a rows remain non-collapsed on OOD public splits."
    )
    lines.append(
        "- **Falsifier:** A8b pass but fusion ≤ best unimodal on CBA, or gains only on Agree/Nominal."
    )
    lines.append("")
    lines.append("## 6. Honest scope")
    lines.append("")
    lines.append("- Official Guardian track is image/text VQA; alone it cannot prove Kino-Tokens.")
    lines.append("- A8b depends on real REFLECT multi-sensory state; blocked ⇒ no proxy headlines.")
    lines.append("- Guardian-8B vs Qwen3-VL-4B is not a method ablation.")
    lines.append("- A8c matched-pair C2ST and A8d physical Go2 are out of scope here.")
    lines.append("- Heuristic/diagnostic rows are labeled and not sold as trained-adapter headlines.")
    lines.append("")
    lines.append("## Appendix — Allowed inputs / eval-only metadata")
    lines.append("")
    lines.append("| Allowed model inputs | Eval-only / strata / oracle |")
    lines.append("|---|---|")
    lines.append(
        "| images, task_instruction, raw robot_state / state_summary | failure_mode, failure_reason, reward*, captions, GT object ids, stratum |"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/a8.yaml")
    ap.add_argument("--out", default="A实验/A8.md")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    md = render(cfg, args.config)
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
