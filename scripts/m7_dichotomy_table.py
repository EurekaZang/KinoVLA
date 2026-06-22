#!/usr/bin/env python
"""Consolidate the closed-loop dichotomy arms into a table (spec §2.5, Gap-3 / §6 #34).

Reads each arm's ``isaac_rollout_summary.json`` and reports, per arm: closed-loop attribution
accuracy (first_attribution vs the privileged truth), no-fall (safety) rate, and goal-reached rate,
both aggregate and per operator. The decisive comparisons:

- **B5+probe vs B5-probe** — does the active-sensing probe fix closed-loop attribution (#34c)?
- **B5 vs B2 (rule-FSM)** — does cause attribution + operator-specific recovery beat the cause-blind
  detour on safety (the §2.5 Recoverability Dichotomy)?
- **B5 (latent) vs B3 (vision-only)** — is proprioception load-bearing in the loop (Gap-1)?

    python scripts/m7_dichotomy_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

EXPECTED = {
    "O1_mu_field": "low_friction",
    "O2_compliance": "compliant_terrain",
    "O3_collapse": "region_collapse",
    "O4_tether": "adhesion",
    "O5_payload": "overload",
}
ARMS = [
    ("b5_latent", "B5 full (latent + probe)"),
    ("b5_noprobe", "B5 latent, NO probe (Gap-3 ablation)"),
    ("b4_text", "B4 text route (+ probe)"),
    ("b3_vision", "B3 vision-only (+ probe)"),
    ("b2_fsm", "B2 rule-FSM (cause-blind)"),
]
ROOT = Path("outputs/vla/dichotomy")


def _load(arm: str) -> list[dict]:
    p = ROOT / arm / "isaac_rollout_summary.json"
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("rows", [])


def _rate(rows: list[dict], pred: str) -> float:
    vals = [r for r in rows if r.get("temp", 0.0) == 0.0]
    if not vals:
        return float("nan")
    if pred == "attr":
        hit = sum(1 for r in vals if r.get("first_attribution") == EXPECTED.get(r["scenario"]))
    elif pred == "nofall":
        hit = sum(1 for r in vals if not r.get("fell"))
    elif pred == "reached":
        hit = sum(1 for r in vals if r.get("reached"))
    else:  # success
        hit = sum(1 for r in vals if r.get("success"))
    return hit / len(vals)


def main() -> int:
    out = ["# Closed-loop Recoverability Dichotomy (Isaac Go2, 5 B-class ops × seeds)", ""]
    out.append("| Arm | attribution acc | no-fall (safe) | reached | success |")
    out.append("|---|---|---|---|---|")
    per_arm_attr: dict[str, dict] = {}
    for arm, label in ARMS:
        rows = _load(arm)
        if not rows:
            out.append(f"| {label} | — (pending) | | | |")
            continue
        out.append(
            f"| {label} | {_rate(rows, 'attr'):.2f} | {_rate(rows, 'nofall'):.2f} "
            f"| {_rate(rows, 'reached'):.2f} | {_rate(rows, 'success'):.2f} |"
        )
        per_arm_attr[arm] = {
            op: [r.get("first_attribution") for r in rows if r["scenario"] == op] for op in EXPECTED
        }

    out += ["", "## Per-operator first_attribution (closed loop)", ""]
    out.append("| op (truth) | " + " | ".join(a for a, _ in ARMS) + " |")
    out.append("|---|" + "|".join("---" for _ in ARMS) + "|")
    for op, truth in EXPECTED.items():
        cells = []
        for arm, _ in ARMS:
            preds = per_arm_attr.get(arm, {}).get(op, [])
            if not preds:
                cells.append("—")
            else:
                ok = sum(1 for p in preds if p == truth)
                top = max(set(preds), key=preds.count)
                cells.append(f"{top} {ok}/{len(preds)}")
        out.append(f"| {op} → {truth} | " + " | ".join(cells) + " |")

    text = "\n".join(out)
    (ROOT / "TABLE.md").write_text(text)
    print(text)
    print(f"\nwrote {ROOT / 'TABLE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
