#!/usr/bin/env python
"""O4↔O2 proprioceptive-statistics matching (spec §8.1 P4) — paper artifact.

Builds the matched compliance/tether pair, rolls each through its region, and reports
the tangential-resistance-vs-displacement curve match, the base-height / slip trace
match, and the appearance separation — the constructive evidence that proprioception
cannot disambiguate them (so the visual map is irreplaceable). Writes a markdown report
to outputs/map/ambiguity_match.md and prints PASS/FAIL against configs tolerances.
"""

from __future__ import annotations

from pathlib import Path

from kino_vla.eval.ambiguity import build_matched_pair, match_ambiguity_pair
from kino_vla.utils.config import REPO_ROOT, load_config


def main() -> int:
    o2, o4 = build_matched_pair()
    result = match_ambiguity_pair(o2, o4)
    tol = load_config("operators/apply_tolerances.yaml").ambiguity_o4_o2

    checks = {
        "resistance curve match [N]": (
            result.resistance_curve_max_abs_diff_n,
            float(tol.resistance_curve_max_abs_diff_n),
            "<=",
        ),
        "base-height trace match [m]": (
            result.base_height_max_abs_diff_m,
            float(tol.base_height_max_abs_diff_m),
            "<=",
        ),
        "slip trace match": (
            result.slip_max_abs_diff,
            float(tol.slip_max_abs_diff),
            "<=",
        ),
        "appearance separation (1−cos)": (
            result.appearance_separation,
            float(tol.min_appearance_separation),
            ">=",
        ),
    }
    passed = all((v <= bound) if op == "<=" else (v >= bound) for v, bound, op in checks.values())

    lines = [
        "# O4↔O2 Ambiguity-Pair Matching (spec §8.1 P4)",
        "",
        "Constructive proof that elastic adhesion (O4) and compliance sink (O2) are",
        "proprioceptively indistinguishable — only the visual semantic map separates them.",
        "",
        "| statistic | measured | bound | verdict |",
        "|---|---|---|---|",
    ]
    for name, (val, bound, op) in checks.items():
        ok = (val <= bound) if op == "<=" else (val >= bound)
        lines.append(f"| {name} | {val:.4g} | {op} {bound:.4g} | {'PASS' if ok else 'FAIL'} |")
    lines += [
        "",
        f"Appearance cosine similarity: {result.appearance_similarity:.3f} "
        f"({o2.scene_region().appearance_class} vs {o4.scene_region().appearance_class})",
        "",
        f"**{'PASS' if passed else 'FAIL'}: proprioception "
        f"{'cannot' if passed else 'CAN'} disambiguate; vision "
        f"{'must' if passed else 'need not'} decide.**",
    ]
    report = "\n".join(lines) + "\n"
    out = Path(REPO_ROOT) / "outputs" / "map" / "ambiguity_match.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"[ambiguity_match] wrote {out}")
    print("PASS: ambiguity pair matched" if passed else "FAIL: ambiguity pair NOT matched")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
