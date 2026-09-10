#!/usr/bin/env python3
"""Plot the definitive 11-operator action-consequence evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TEAL = "#2C9CA0"
BLUE = "#2665EA"
GRAY = "#999999"
BLACK = "#000000"
PALE_TEAL = "#D7EEEE"

OPERATOR_LABELS = {
    "O1_mu_field": "O1 low friction",
    "O2_compliance": "O2 compliance",
    "O3_collapse": "O3 collapse",
    "O4_tether": "O4 adhesion",
    "O5_payload": "O5 overload",
    "O6_push": "O6 external push",
    "O7_visual_remap": "O7 remapped friction",
    "O8_invisible_collider": "O8 invisible obstacle",
    "O9_high_centering": "O9 high centering",
    "O10_effort_decay": "O10 effort decay",
    "O11_obs_bias": "O11 observation bias",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def operator_number(name: str) -> int:
    return int(name.split("_", 1)[0].removeprefix("O"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = load(args.analysis.resolve())
    if analysis.get("publication_evidence_eligible") is not True:
        raise RuntimeError("refusing to plot an ineligible analysis")
    rows = sorted(analysis["operator_results"], key=lambda row: operator_number(row["operator"]))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.15, 3.05), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=(2.35, 1.0), wspace=0.32)
    ax = fig.add_subplot(grid[0, 0])
    ax_summary = fig.add_subplot(grid[0, 1])

    labels = [OPERATOR_LABELS[row["operator"]] for row in rows]
    benefit = np.asarray(
        [-float(row["registered_minus_mean_mismatched_cost"]) for row in rows]
    )
    ci_low = np.asarray([-float(row["primary_scene_cluster_ci_95"][1]) for row in rows])
    ci_high = np.asarray([-float(row["primary_scene_cluster_ci_95"][0]) for row in rows])
    y = np.arange(len(rows))[::-1]
    for index, row in enumerate(rows):
        color = BLUE if row["operator"] == "O6_push" else TEAL
        marker = "D" if row["operator"] == "O6_push" else "o"
        yy = y[index]
        ax.plot(
            [ci_low[index], ci_high[index]],
            [yy, yy],
            color=color,
            lw=2.0,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            [ci_low[index], ci_low[index]],
            [yy - 0.10, yy + 0.10],
            color=color,
            lw=1.0,
            zorder=2,
        )
        ax.plot(
            [ci_high[index], ci_high[index]],
            [yy - 0.10, yy + 0.10],
            color=color,
            lw=1.0,
            zorder=2,
        )
        ax.scatter(
            benefit[index], yy, s=29, marker=marker, color=color,
            edgecolor="white", linewidth=0.55, zorder=3,
        )
        ax.text(
            ci_high[index] + 1.1,
            yy,
            f"{benefit[index]:.1f}",
            ha="left",
            va="center",
            fontsize=7.4,
            color=BLACK,
        )
    ax.axvline(0.0, color=BLACK, lw=0.8, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlabel("Terminal-cost reduction vs. mismatched recovery (higher is better)")
    ax.set_title("a  Cause-matched recovery benefits every operator", loc="left", fontweight="bold", pad=7)
    ax.grid(axis="x", color="#E6E6E6", lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-2, max(ci_high) + 12)
    ax.text(
        0.995,
        0.015,
        "Bars: scene-cluster 95% CI\nBlue diamond: independent O6 confirmation",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )

    overall = analysis["overall_vs_continue"]
    groups = ["Recovery\nsuccess", "Fall rate"]
    comparator = np.asarray(
        [100.0 * overall["continue_success_rate"], 100.0 * overall["continue_fall_rate"]]
    )
    registered = np.asarray(
        [100.0 * overall["registered_success_rate"], 100.0 * overall["registered_fall_rate"]]
    )
    x = np.arange(2)
    width = 0.31
    ax_summary.bar(
        x - width / 2,
        comparator,
        width=width,
        color=GRAY,
        label="Continue",
        zorder=3,
    )
    ax_summary.bar(
        x + width / 2,
        registered,
        width=width,
        color=TEAL,
        label="Cause-matched",
        zorder=3,
    )
    for xpos, baseline, method in zip(x, comparator, registered, strict=True):
        top = max(baseline, method)
        ax_summary.text(
            xpos - width / 2,
            baseline + 2.0,
            f"{baseline:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#555555",
        )
        ax_summary.text(
            xpos + width / 2,
            method + 2.0,
            f"{method:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=TEAL,
            fontweight="bold",
        )
        ax_summary.plot(
            [xpos - width * 0.78, xpos + width * 0.78],
            [baseline, baseline],
            color=BLACK,
            lw=0.65,
            ls="--",
            zorder=4,
        )
        ax_summary.annotate(
            "",
            xy=(xpos + width / 2, method),
            xytext=(xpos + width / 2, baseline),
            arrowprops={"arrowstyle": "->", "color": BLACK, "lw": 0.75},
            zorder=5,
        )
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(groups, fontsize=7.8)
    ax_summary.set_ylabel("Physical cases (%)")
    ax_summary.set_ylim(0, 108)
    ax_summary.set_title("b  Aggregate outcomes", loc="left", fontweight="bold", pad=7)
    ax_summary.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0)
    ax_summary.spines["top"].set_visible(False)
    ax_summary.spines["right"].set_visible(False)
    ax_summary.legend(frameon=False, fontsize=7.1, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

    fig.text(
        0.01,
        0.995,
        "KINO-FAIL: attribution changes physical recovery consequences",
        ha="left",
        va="top",
        fontsize=10.3,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.88, bottom=0.23)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    png = output / "kinofail_action_consequence_definitive_v1.png"
    pdf = output / "kinofail_action_consequence_definitive_v1.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
