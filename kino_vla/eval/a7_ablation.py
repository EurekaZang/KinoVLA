"""A7 method-ablation reducers (Paper-A §4 A7).

This module is intentionally torch-free: it reduces frozen-evaluation artifacts into the statistics
A7 reports. Real model training/evaluation lives in scripts; the helpers here make the result tables
unit-testable and prevent hand-edited numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CATEGORY_CUES: dict[str, tuple[str, ...]] = {
    "low_friction": ("slip", "sliding", "slide", "low traction", "friction", "ice"),
    "compliant_terrain": ("mud", "soft", "compliant", "sink", "sinking", "deform"),
    "region_collapse": ("collapse", "gave way", "support loss", "thin ice", "break"),
    "adhesion": ("sticky", "adhesive", "tether", "pull", "grip", "stuck"),
    "overload": ("payload", "heavy", "external weight", "load", "overload", "crouch"),
    "effort_decay": ("actuator", "motor", "effort", "torque", "weak", "decay", "crawl"),
    "external_push": ("push", "shove", "impulse", "external force"),
    "invisible_obstacle": ("obstacle", "barrier", "blocked", "rigid", "invisible"),
    "high_centering": ("beached", "ridge", "unloaded", "high-center", "high centering"),
    "obs_bias": ("sensor", "bias", "observation", "false"),
    "nominal": ("nominal", "no failure", "continue", "benign", "normal"),
}
CONTRADICTIONS: dict[str, tuple[str, ...]] = {
    "low_friction": ("sticky", "payload", "actuator weakened", "barrier"),
    "compliant_terrain": ("sticky", "adhesive", "actuator", "barrier"),
    "adhesion": ("mud", "soft terrain", "actuator", "payload"),
    "overload": ("actuator weak", "motor weak", "slip", "sticky"),
    "effort_decay": ("heavy external", "payload", "sticky", "mud"),
    "invisible_obstacle": ("slip", "mud", "sticky", "payload"),
    "nominal": ("failure", "hazard", "stuck", "fall", "slip", "sticky"),
}


@dataclass(frozen=True)
class RationaleGrounding:
    """Deterministic grounding verdict for one emitted rationale."""

    grounded: bool
    expected_cues: tuple[str, ...]
    matched_cues: tuple[str, ...]
    missing_cues: tuple[str, ...]
    contradiction: str | None = None


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with repo_path(path).open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"top level of {path} must be a mapping")
    return data


def load_json(path: str | Path) -> Any:
    return json.loads(repo_path(path).read_text())


def sha256_file(path: str | Path, *, short: int | None = 12) -> str:
    h = hashlib.sha256(repo_path(path).read_bytes()).hexdigest()
    return h[:short] if short else h


def sha256_dir(path: str | Path, *, short: int | None = 12) -> str:
    root = repo_path(path)
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(root)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    digest = h.hexdigest()
    return digest[:short] if short else digest


def config_hash(path: str | Path) -> str:
    return sha256_file(path, short=12)


def git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return res.stdout.strip()
    except Exception:  # pragma: no cover - defensive for exported artifacts
        return "unknown"


def artifact_meta(
    config_path: str | Path, *, sources: dict[str, str] | None = None
) -> dict[str, Any]:
    src_hashes: dict[str, str] = {}
    for name, path in (sources or {}).items():
        p = repo_path(path)
        if p.is_dir():
            src_hashes[name] = sha256_dir(p)
        elif p.exists():
            src_hashes[name] = sha256_file(p)
        else:
            src_hashes[name] = "missing"
    return {
        "experiment": "A7",
        "config_path": str(config_path),
        "config_sha256": config_hash(config_path),
        "commit": git_commit(),
        "source_hashes": src_hashes,
    }


def write_json(path: str | Path, payload: Any) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, center - half), 3), round(min(1.0, center + half), 3))


def _binom_two_sided_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    from math import comb

    probs = [comb(n, i) * 0.5**n for i in range(n + 1)]
    p_obs = probs[k]
    return float(min(1.0, sum(pi for pi in probs if pi <= p_obs + 1e-12)))


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict[str, Any]:
    if len(a_correct) != len(b_correct):
        raise ValueError("paired McNemar needs equal-length vectors")
    b = sum(1 for ai, bi in zip(a_correct, b_correct, strict=True) if ai and not bi)
    c = sum(1 for ai, bi in zip(a_correct, b_correct, strict=True) if bi and not ai)
    disc = b + c
    p = _binom_two_sided_p(min(b, c), disc)
    return {
        "n_pairs": len(a_correct),
        "b_a_right_b_wrong": b,
        "c_a_wrong_b_right": c,
        "discordant": disc,
        "p_exact_two_sided": round(p, 6),
        "b_better": c > b,
        "significant_05": p < 0.05,
    }


def auroc(score: list[float], label: list[int]) -> float:
    if len(score) != len(label):
        raise ValueError("score and label length mismatch")
    n1 = sum(1 for x in label if x == 1)
    n0 = sum(1 for x in label if x == 0)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = sorted(range(len(score)), key=lambda i: score[i])
    ranks = [0.0] * len(score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, y in zip(ranks, label, strict=True) if y == 1)
    return float((sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_mean_ci(
    values: list[float], *, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05
) -> list[float]:
    """Percentile bootstrap CI for a mean, deterministic for report artifacts."""
    xs = [float(x) for x in values]
    if not xs:
        return [0.0, 0.0]
    rng = random.Random(int(seed))
    means = []
    n = len(xs)
    for _ in range(int(n_boot)):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_i = max(0, min(len(means) - 1, int(math.floor((alpha / 2.0) * len(means)))))
    hi_i = max(0, min(len(means) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(means))) - 1))
    return [round(means[lo_i], 3), round(means[hi_i], 3)]


def risk_coverage_curve(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    cost_agent_field: str = "cost_agent",
    cost_safe_field: str = "cost_safe",
) -> list[dict[str, Any]]:
    """Expected cost as high-score rows abstain to a safe default.

    Lower scores are treated as safer/more confident. A row is covered (uses the agent decision) when
    ``score <= tau`` and abstains to ``cost_safe`` otherwise. The curve always includes coverage 0 and
    coverage 1 endpoints and is suitable for AURC integration.
    """
    if not rows:
        return []
    pts = [
        {
            "score": float(r[score_field]),
            "cost_agent": float(r[cost_agent_field]),
            "cost_safe": float(r[cost_safe_field]),
        }
        for r in rows
    ]
    thresholds = [min(p["score"] for p in pts) - 1.0e-12, *sorted({p["score"] for p in pts})]
    curve = []
    for tau in thresholds:
        covered = [p["score"] <= tau for p in pts]
        costs = [p["cost_agent"] if keep else p["cost_safe"] for p, keep in zip(pts, covered, strict=True)]
        curve.append(
            {
                "tau": float(tau),
                "coverage": round(sum(covered) / len(pts), 3),
                "expected_cost": round(sum(costs) / len(costs), 3),
                "n": len(pts),
            }
        )
    dedup: dict[float, dict[str, Any]] = {}
    for point in curve:
        cov = float(point["coverage"])
        if cov not in dedup or point["expected_cost"] < dedup[cov]["expected_cost"]:
            dedup[cov] = point
    return [dedup[cov] for cov in sorted(dedup)]


def risk_coverage_auc(curve: list[dict[str, Any]]) -> float:
    """Trapezoidal area under expected-cost vs coverage; lower is better."""
    if len(curve) < 2:
        return 0.0
    pts = sorted((float(p["coverage"]), float(p["expected_cost"])) for p in curve)
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return round(area, 3)


def _eval_threshold(rows: list[dict[str, Any]], *, score_field: str, tau: float) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "coverage": 0.0, "expected_cost": 0.0, "expected_cost_ci": [0.0, 0.0]}
    costs = []
    covered = 0
    for r in rows:
        keep = float(r[score_field]) <= float(tau)
        covered += int(keep)
        costs.append(float(r["cost_agent"] if keep else r["cost_safe"]))
    return {
        "n": len(rows),
        "coverage": round(covered / len(rows), 3),
        "expected_cost": round(sum(costs) / len(costs), 3),
        "expected_cost_ci": bootstrap_mean_ci(costs, seed=0),
    }


def heldout_threshold_selection(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    split_seed: int = 0,
    calibration_frac: float = 0.5,
) -> dict[str, Any]:
    """Select the risk-coverage threshold on calibration rows and evaluate it on held-out rows."""
    ordered = sorted(rows, key=lambda r: str(r.get("sample_id") or r.get("sid") or ""))
    rng = random.Random(int(split_seed))
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    n_cal = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(calibration_frac)))))
    cal = shuffled[:n_cal]
    test = shuffled[n_cal:]
    curve = risk_coverage_curve(cal, score_field=score_field)
    best = min(curve, key=lambda p: (float(p["expected_cost"]), -float(p["coverage"])))
    tau = float(best["tau"])
    return {
        "tau": tau,
        "calibration": best,
        "test": _eval_threshold(test, score_field=score_field, tau=tau),
        "splits": {
            "split_seed": int(split_seed),
            "calibration_frac": float(calibration_frac),
            "calibration_ids": [str(r.get("sample_id") or r.get("sid")) for r in cal],
            "test_ids": [str(r.get("sample_id") or r.get("sid")) for r in test],
        },
    }


def _normal_hoeffding_ucb(mean: float, n: int, alpha: float, cost_range: float) -> float:
    if n <= 0:
        return float("inf")
    return mean + float(cost_range) * math.sqrt(math.log(1.0 / max(alpha, 1.0e-12)) / (2.0 * n))


def conformal_risk_control(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    risk_budget: float,
    alpha: float = 0.1,
    cost_range: float = 6.0,
) -> dict[str, Any]:
    """Pick the highest-coverage threshold whose empirical cost UCB is below a risk budget."""
    candidates = []
    for point in risk_coverage_curve(rows, score_field=score_field):
        stats = _eval_threshold(rows, score_field=score_field, tau=float(point["tau"]))
        ucb = _normal_hoeffding_ucb(float(stats["expected_cost"]), int(stats["n"]), alpha, cost_range)
        candidates.append({**point, "ucb_mean_cost": round(ucb, 3)})
    feasible = [c for c in candidates if float(c["ucb_mean_cost"]) <= float(risk_budget)]
    selected = max(feasible, key=lambda c: (float(c["coverage"]), -float(c["expected_cost"]))) if feasible else min(candidates, key=lambda c: float(c["ucb_mean_cost"]))
    return {
        "risk_budget": float(risk_budget),
        "alpha": float(alpha),
        "cost_range": float(cost_range),
        "selected": selected,
        "n_candidates": len(candidates),
        "feasible": bool(feasible),
    }


def posterior_mean_variance(
    posteriors: list[dict[str, dict[str, float]]], sample_id: str
) -> float:
    """Mean per-category predictive variance across posterior-producing seeds."""
    rows = [p[sample_id] for p in posteriors if sample_id in p]
    cats = sorted({cat for row in rows for cat in row})
    if len(rows) < 2 or not cats:
        return 0.0
    variances = []
    for cat in cats:
        vals = [float(row.get(cat, 0.0)) for row in rows]
        mean = sum(vals) / len(vals)
        variances.append(sum((v - mean) ** 2 for v in vals) / len(vals))
    return round(sum(variances) / len(variances), 3)


def expected_calibration_error(
    rows: list[dict[str, Any]],
    *,
    confidence_field: str = "confidence",
    correct_field: str = "correct",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Reliability bins and ECE for posterior confidence vs correctness."""
    bins = []
    ece = 0.0
    n = len(rows)
    for i in range(int(n_bins)):
        lo = i / int(n_bins)
        hi = (i + 1) / int(n_bins)
        if i == int(n_bins) - 1:
            rs = [r for r in rows if lo <= float(r[confidence_field]) <= hi]
        else:
            rs = [r for r in rows if lo <= float(r[confidence_field]) < hi]
        if rs:
            conf = sum(float(r[confidence_field]) for r in rs) / len(rs)
            acc = sum(1 for r in rs if bool(r[correct_field])) / len(rs)
            ece += (len(rs) / max(1, n)) * abs(acc - conf)
        else:
            conf = 0.0
            acc = 0.0
        bins.append(
            {
                "bin": i,
                "lo": round(lo, 3),
                "hi": round(hi, 3),
                "n": len(rs),
                "confidence": round(conf, 3),
                "accuracy": round(acc, 3),
            }
        )
    return {"n": n, "n_bins": int(n_bins), "ece": round(ece, 3), "bins": bins}


def proportion_cell(rows: list[dict[str, Any]], field: str = "attr_ok") -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if bool(r.get(field)))
    return {"n": n, "k": k, "rate": round(k / max(1, n), 3), "ci": list(wilson(k, n))}


def summarize_by(rows: list[dict[str, Any]], key: str, field: str = "attr_ok") -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key, "?"))].append(r)
    for name, rs in sorted(groups.items()):
        out[name] = proportion_cell(rs, field)
    return out


def rationale_grounding(thought: str | None, truth_category: str) -> RationaleGrounding:
    text = (thought or "").lower()
    cues = CATEGORY_CUES.get(truth_category, ())
    matched = tuple(c for c in cues if c in text)
    contradictions = CONTRADICTIONS.get(truth_category, ())
    contradiction = next((c for c in contradictions if c in text), None)
    grounded = bool(matched) and contradiction is None
    return RationaleGrounding(
        grounded=grounded,
        expected_cues=cues,
        matched_cues=matched,
        missing_cues=tuple(c for c in cues if c not in matched),
        contradiction=contradiction,
    )


def grounding_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rationale grounding over rows with ``thought`` and ``truth``/``truth_category``."""
    out_rows = []
    for r in rows:
        truth = r.get("truth") or r.get("truth_category") or r.get("ground_truth")
        verdict = rationale_grounding(r.get("thought"), str(truth))
        attr_ok = bool(r.get("attr_ok"))
        out_rows.append({"grounded": verdict.grounded, "attr_ok": attr_ok})
    n = len(out_rows)
    grounded = sum(r["grounded"] for r in out_rows)
    correct = [r for r in out_rows if r["attr_ok"]]
    grounded_correct = sum(r["grounded"] for r in correct)
    overall_gc = sum(r["grounded"] and r["attr_ok"] for r in out_rows)
    return {
        "n": n,
        "grounding_rate": round(grounded / max(1, n), 3),
        "grounding_ci": list(wilson(grounded, n)),
        "n_correct": len(correct),
        "grounded_given_correct": round(grounded_correct / max(1, len(correct)), 3),
        "grounded_correct_rate": round(overall_gc / max(1, n), 3),
        "grounded_correct_ci": list(wilson(overall_gc, n)),
    }


def dose_curve_summary(
    dose_rows: dict[str, list[dict[str, Any]]], field: str = "attr_ok"
) -> dict[str, Any]:
    curve = []
    prev_rows: list[dict[str, Any]] | None = None
    prev_dose: str | None = None
    for dose in sorted(dose_rows, key=lambda x: int(x)):
        rows = dose_rows[dose]
        cell = proportion_cell(rows, field)
        rec = {"dose": int(dose), **cell}
        if prev_rows is not None:
            by_id_prev = {
                str(r.get("sample_id") or r.get("sid")): bool(r.get(field))
                for r in prev_rows
            }
            paired_prev, paired_cur = [], []
            for r in rows:
                sid = str(r.get("sample_id") or r.get("sid"))
                if sid in by_id_prev:
                    paired_prev.append(by_id_prev[sid])
                    paired_cur.append(bool(r.get(field)))
            rec["mcnemar_vs_prev"] = mcnemar(paired_prev, paired_cur) if paired_prev else None
            rec["prev_dose"] = int(prev_dose) if prev_dose is not None else None
        curve.append(rec)
        prev_rows = rows
        prev_dose = dose
    monotone = all(curve[i]["rate"] <= curve[i + 1]["rate"] + 1e-9 for i in range(len(curve) - 1))
    return {"curve": curve, "monotone_non_decreasing": monotone}


def status_counts(rows: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Count A7 row statuses in stable sorted-key order."""
    counts: Counter[str] = Counter()
    for row in rows.values():
        counts[str(row.get("status", "missing"))] += 1
    return dict(sorted(counts.items()))


def mean_rate_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Average already-reduced proportion cells across seeds without pretending raw trials pooled."""
    if not cells:
        return {"n_cells": 0, "mean_rate": 0.0, "min_rate": 0.0, "max_rate": 0.0}
    rates = [float(c.get("rate", c.get("acc", 0.0))) for c in cells]
    return {
        "n_cells": len(cells),
        "mean_rate": round(sum(rates) / len(rates), 3),
        "min_rate": round(min(rates), 3),
        "max_rate": round(max(rates), 3),
    }


def aggregate_dose_curves(curves: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed dose curves for report-level interpretation."""
    by_dose: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for curve in curves.values():
        for rec in curve.get("curve", []):
            by_dose[str(int(rec["dose"]))].append(rec)
    reduced = {dose: mean_rate_cells(cells) for dose, cells in sorted(by_dose.items(), key=lambda kv: int(kv[0]))}
    best_dose = None
    if reduced:
        best_dose = max(reduced.items(), key=lambda kv: (kv[1]["mean_rate"], -int(kv[0])))[0]
    return {
        "n_seeds": len(curves),
        "by_dose": reduced,
        "best_mean_dose": None if best_dose is None else int(best_dose),
        "monotone_all_seeds": bool(curves) and all(bool(c.get("monotone_non_decreasing")) for c in curves.values()),
    }


def encoder_cell_key(variant: str, window_length: int, gate: bool) -> str:
    """Stable key for one A7 encoder-grid cell."""
    return f"{variant}__T{int(window_length)}__gate_{'on' if gate else 'off'}"


def encoder_grid_keys(cfg: dict[str, Any]) -> list[str]:
    """Return the full declared variant × window × gate grid in config order."""
    variants = list((cfg.get("variants") or {}).keys())
    windows = [int(x) for x in cfg.get("window_lengths", [])]
    gates = [bool(x) for x in cfg.get("anomaly_gate", [])]
    return [encoder_cell_key(v, w, g) for v in variants for w in windows for g in gates]


def tail_window_rows(rows: list[list[float]] | tuple[Any, ...], length: int) -> list[list[float]]:
    """Take the most recent ``length`` observable rows, failing closed if the source is shorter."""
    seq = [list(r) for r in rows]
    n = int(length)
    if n <= 0:
        raise ValueError("window length must be positive")
    if len(seq) < n:
        raise ValueError(f"cannot build T={n} from only {len(seq)} observed rows")
    return seq[-n:]


def _best_available_encoder(cells: dict[str, dict[str, Any]], metric: str, *, higher: bool) -> dict[str, Any] | None:
    vals = []
    for key, cell in cells.items():
        if cell.get("status") != "available" or metric not in cell:
            continue
        value = cell.get(metric)
        if isinstance(value, dict):
            value = value.get("rate", value.get("mean_rate", value.get("value")))
        if value is None:
            continue
        vals.append((key, float(value), cell))
    if not vals:
        return None
    key, value, cell = (max if higher else min)(vals, key=lambda kv: (kv[1], kv[0]))
    return {
        "key": key,
        "value": round(value, 4),
        "variant": cell.get("variant"),
        "window_length": cell.get("window_length"),
        "gate": cell.get("gate"),
    }


def aggregate_encoder_grid(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize a full A7 encoder grid while preserving blocked/missing real cells."""
    counts = status_counts(cells)
    available = sum(1 for c in cells.values() if c.get("status") == "available")
    return {
        "status_counts": counts,
        "available_cells": available,
        "blocked_cells": len(cells) - available,
        "best_theta_mae": _best_available_encoder(cells, "theta_mae_mean", higher=False),
        "best_attr_acc": _best_available_encoder(cells, "attr_acc", higher=True),
        "best_residual_error_auroc": _best_available_encoder(cells, "residual_error_auroc", higher=True),
    }



def ers_regret_from_agent_rows(
    agent_rows: dict[str, list[dict[str, Any]]],
    m_cost: dict[str, dict[str, float]],
    canonical_label: dict[str, str],
) -> dict[str, Any]:
    """Project agent primitive choices through the A4 cost matrix.

    Each row must provide a scenario key and a primitive/label. Rows with scenarios or labels not in
    the matrix are counted as missing and excluded from means; this prevents accidental fabrication.
    """
    out: dict[str, Any] = {}
    for agent, rows in sorted(agent_rows.items()):
        costs: list[float] = []
        regrets: list[float] = []
        missing = 0
        for row in rows:
            scenario = str(row.get("scenario") or row.get("a4_scenario") or row.get("scenario_id"))
            label = str(row.get("primitive") or row.get("label") or row.get("pred_label"))
            scenario_costs = m_cost.get(scenario)
            canonical = canonical_label.get(scenario)
            if scenario_costs is None or canonical is None or label not in scenario_costs or canonical not in scenario_costs:
                missing += 1
                continue
            cost = float(scenario_costs[label])
            best_cost = float(scenario_costs[canonical])
            costs.append(cost)
            regrets.append(cost - best_cost)
        out[agent] = {
            "n": len(rows),
            "n_scored": len(costs),
            "n_missing": missing,
            "mean_cost": round(sum(costs) / max(1, len(costs)), 3),
            "mean_regret": round(sum(regrets) / max(1, len(regrets)), 3),
        }
    return out


def a2_per_item_rows(per_item: dict[str, list[dict[str, Any]]], agent: str) -> list[dict[str, Any]]:
    rows = per_item.get(agent)
    if rows is None:
        raise KeyError(f"agent {agent!r} not found in A2 per_item")
    return rows


def primitive_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r.get("primitive")) for r in rows))


def write_blocked_artifact(
    path: str | Path,
    *,
    config_path: str | Path,
    reason: str,
    required: str,
    sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = {
        **artifact_meta(config_path, sources=sources),
        "status": "blocked",
        "reason": reason,
        "required_for_publication": required,
        "finding": False,
    }
    write_json(path, payload)
    return payload
