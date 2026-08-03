#!/usr/bin/env python
"""A6 — boundary audit: base-policy bracket, learned decisions, and projector residual.

This reducer is intentionally Isaac-free: it evaluates frozen snapshots and imports the already
measured A6.1 closed-loop sweep.  It fixes two historical leakage/estimand bugs:

* A6.3 now selects only the 40 T5/O2 nominal snapshots (not all 160 O2 snapshots);
* attribution-level ``nominal`` and an actually non-intervening primitive are reported separately.

The five-point base sweep has complete separation (6/6 reaches at floors .40/.30; 0/6 below), so
the primary physical boundary is the identified bracket ``(.25, .30)``.  A logistic midpoint is
retained only as a descriptive interpolation, not a four-decimal ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

INTERVENTION_PRIMS = {
    "Switch_Gait",
    "Set_Constraint",
    "Hold_and_Request",
    "Backstep",
    "Update_Topology",
    "Replan_Waypoint",
    "Adjust_Posture",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _decision_row(decision, *, sample_id: str, floor: float | None = None) -> dict:
    parsed = bool(decision.ok and decision.annotation is not None)
    attribution = decision.attribution if parsed else None
    primitive = decision.primitive_name if parsed else None
    params = dict(decision.annotation.primitive.params) if parsed else {}
    action_intervene = bool(parsed and primitive in INTERVENTION_PRIMS)
    attribution_abstain = bool(parsed and attribution == "nominal")
    return {
        "sample_id": sample_id,
        "floor": floor,
        "parsed": parsed,
        "attribution": attribution,
        "primitive": primitive,
        "primitive_params": params,
        "attribution_abstain": attribution_abstain,
        "action_intervene": action_intervene,
        "action_nonintervene": bool(parsed and not action_intervene),
        "consistent_abstention": bool(attribution_abstain and not action_intervene),
    }


def _aggregate_decisions(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[float(row[key])].append(row)
    out = []
    for value in sorted(grouped, reverse=True):
        items = grouped[value]
        n = len(items)
        out.append(
            {
                key: value,
                "n": n,
                "parse_rate": round(sum(r["parsed"] for r in items) / n, 4),
                "p_attribution_abstain": round(sum(r["attribution_abstain"] for r in items) / n, 4),
                "p_action_intervene": round(sum(r["action_intervene"] for r in items) / n, 4),
                "p_action_nonintervene": round(sum(r["action_nonintervene"] for r in items) / n, 4),
                "attributions": sorted({str(r["attribution"]) for r in items}),
                "primitives": sorted({str(r["primitive"]) for r in items}),
            }
        )
    return out


def _load_snapshots(directory: Path) -> list[tuple[dict, object]]:
    from kino_vla.vla.dataset_build import _snapshot_from_record

    samples_path = directory / "samples.jsonl"
    frames_path = directory / "frames.npz"
    if not samples_path.exists() or not frames_path.exists():
        return []
    records = [json.loads(line) for line in samples_path.read_text().splitlines() if line]
    frames_npz = np.load(frames_path)
    out = []
    for record in records:
        sid = record["sample_id"]
        frames = {key: frames_npz[f"{sid}__{key}"] for key in ("rgb", "depth", "proprio")}
        out.append((record, _snapshot_from_record(record, frames)))
    return out


def _residual_analysis(
    corpus: Path, adapter: Path, base_sweep: dict
) -> tuple[dict, dict[str, float]]:
    from scripts.a5_5_risk_coverage import compute_residuals

    residuals = compute_residuals(str(adapter), [str(corpus)])
    by_floor: dict[float, list[float]] = defaultdict(list)
    records = [json.loads(line) for line in (corpus / "samples.jsonl").read_text().splitlines()]
    for record in records:
        by_floor[float(record["a6_floor"])].append(residuals[record["sample_id"]])
    rows = [
        {
            "floor": floor,
            "n": len(values),
            "mean_residual": round(float(np.mean(values)), 6),
            "std_residual": round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0,
            "n_unique": len(set(values)),
        }
        for floor, values in sorted(by_floor.items(), reverse=True)
    ]
    reach_by_floor = {float(r["floor"]): float(r["base_reach"]) for r in base_sweep["rows"]}
    floors = np.asarray([r["floor"] for r in rows], dtype=float)
    scores = np.asarray([r["mean_residual"] for r in rows], dtype=float)
    failed = np.asarray([reach_by_floor[f] < 0.5 for f in floors], dtype=bool)

    from scipy.stats import kendalltau, spearmanr

    spearman = spearmanr(floors, scores)
    kendall = kendalltau(floors, scores)
    pair_scores = [
        float(scores[i] > scores[j]) + 0.5 * float(scores[i] == scores[j])
        for i in np.where(failed)[0]
        for j in np.where(~failed)[0]
    ]
    auroc = float(np.mean(pair_scores))
    # Exact one-sided label-permutation p-value at the independent floor-group level (n=5).
    n_failed = int(failed.sum())
    permutation_aurocs = []
    for positive_indices in itertools.combinations(range(len(rows)), n_failed):
        perm_failed = np.zeros(len(rows), dtype=bool)
        perm_failed[list(positive_indices)] = True
        comparisons = [
            float(scores[i] > scores[j]) + 0.5 * float(scores[i] == scores[j])
            for i in np.where(perm_failed)[0]
            for j in np.where(~perm_failed)[0]
        ]
        permutation_aurocs.append(float(np.mean(comparisons)))
    permutation_p = sum(value >= auroc - 1e-12 for value in permutation_aurocs) / len(
        permutation_aurocs
    )
    threshold = (float(scores[failed].min()) + float(scores[~failed].max())) / 2
    monotonic_violations = sum(
        scores[i] > scores[i + 1] for i in range(len(scores) - 1)
    )  # floors sorted mild→severe; residual should rise
    return (
        {
            "unit_of_analysis": (
                "five independent floor settings; six seeds are duplicates for the frozen snapshot"
            ),
            "rows": rows,
            "spearman_floor_vs_residual": {
                "rho": round(float(spearman.statistic), 4),
                "p_two_sided_exploratory": round(float(spearman.pvalue), 4),
            },
            "kendall_floor_vs_residual": {
                "tau": round(float(kendall.statistic), 4),
                "p_two_sided_exploratory": round(float(kendall.pvalue), 4),
            },
            "failure_auroc_floor_group": round(auroc, 4),
            "failure_auroc_exact_permutation_p_one_sided": round(permutation_p, 4),
            "descriptive_separating_threshold": round(threshold, 6),
            "monotonic_violations": int(monotonic_violations),
            "inference": (
                "perfect descriptive separation, but only five floor groups "
                "(exact p=0.1); boundary evidence is suggestive, not confirmatory"
            ),
        },
        residuals,
    )


def _base_boundary(base: dict) -> dict:
    successful = [float(row["floor"]) for row in base["rows"] if row["base_reach"] >= 0.5]
    failed = [float(row["floor"]) for row in base["rows"] if row["base_reach"] < 0.5]
    lower = max(failed)
    upper = min(successful)
    rows = []
    for row in base["rows"]:
        k = round(float(row["base_reach"]) * int(row["n"]))
        rows.append({**row, "success_ci_wilson95": wilson(k, int(row["n"]))})
    return {
        "primary_identified_bracket": [lower, upper],
        "bracket_width": round(upper - lower, 4),
        "interpretation": f"base-policy transition is identified only within ({lower}, {upper})",
        "logistic_midpoint_descriptive": base.get("theta_star"),
        "logistic_fit_warning": (
            "complete separation on a five-point grid makes slope/midpoint "
            "precision model-dependent"
        ),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="A6 frozen boundary audit")
    ap.add_argument("--o10-sweep", default="outputs/eval/a6/corpus_o10_sweep")
    ap.add_argument("--base-sweep", default="outputs/eval/a6/a6_1_theta_sweep.json")
    ap.add_argument(
        "--adapters",
        default=(
            "B5-unshaped=outputs/vla/sft_latent/adapter_best,"
            "B5-conflict-bi=outputs/eval/a3/b5_conflict_bi/adapter_best,"
            "zero-shot=none"
        ),
    )
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--a3-dir", default="outputs/eval/a3")
    ap.add_argument("--residual-adapter", default="outputs/eval/a3/b5_conflict_bi/adapter_best")
    ap.add_argument("--out", default="outputs/eval/a6/a6_eval.json")
    args = ap.parse_args()

    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.utils.config import load_config
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy
    from scripts.eval_a3 import vla_input_key

    o10_dir = Path(args.o10_sweep)
    base_path = Path(args.base_sweep)
    base = json.loads(base_path.read_text())
    snapshots = _load_snapshots(o10_dir)
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    vcfg = load_config(args.config)
    n_images = int(vcfg.data.get("n_images", 1))
    proprio_detail = str(vcfg.data.get("proprio_detail", "binned"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    roster = [
        (item.split("=", 1)[0], None if item.split("=", 1)[1] == "none" else item.split("=", 1)[1])
        for item in args.adapters.split(",")
    ]

    decision_by_agent = {}
    input_hashes = {
        "base_sweep": _sha256(base_path),
        "o10_samples": _sha256(o10_dir / "samples.jsonl"),
        "o10_frames": _sha256(o10_dir / "frames.npz"),
    }
    for name, adapter in roster:
        model = KinoVLA.from_pretrained(vcfg, device=device, adapter_dir=adapter)
        model.eval()
        policy = ModelVlaPolicy(
            model,
            pcfg,
            tax,
            route="latent",
            n_images=n_images,
            temperature=0.0,
            proprio_detail=proprio_detail,
        )
        cache = {}
        rows = []
        for record, snapshot in snapshots:
            key = vla_input_key(
                snapshot,
                cfg=pcfg,
                route="latent",
                n_images=n_images,
                proprio_detail=proprio_detail,
                mask=False,
            )
            if key not in cache:
                cache[key] = policy.decide(snapshot)
            rows.append(
                _decision_row(
                    cache[key], sample_id=record["sample_id"], floor=float(record["a6_floor"])
                )
            )
        decision_by_agent[name] = {
            "n_rows": len(rows),
            "n_unique_model_inputs": len(cache),
            "by_floor": _aggregate_decisions(rows, "floor"),
            "per_item": rows,
        }
        if adapter is not None:
            input_hashes[f"adapter_{name}"] = _sha256(Path(adapter) / "adapter_model.safetensors")
        del policy, model
        torch.cuda.empty_cache()
        print(f"[a6] {name}: {len(cache)} unique inputs / {len(rows)} rows", flush=True)

    # Strict O2_A_nominal panel from the newly version-consistent A3 files.
    file_stems = {
        "B5-unshaped": "B5_unshaped",
        "B5-conflict-bi": "B5_conflict_bi",
        "zero-shot": "zero_shot",
    }
    o2_panel = {}
    for name, _adapter in roster:
        path = Path(args.a3_dir) / f"per_item_{file_stems[name]}.json"
        input_hashes[f"a3_per_item_{name}"] = _sha256(path)
        all_rows = json.loads(path.read_text())
        selected = [
            row
            for row in all_rows
            if row.get("cell") == "T5" and row.get("operator") == "O2_compliance"
        ]
        if len(selected) != 40:
            raise ValueError(f"expected 40 O2_A_nominal rows for {name}, got {len(selected)}")
        rows = [
            {
                "parsed": bool(row.get("parsed")),
                "attribution_abstain": row.get("attribution") == "nominal",
                "action_intervene": row.get("primitive") in INTERVENTION_PRIMS,
            }
            for row in selected
        ]
        n = len(rows)
        o2_panel[name] = {
            "n": n,
            "p_attribution_abstain": round(sum(r["attribution_abstain"] for r in rows) / n, 4),
            "p_action_intervene": round(sum(r["action_intervene"] for r in rows) / n, 4),
            "p_action_nonintervene": round(sum(not r["action_intervene"] for r in rows) / n, 4),
        }

    residual, _residuals = _residual_analysis(o10_dir, Path(args.residual_adapter), base)
    input_hashes["residual_projector"] = _sha256(Path(args.residual_adapter) / "kino_projector.pt")
    result = {
        "schema_version": 2,
        "input_sha256": input_hashes,
        "base_boundary": _base_boundary(base),
        "learned_decisions": {
            "by_agent": decision_by_agent,
            "estimand": "attribution abstention and actual intervention are separate",
        },
        "o2_a_nominal": {
            "by_agent": o2_panel,
            "selection": "cell == T5 and operator == O2_compliance (exactly 40 snapshots)",
        },
        "projector_residual": residual,
        "conclusion": (
            "base boundary is bracketed, projector residual is suggestive at n=5 "
            "floor groups, and no evaluated learned agent exhibits a calibrated "
            "intervention flip"
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    _render(out_path.parent, result)
    print(f"[OK] wrote {out_path} + boundary_audit.png")


def _render(out: Path, result: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    base_rows = result["base_boundary"]["rows"]
    residual_rows = result["projector_residual"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    floors = [row["floor"] for row in base_rows]
    ax.plot(floors, [row["base_reach"] for row in base_rows], "o-", label="base reach")
    ax.axvspan(0.25, 0.30, alpha=0.2, color="#d62728", label="identified bracket")
    ax.set_xlabel("effort floor (mild → severe leftward)")
    ax.set_ylabel("base reach rate")
    ax.set_title("A6 physical boundary")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    rfloors = [row["floor"] for row in residual_rows]
    ax.plot(rfloors, [row["mean_residual"] for row in residual_rows], "o-", label="θ residual")
    for name, payload in result["learned_decisions"]["by_agent"].items():
        ax.plot(
            [row["floor"] for row in payload["by_floor"]],
            [row["p_action_intervene"] for row in payload["by_floor"]],
            "--",
            label=f"{name} P(intervene)",
        )
    ax.set_xlabel("effort floor")
    ax.set_ylabel("residual / intervention probability")
    ax.set_title("Residual tracks severity; agents do not flip")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "boundary_audit.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
