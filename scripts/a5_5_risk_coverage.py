#!/usr/bin/env python
"""A5.5 — leakage-resistant selective prediction in A4 physical-cost units.

The previous reducer selected and evaluated a threshold on the same snapshots and allowed the
coverage-zero endpoint to serve as the headline optimum.  This implementation uses the frozen
appearance split (train appearances calibrate; test appearances evaluate), reports non-zero target
coverage operating points, and obtains paired two-stage bootstrap intervals by resampling test
appearance clusters and A4 matrix episodes.

Two protocols are deliberately separate:

* ``broad_overlap``: every A3 scenario that overlaps the measured A4 matrix;
* ``t2_pair``: the adhesion/mud ambiguity where Backstep is the justified conservative fallback.

The T2 projector residual is expected to be constant because matched O4/O2 proprioception is
byte-identical by construction.  A constant score is reported as non-identifiable, never promoted
to a selective-prediction gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from kino_vla.eval.a4_consequence import (
    Outcome,
    a3_row_to_a4_scenario,
    physical_cost,
    primitive_to_label,
)

SAFE_DEFAULT = "backstep_detour"
T2_SCENARIOS = {"matched_O4_twophase", "matched_O2"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_residuals(adapter: str, corpus_dirs: list[str]) -> dict[str, float]:
    """Compute the deployed Kino-Projector θ residual for each frozen sample."""
    from kino_vla.vla.projector import KinoProjector

    pj = KinoProjector(
        vlm_dim=2560,
        conv_channels=(48, 48),
        conv_kernels=(5, 3),
        latent_dim=96,
        n_latents=6,
        n_heads=4,
        mlp_hidden=96,
        proj_hidden=512,
    )
    pj.load_state_dict(torch.load(Path(adapter, "kino_projector.pt"), map_location="cpu"))
    pj.eval()
    sid2res: dict[str, float] = {}
    with torch.no_grad():
        for directory in corpus_dirs:
            sp = Path(directory, "samples.jsonl")
            npz_path = Path(directory, "frames.npz")
            if not sp.exists() or not npz_path.exists():
                continue
            npz = np.load(npz_path)
            for line in sp.read_text().splitlines():
                if not line:
                    continue
                record = json.loads(line)
                sid = record["sample_id"]
                target = record.get("target_theta")
                if f"{sid}__proprio" not in npz or target is None or len(target) != 4:
                    continue
                window = torch.from_numpy(npz[f"{sid}__proprio"].astype(np.float32)).unsqueeze(0)
                _soft, theta = pj(window)
                prediction = theta.squeeze(0).float().cpu().numpy()
                sid2res[sid] = float(
                    np.linalg.norm(prediction - np.asarray(target, dtype=np.float32))
                )
    return sid2res


def _actual_label(row: dict) -> str:
    primitive = row.get("primitive")
    if not primitive:
        return "continue"
    if primitive == "Switch_Gait" and not row.get("action_params_observed", False):
        raise ValueError(
            f"legacy row {row.get('sid')} lost Switch_Gait mode; rerun scripts.eval_a3 first"
        )
    return primitive_to_label(primitive, row.get("primitive_params") or {}, strict_params=True)


def build_points(
    rows: list[dict], residuals: dict[str, float], matrix: dict[str, dict]
) -> list[dict]:
    """Join A3 decisions, projector residuals, frozen split metadata, and A4 labels."""
    points: list[dict] = []
    for row in rows:
        scenario = a3_row_to_a4_scenario(row)
        sid = str(row.get("sid", ""))
        if scenario not in matrix or sid not in residuals or not row.get("parsed"):
            continue
        split = str(row.get("appearance_split", ""))
        appearance = str(row.get("appearance_id", ""))
        if split not in {"train", "test"} or not appearance:
            raise ValueError(f"row {sid} lacks the frozen appearance split metadata")
        agent_label = _actual_label(row)
        points.append(
            {
                "sample_id": sid,
                "scenario": scenario,
                "appearance_id": appearance,
                "appearance_split": split,
                "cluster_id": f"{scenario}|{appearance}",
                "score": float(residuals[sid]),
                "agent_label": agent_label,
                "safe_label": SAFE_DEFAULT,
                "continue_label": "continue",
                "attr_ok": bool(row.get("attr_ok")),
            }
        )
    return points


def _mean_cost(
    rows: list[dict], policy: str, *, tau: float | None, cost_map: dict[tuple[str, str], float]
) -> tuple[float, float]:
    costs: list[float] = []
    covered = 0
    for row in rows:
        if policy == "selective":
            assert tau is not None
            act = float(row["score"]) <= tau
            label = row["agent_label"] if act else row["safe_label"]
            covered += int(act)
        elif policy == "agent":
            label = row["agent_label"]
            covered += 1
        elif policy == "safe":
            label = row["safe_label"]
        elif policy == "continue":
            label = row["continue_label"]
        else:
            raise ValueError(policy)
        costs.append(float(cost_map[(row["scenario"], label)]))
    return float(np.mean(costs)), covered / len(rows)


def risk_coverage_curve(rows: list[dict], cost_map: dict[tuple[str, str], float]) -> list[dict]:
    scores = sorted({float(row["score"]) for row in rows})
    if not scores:
        return []
    thresholds = [scores[0] - 1e-12, *scores]
    curve = []
    for tau in thresholds:
        cost, coverage = _mean_cost(rows, "selective", tau=tau, cost_map=cost_map)
        curve.append(
            {
                "tau": float(tau),
                "coverage": round(coverage, 4),
                "expected_cost": round(cost, 4),
            }
        )
    return curve


def _choose_best(curve: list[dict], *, require_nonzero: bool = False) -> dict | None:
    candidates = [p for p in curve if not require_nonzero or float(p["coverage"]) > 0.0]
    if not candidates:
        return None
    return min(candidates, key=lambda p: (float(p["expected_cost"]), -float(p["coverage"])))


def _choose_target(curve: list[dict], target: float) -> dict:
    return min(
        curve,
        key=lambda p: (
            abs(float(p["coverage"]) - target),
            float(p["expected_cost"]),
            -float(p["coverage"]),
        ),
    )


def _matrix_costs(a4: dict) -> dict[tuple[str, str], float]:
    return {
        (scenario, label): float(cell["mean_cost"])
        for scenario, cells in a4["M_full"].items()
        for label, cell in cells.items()
        if int(cell.get("n", 0)) > 0
    }


def _matrix_episode_costs(path: Path) -> dict[tuple[str, str], list[float]]:
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line:
            continue
        outcome = Outcome(**json.loads(line))
        cells[(outcome.scenario, outcome.label)].append(float(physical_cost(outcome)))
    return dict(cells)


def _resample_test_clusters(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """Stratified appearance-cluster bootstrap preserving every test scenario."""
    by_scenario: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_scenario[row["scenario"]][row["cluster_id"]].append(row)
    sampled: list[dict] = []
    for clusters in by_scenario.values():
        keys = list(clusters)
        for index in rng.integers(0, len(keys), size=len(keys)):
            sampled.extend(clusters[keys[int(index)]])
    return sampled


def _bootstrap(
    rows: list[dict],
    tau: float,
    episode_costs: dict[tuple[str, str], list[float]],
    *,
    reps: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    names = ("selective", "agent", "safe", "continue")
    draws = {name: [] for name in names}
    coverages: list[float] = []
    for _ in range(reps):
        sampled_rows = _resample_test_clusters(rows, rng)
        sampled_costs = {
            key: float(np.mean(rng.choice(values, size=len(values), replace=True)))
            for key, values in episode_costs.items()
        }
        for name in names:
            cost, coverage = _mean_cost(
                sampled_rows,
                name,
                tau=tau if name == "selective" else None,
                cost_map=sampled_costs,
            )
            draws[name].append(cost)
            if name == "selective":
                coverages.append(coverage)

    def ci(values: list[float]) -> list[float]:
        return [
            round(float(np.quantile(values, 0.025)), 4),
            round(float(np.quantile(values, 0.975)), 4),
        ]

    deltas = {
        f"selective_minus_{name}": [
            a - b for a, b in zip(draws["selective"], draws[name], strict=True)
        ]
        for name in ("agent", "safe", "continue")
    }
    return {
        "reps": reps,
        "unit": "appearance cluster within scenario + A4 episode within (scenario,label)",
        "cost_ci": {name: ci(values) for name, values in draws.items()},
        "coverage_ci": ci(coverages),
        "paired_delta_ci": {name: ci(values) for name, values in deltas.items()},
    }


def evaluate_protocol(
    name: str,
    points: list[dict],
    cost_map: dict[tuple[str, str], float],
    episode_costs: dict[tuple[str, str], list[float]],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict:
    calibration = [p for p in points if p["appearance_split"] == "train"]
    test = [p for p in points if p["appearance_split"] == "test"]
    if not calibration or not test:
        raise ValueError(f"protocol {name} needs non-empty frozen train/test appearance splits")
    cal_curve = risk_coverage_curve(calibration, cost_map)
    test_curve = risk_coverage_curve(test, cost_map)  # descriptive oracle curve, not for selection
    selected_cal = _choose_best(cal_curve)
    selected_nonzero_cal = _choose_best(cal_curve, require_nonzero=True)
    assert selected_cal is not None

    def deploy(cal_point: dict | None, tag: str, seed_offset: int) -> dict | None:
        if cal_point is None:
            return None
        tau = float(cal_point["tau"])
        selective_cost, coverage = _mean_cost(test, "selective", tau=tau, cost_map=cost_map)
        baselines = {}
        for baseline in ("agent", "safe", "continue"):
            cost, _ = _mean_cost(test, baseline, tau=None, cost_map=cost_map)
            baselines[baseline] = round(cost, 4)
        boot = _bootstrap(
            test,
            tau,
            episode_costs,
            reps=bootstrap_reps,
            seed=bootstrap_seed + seed_offset,
        )
        deltas = {key: round(selective_cost - value, 4) for key, value in baselines.items()}
        return {
            "selection": tag,
            "tau_from_calibration": tau,
            "calibration": cal_point,
            "test": {
                "n": len(test),
                "coverage": round(coverage, 4),
                "expected_cost": round(selective_cost, 4),
                "baseline_cost": baselines,
                "paired_delta": deltas,
            },
            "bootstrap": boot,
            "strictly_better_with_95ci": {
                key.removeprefix("selective_minus_"): ci[1] < 0.0
                for key, ci in boot["paired_delta_ci"].items()
            },
        }

    targets = {}
    for index, target in enumerate((0.25, 0.5, 0.75), start=10):
        point = _choose_target(cal_curve, target)
        targets[str(target)] = deploy(point, f"calibration_target_{target}", index)

    unique_cal = len({float(p["score"]) for p in calibration})
    unique_test = len({float(p["score"]) for p in test})
    return {
        "protocol": name,
        "scenarios": sorted({p["scenario"] for p in points}),
        "n": {"calibration": len(calibration), "test": len(test)},
        "n_appearance_clusters": {
            "calibration": len({p["cluster_id"] for p in calibration}),
            "test": len({p["cluster_id"] for p in test}),
        },
        "score_support": {
            "n_unique_calibration": unique_cal,
            "n_unique_test": unique_test,
            "selective_identifiable": unique_cal > 1 and unique_test > 1,
            "reason_if_not": (
                None
                if unique_cal > 1 and unique_test > 1
                else "projector residual is constant on this protocol; only endpoint policies exist"
            ),
        },
        "selected_min_cost": deploy(selected_cal, "minimum_calibration_cost", 0),
        "selected_min_cost_nonzero": deploy(
            selected_nonzero_cal, "minimum_calibration_cost_subject_to_nonzero_coverage", 1
        ),
        "target_coverage": targets,
        "calibration_curve": cal_curve,
        "test_oracle_curve": test_curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="A5.5 held-out risk-coverage analysis")
    ap.add_argument("--agent-file", default="per_item_B5_conflict_bi")
    ap.add_argument("--agent-name", default="B5-conflict-bi")
    ap.add_argument("--adapter", default="outputs/eval/a3/b5_conflict_bi/adapter_best")
    ap.add_argument("--a3-dir", default="outputs/eval/a3")
    ap.add_argument("--a4-results", default="outputs/eval/a4/a4_results.json")
    ap.add_argument("--matrix", default="outputs/eval/a4/matrix.jsonl")
    ap.add_argument("--corpus-dirs", default="outputs/eval/a0/corpus,outputs/eval/a3/corpus_t3")
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260718)
    ap.add_argument("--out", default="outputs/eval/a5")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a4_path = Path(args.a4_results)
    matrix_path = Path(args.matrix)
    per_item_path = Path(args.a3_dir) / f"{args.agent_file}.json"
    a4 = json.loads(a4_path.read_text())
    rows = json.loads(per_item_path.read_text())
    residuals = compute_residuals(args.adapter, args.corpus_dirs.split(","))
    points = build_points(rows, residuals, a4["M_full"])
    costs = _matrix_costs(a4)
    episode_costs = _matrix_episode_costs(matrix_path)

    protocols = {
        "broad_overlap": points,
        "t2_pair": [p for p in points if p["scenario"] in T2_SCENARIOS],
    }
    results = {
        name: evaluate_protocol(
            name,
            protocol_points,
            costs,
            episode_costs,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed + 100 * index,
        )
        for index, (name, protocol_points) in enumerate(protocols.items())
    }

    result = {
        "schema_version": 2,
        "agent": args.agent_name,
        "score": "Kino-Projector L2 theta residual (larger => abstain)",
        "fallback": SAFE_DEFAULT,
        "selection_protocol": (
            "frozen train appearances calibrate; frozen test appearances evaluate"
        ),
        "input_sha256": {
            "a3_per_item": _sha256(per_item_path),
            "a4_results": _sha256(a4_path),
            "a4_matrix": _sha256(matrix_path),
            "projector": _sha256(Path(args.adapter, "kino_projector.pt")),
        },
        "protocols": results,
    }
    result_path = out / "a5_5_risk_coverage.json"
    result_path.write_text(json.dumps(result, indent=2))
    _render(out, results, costs)

    for name, protocol in results.items():
        selected = protocol["selected_min_cost"]
        print(
            f"[a5.5] {name}: cal/test={protocol['n']} unique_score="
            f"{protocol['score_support']['n_unique_calibration']}/"
            f"{protocol['score_support']['n_unique_test']} test coverage="
            f"{selected['test']['coverage']:.3f} cost={selected['test']['expected_cost']:.3f}"
        )
    print(f"[OK] wrote {result_path} + risk_coverage.png")


def _render(out: Path, results: dict, cost_map: dict[tuple[str, str], float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, (name, protocol) in zip(axes, results.items(), strict=True):
        curve = protocol["test_oracle_curve"]
        ax.plot(
            [p["coverage"] for p in curve],
            [p["expected_cost"] for p in curve],
            "o-",
            lw=1.5,
            color="#7f7f7f",
            label="test curve (oracle diagnostic)",
        )
        deployed = protocol["selected_min_cost"]
        ax.scatter(
            [deployed["test"]["coverage"]],
            [deployed["test"]["expected_cost"]],
            color="#2ca02c",
            s=55,
            zorder=4,
            label="threshold selected on calibration",
        )
        for baseline, color in (("agent", "#d62728"), ("safe", "#1f77b4")):
            ax.axhline(
                deployed["test"]["baseline_cost"][baseline],
                color=color,
                ls="--",
                label=f"always {baseline}",
            )
        ax.set_title(f"{name}\nidentifiable={protocol['score_support']['selective_identifiable']}")
        ax.set_xlabel("coverage on frozen test appearances")
        ax.set_ylabel("expected physical cost")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("A5.5 held-out risk–coverage (selection never uses test costs)")
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
