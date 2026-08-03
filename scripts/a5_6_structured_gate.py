#!/usr/bin/env python
"""A5.6 — fit and audit the registry-consistent structured recovery gate.

This first artifact is explicitly retrospective because the existing appearance-test split was
already inspected during A5/A7 method development.  It establishes feasibility and freezes the
gate specification; a new final-validation corpus is required before the result becomes a C4
headline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kino_vla.eval.a4_consequence import Outcome, physical_cost
from kino_vla.eval.structured_gate import bootstrap_gate, evaluate_gate, fit_structured_gate
from kino_vla.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_costs(path: Path) -> dict[tuple[str, str], list[float]]:
    cells: dict[tuple[str, str], list[float]] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        outcome = Outcome(**json.loads(line))
        cells.setdefault((outcome.scenario, outcome.label), []).append(physical_cost(outcome))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="A5.6 structured gate retrospective audit")
    parser.add_argument(
        "--points", default="outputs/eval/a7/abstention_baselines/per_item_scores.json"
    )
    parser.add_argument("--matrix", default="outputs/eval/a4/matrix.jsonl")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--out", default="outputs/eval/a5/a5_6_structured_gate.json")
    args = parser.parse_args()

    points_path = Path(args.points)
    matrix_path = Path(args.matrix)
    points = json.loads(points_path.read_text())
    calibration = [row for row in points if row.get("appearance_split") == "train"]
    test = [row for row in points if row.get("appearance_split") == "test"]
    cfg = load_config("data/hindsight.yaml")
    canonical = {str(k): str(v) for k, v in cfg.recovery.canonical.to_dict().items()}
    gate = fit_structured_gate(calibration, canonical)
    result = {
        "schema_version": 1,
        "status": "retrospective_feasibility_not_headline",
        "reason": "the existing test appearances were inspected before this gate was frozen",
        "method": "registry-consistent action/posterior gate; frozen VLA unchanged",
        "input_sha256": {
            "points": _sha256(points_path),
            "matrix": _sha256(matrix_path),
        },
        "n": {"calibration": len(calibration), "retrospective_test": len(test)},
        "gate": gate,
        "calibration": evaluate_gate(calibration, gate),
        "retrospective_test": evaluate_gate(test, gate),
        "retrospective_two_stage_bootstrap": bootstrap_gate(
            test,
            gate,
            _episode_costs(matrix_path),
            reps=args.bootstrap_reps,
            seed=args.bootstrap_seed,
        ),
        "headline_gate": {
            "eligible": False,
            "required_next": "new frozen final appearances and/or severity settings",
        },
        "non_interference": {
            "attributor_weights_modified": False,
            "training_data_modified": False,
            "a1_a4_estimands_modified": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
