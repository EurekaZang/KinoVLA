"""Run the paper's registered evaluators without changing their inference code."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/eval/publication_evaluation.json"


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_command(
    registry: dict[str, Any], method: str, output: Path, feature_root: Path
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / registry["evaluation_script"]),
        "score",
        "--freeze",
        str(ROOT / registry["methods"][method]["freeze_dir"]),
        "--all191",
        str(feature_root),
        "--output",
        str(output),
    ]


def check_paths(registry: dict[str, Any], methods: list[str], features: Path) -> None:
    required = [ROOT / p for p in registry["protected_dependencies"]]
    required += [
        features / p
        for p in (
            "scale/features.npz",
            "scale/records.jsonl",
            "scale/dinov2/features.npz",
            "conflict/base_features/features.npz",
            "conflict/base_features/records.jsonl",
            "conflict/invariant_features/features.npz",
            "conflict/dinov2/features.npz",
        )
    ]
    for method in methods:
        spec = registry["methods"][method]
        freeze = ROOT / spec["freeze_dir"]
        required += [freeze / "single_gmu.pt", freeze / "freeze_manifest.json"]
        manifest = freeze / "freeze_manifest.json"
        if manifest.is_file():
            saved = json.loads(manifest.read_text())
            config = saved[spec["freeze_configuration_key"]]
            if config != spec["freeze_configuration"]:
                raise RuntimeError(f"Registered model configuration changed: {method}")
            if [row["seed"] for row in saved["training"]] != spec["training_seeds"]:
                raise RuntimeError(f"Registered training seeds changed: {method}")
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing registered evaluation inputs:\n" + "\n".join(missing))


def compare_predictions(actual: Path, reference: Path) -> int:
    # Compare outputs directly, including row order and all metadata. Small floating
    # point differences across devices are allowed; labels and IDs must be exact.
    actual_rows = [json.loads(line) for line in actual.read_text().splitlines() if line]
    reference_rows = [json.loads(line) for line in reference.read_text().splitlines() if line]
    if len(actual_rows) != len(reference_rows):
        raise RuntimeError(f"Physical-unit count changed: {actual}")

    def equal(left: Any, right: Any) -> bool:
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, float) and isinstance(right, float):
            return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)
        return left == right

    for index, (left, right) in enumerate(zip(actual_rows, reference_rows, strict=True)):
        if not equal(left, right):
            raise RuntimeError(f"Prediction changed at row {index}: {actual}")
    return len(actual_rows)


def verify_scores(registry: dict[str, Any], method: str, output: Path) -> dict[str, Any]:
    spec = registry["methods"][method]
    report = json.loads((output / "report.json").read_text())
    if report["config"] != spec["config"]:
        raise RuntimeError(f"Evaluator used a different configuration: {method}")
    metrics = report["metrics"]["cross_battery"]
    for key, expected in spec["expected_balanced_accuracy"].items():
        if not math.isclose(metrics[key], expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"{method}/{key} changed: {metrics[key]} != {expected}")
    count = compare_predictions(
        output / "physical_units.jsonl",
        ROOT / spec["reference_dir"] / "physical_units.jsonl",
    )
    return {"method": method, "physical_units": count, "metrics": metrics, "passed": True}


def main() -> int:
    registry = load_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["all", *registry["methods"]], default="all")
    parser.add_argument("--feature-root", type=Path, default=Path(registry["feature_root"]))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--verify", action="store_true", help="compare against registered predictions"
    )
    parser.add_argument(
        "--include-action", action="store_true", help="also score the 277 KiNO cases"
    )
    args = parser.parse_args()
    methods = list(registry["methods"]) if args.method == "all" else [args.method]
    if args.include_action and "gmu" not in methods:
        parser.error("--include-action requires --method gmu or all")
    check_paths(registry, methods, args.feature_root)
    if args.check_only:
        print(f"Registered dependencies and inputs are available for {len(methods)} methods.")
        return 0
    if args.output is None:
        parser.error("--output is required unless --check-only is selected")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    verification = []
    for method in methods:
        destination = output / method
        print(f"Scoring {method}: {registry['methods'][method]['freeze_dir']}", flush=True)
        subprocess.run(
            score_command(registry, method, destination, args.feature_root), cwd=ROOT, check=True
        )
        if args.verify:
            verification.append(verify_scores(registry, method, destination))
    if args.include_action:
        action = registry["action"]
        destination = output / "action"
        command = [
            sys.executable,
            str(ROOT / registry["evaluation_script"]),
            "action",
            "--freeze",
            str(ROOT / registry["methods"]["gmu"]["freeze_dir"]),
            "--all191",
            str(args.feature_root),
            "--action-case-csv",
            str(ROOT / action["case_csv"]),
            "--output",
            str(destination),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        if args.verify:
            count = compare_predictions(
                destination / "primary_view_predictions.jsonl",
                ROOT / action["reference_dir"] / "primary_view_predictions.jsonl",
            )
            if count != action["expected_cases"]:
                raise RuntimeError("Action-study sample count changed")
            verification.append({"method": "gmu_action", "cases": count, "passed": True})
    if args.verify:
        (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        print(f"Verified {len(verification)} evaluations against registered predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
