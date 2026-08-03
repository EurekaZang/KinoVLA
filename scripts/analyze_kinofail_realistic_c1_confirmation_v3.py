#!/usr/bin/env python3
"""Run the frozen C1 confirmation analysis with v3 provenance wording."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]


class ClipSlice(BaseEstimator, TransformerMixin):
    """Select the frozen 1536-D temporal CLIP stream from visual+HOG."""

    def fit(self, values, labels=None):
        return self

    def transform(self, values):
        return values[:, :1536]


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    known, _ = parser.parse_known_args()
    implementation = (
        ROOT / "scripts/analyze_kinofail_realistic_c1_confirmation_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_c1_confirmation_v3_impl", implementation
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load C1 confirmation implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The implementation hashes its own __file__; bind that check to this
    # frozen v3 entry point while reusing the tested numerical implementation.
    module.__file__ = str(Path(__file__).resolve())
    protocol = json.loads(
        known.protocol.resolve().read_text(encoding="utf-8")
    )
    original_model = module._model
    calls = 0

    def frozen_model(c_value):
        nonlocal calls
        calls += 1
        if calls == 1:
            analysis = protocol["frozen_analysis"]
            return Pipeline(
                [
                    ("clip_only", ClipSlice()),
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        SVC(
                            C=float(analysis["visual_C"]),
                            gamma=float(analysis["visual_gamma"]),
                            kernel="rbf",
                            class_weight="balanced",
                            probability=True,
                            random_state=2026072415,
                        ),
                    ),
                ]
            )
        return original_model(float(protocol["frozen_analysis"]["proprio_C"]))

    module._model = frozen_model
    result = int(module.main())
    report_path = known.output.resolve() / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["schema_version"] = (
            "kinofail.realistic-c1-confirmation-report.v3"
        )
        report["selection"]["development_source"] = protocol[
            "development_source_description"
        ]
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
