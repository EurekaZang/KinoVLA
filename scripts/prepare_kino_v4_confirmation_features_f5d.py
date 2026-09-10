#!/usr/bin/env python3
"""Prepare frozen features from the F5c observation seal."""

from __future__ import annotations

from pathlib import Path

from scripts import prepare_kino_v4_confirmation_features_f4m as predecessor


ROOT = Path(__file__).resolve().parents[1]
OBSERVATION = (
    ROOT / "outputs/freeze/kino_v4_confirmation_observations_f5c/observation_seal.json"
)
OUTPUT = ROOT / "outputs/eval/kino_v4_confirmation_v1_f5d"
COMBINED_CORPUS = Path(
    "/data/eureka/kinofail_kino_v4_confirmation_t3_combined_f5d/corpus"
)


def main() -> int:
    module = predecessor.module()
    module.F4 = OBSERVATION
    module.DEFAULT_OUTPUT = OUTPUT
    module.T3_CORPUS = COMBINED_CORPUS
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
