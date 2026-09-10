#!/usr/bin/env python3
"""Audit KiNO-v4 with decision-window proprioception only.

The original development bundle contains an 80-D descriptor computed from the
registered 21-sample decision window and a 190-D compatibility descriptor whose
T3 source predates that alignment.  This model-blind correction keeps the
aligned descriptor and replaces the compatibility block by zeros for *every*
battery and cell.  Consequently no battery can be identified from a selective
missing-feature pattern and no post-decision trajectory statistic can enter the
model.

This script delegates all fitting, folds, visual inputs, labels, grouping, and
metrics to the existing development evaluator.  It changes only the uniformly
disabled compatibility block and is intended for development audit before a
new checkpoint is frozen.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts/develop_kinofail_conflict_invariant_kino_v4.py"
DEFAULT_OUTPUT = (
    ROOT / "outputs/eval/kino_v4_decision_window_invariant_only_v1_development"
)


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location(
        "kino_v4_decision_window_development_base", BASE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()


_load_scale_base = base._load_scale
_load_conflict_base = base._load_conflict


def _decision_window_only(source: dict) -> dict:
    result = dict(source)
    full = np.asarray(result["full"], dtype=np.float32)
    if full.ndim != 2 or full.shape[1] != 190:
        raise RuntimeError(f"unexpected compatibility block: {full.shape}")
    invariant = np.asarray(result["invariant"], dtype=np.float32)
    if invariant.ndim != 2 or invariant.shape != (len(full), 80):
        raise RuntimeError(f"unexpected decision-window block: {invariant.shape}")
    if not np.isfinite(invariant).all():
        raise RuntimeError("decision-window descriptor contains non-finite values")
    result["full"] = np.zeros_like(full, dtype=np.float32)
    return result


def _load_scale() -> dict:
    return _decision_window_only(_load_scale_base())


def _load_conflict() -> dict:
    return _decision_window_only(_load_conflict_base())


def main() -> int:
    base._load_scale = _load_scale
    base._load_conflict = _load_conflict
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
