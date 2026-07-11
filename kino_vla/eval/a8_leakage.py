"""A8 leakage firewall: allowed model inputs vs eval-only metadata."""

from __future__ import annotations

from typing import Any, Mapping

# Model may consume only these families at train/eval time.
ALLOWED_INPUT_FIELDS: frozenset[str] = frozenset(
    {
        "sample_id",
        "images",
        "image",
        "views",
        "task_instruction",
        "instruction",
        "subtask_instruction",
        "robot_state",
        "proprio",
        "proprio_window",
        "state_window",
        "state_summary",
        "prompt_policy",
        "split",
        "track",
        "arm",
        "route",
        "proprio_detail",
    }
)

# Never feed these into model prompts/features; use only for labels/strata/oracles.
EVAL_ONLY_FIELDS: frozenset[str] = frozenset(
    {
        "failure_mode",
        "failure_reason",
        "reward",
        "planning_reward",
        "execution_reward",
        "start_caption",
        "end_caption",
        "gt_item",
        "wrong_item",
        "ground_truth_answer",
        "label",
        "correct_plan",
        "old_plan",
        "old_task_instruction",
        "old_visible_objects",
        "visible_objects",
        "history_plans",
        "plan",
        "stratum",  # derived label; kept in sidecars, not model text
        "oracle_modality",
    }
)


def filter_allowed_fields(
    sample: Mapping[str, Any],
    *,
    mode: str = "model_input",
) -> dict[str, Any]:
    """Return a shallow-filtered sample for the requested mode.

    mode="model_input": only ALLOWED_INPUT_FIELDS (plus unknown non-eval fields dropped).
    mode="eval": keep everything (labels available to metrics).
    mode="sidecar": keep eval-only + ids for stratification artifacts.
    """
    if mode == "eval":
        return dict(sample)
    if mode == "sidecar":
        keep = {"sample_id", "track", "split", "stratum"} | set(EVAL_ONLY_FIELDS)
        return {k: v for k, v in sample.items() if k in keep}
    if mode != "model_input":
        raise ValueError(f"unknown filter mode: {mode!r}")
    out: dict[str, Any] = {}
    for k, v in sample.items():
        if k in EVAL_ONLY_FIELDS:
            continue
        if k in ALLOWED_INPUT_FIELDS:
            out[k] = v
    return out


def assert_no_leakage(sample: Mapping[str, Any]) -> None:
    """Raise if a model-input sample still contains eval-only fields."""
    leaked = sorted(k for k in sample if k in EVAL_ONLY_FIELDS)
    if leaked:
        raise ValueError(f"eval-only fields leaked into model input: {leaked}")
