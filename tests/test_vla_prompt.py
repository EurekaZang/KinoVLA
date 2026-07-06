"""VLA planner prompt + target formatting (spec §10 PHASE 4 / §11)."""

from __future__ import annotations

import numpy as np
import pytest

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive, Snapshot
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import load_config
from kino_vla.vla.output import parse_vla_decision
from kino_vla.vla.prompt import (
    KINO_TAG,
    build_messages,
    context_from_snapshot,
    format_target,
    system_prompt,
    user_text,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config("data/hindsight.yaml")


def _snapshot(op="O1_mu_field", appr="ice_sheet"):
    w = np.zeros((25, 11), dtype=np.float32)
    w[:, 7] = 0.6  # slip_ratio channel
    return Snapshot(
        operator_name=op,
        appearance_class=appr,
        t=2.0,
        pose_xy=np.array([2.5, 0.0]),
        heading=0.0,
        rgb=np.zeros((5, 8, 8, 3), dtype=np.float32),
        depth=np.ones((5, 8, 8), dtype=np.float32),
        proprio_window=w,
        prior_outputs=[],
        privileged_theta={"mu": 0.1},
        monitor_channel="slip",
    )


def test_system_prompt_is_english_and_lists_primitives(cfg):
    sp = system_prompt(cfg)
    assert sp.isascii(), "prompt must be all-English ASCII (project directive)"
    for prim in cfg.oracle.prompt.primitive_library:
        assert prim in sp
    for cat in cfg.oracle.prompt.category_vocabulary:
        assert cat in sp
    assert "<Thought>" in sp and "<Action>" in sp
    assert "iron-rule" in sp.lower() or "escape first" in sp.lower()


def test_latent_route_has_kino_tag_no_numbers(cfg):
    ctx = context_from_snapshot(_snapshot(), route="latent")
    txt = user_text(ctx, route="latent")
    assert KINO_TAG in txt
    assert ctx.proprio_summary is None


def test_text_route_embeds_proprio_summary(cfg):
    ctx = context_from_snapshot(_snapshot(), route="text")
    txt = user_text(ctx, route="text")
    assert KINO_TAG not in txt
    assert "slip_trace" in txt  # the _proprio_summary time-series is inlined


def test_build_messages_image_placeholders(cfg):
    ctx = context_from_snapshot(_snapshot(), route="latent")
    msgs = build_messages(ctx, cfg, route="latent", n_images=2)
    assert msgs[0]["role"] == "system"
    user = msgs[1]
    n_images = sum(1 for c in user["content"] if c["type"] == "image")
    assert n_images == 2
    assert any(c["type"] == "text" for c in user["content"])
    # n_images=0 → text-only
    msgs0 = build_messages(ctx, cfg, route="latent", n_images=0)
    assert all(c["type"] != "image" for c in msgs0[1]["content"])


def test_reveal_appearance_names_surface(cfg):
    ctx = context_from_snapshot(
        _snapshot(appr="yellow_adhesive"), route="text", reveal_appearance=True
    )
    assert "yellow_adhesive" in user_text(ctx, route="text")
    ctx2 = context_from_snapshot(
        _snapshot(appr="yellow_adhesive"), route="text", reveal_appearance=False
    )
    assert "yellow_adhesive" not in user_text(ctx2, route="text")


def test_proprio_fidelity_levels_render_distinctly(cfg):
    """The §3 information-fidelity ablation knob (Gap-1, #35): binned ⊃ scalar ⊃ none."""
    snap = _snapshot()
    t_binned = user_text(
        context_from_snapshot(snap, route="text", proprio_detail="binned"), route="text"
    )
    assert "slip_trace" in t_binned and "slip_mean" in t_binned  # full waveform
    t_scalar = user_text(
        context_from_snapshot(snap, route="text", proprio_detail="scalar"), route="text"
    )
    assert "slip_mean" in t_scalar  # the sustained mean survives
    assert "slip_trace" not in t_scalar and "effort_trace" not in t_scalar  # the SHAPE is dropped
    assert "sustained means" in t_scalar
    ctx_none = context_from_snapshot(snap, route="text", proprio_detail="none")
    t_none = user_text(ctx_none, route="text")
    assert ctx_none.proprio_summary is None  # vision-only floor
    assert "slip_mean" not in t_none and "slip_trace" not in t_none and "not provided" in t_none


def test_reduce_proprio_keeps_means_drops_traces(cfg):
    from kino_vla.data.oracle import _proprio_summary
    from kino_vla.vla.prompt import _SCALAR_KEYS, _reduce_proprio

    full = _proprio_summary(_snapshot())
    assert _reduce_proprio(full, "binned") == full
    scalar = _reduce_proprio(full, "scalar")
    assert set(scalar) <= set(_SCALAR_KEYS) and all("trace" not in k for k in scalar)
    assert _reduce_proprio(full, "none") is None


def test_invalid_proprio_detail_raises(cfg):
    with pytest.raises(ValueError):
        context_from_snapshot(_snapshot(), route="text", proprio_detail="bogus")


def test_nav_system_prompt_turn_grammar_is_unconditional(cfg):
    """#43 data-level fix: the nav prompt now ALWAYS describes both Replan_Waypoint AND Turn (no
    has_hazard gate), so training and deployment share one prompt and the VLA can turn in cruise."""
    from kino_vla.vla.prompt import nav_system_prompt

    sp_free = nav_system_prompt(cfg, has_hazard=False)
    sp_hazard = nav_system_prompt(cfg, has_hazard=True)
    assert sp_free == sp_hazard, "the Turn grammar no longer depends on has_hazard"
    assert sp_free.isascii(), "prompt must be all-English ASCII (project directive)"
    assert '"primitive": "Replan_Waypoint"' in sp_free
    assert '"primitive": "Turn"' in sp_free and "yaw_deg" in sp_free
    assert "in place" in sp_free.lower()  # Turn framed as an in-place (active-perception) rotation


def test_format_target_round_trips_through_parser(cfg):
    """The SFT target must itself parse back to the same atomic decision (no train/serve skew)."""
    tax = FailureTaxonomy(cfg)
    ann = CoTAnnotation(
        thought="The slip trace is sustained high on a reflective surface.",
        attribution="low_friction",
        primitive=RecoveryPrimitive("Set_Constraint", {"max_speed": 0.4, "stiffness": 0.5}),
        attribution_raw="low_friction",
        raw_text="",
    )
    target = format_target(ann)
    assert target.startswith("<Thought>") and "<Action>" in target
    parsed = parse_vla_decision(
        target, synonyms=tax.synonyms, valid_categories=tax.valid_categories
    )
    assert parsed.ok
    assert parsed.attribution == "low_friction"
    assert parsed.primitive_name == "Set_Constraint"
