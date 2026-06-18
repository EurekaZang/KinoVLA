"""VLA output parser: 100% parse-or-reject + correct compiler mapping (M7 exit criterion 2)."""

from __future__ import annotations

import json

import pytest

from kino_vla.data.schema import CoTParseError, RecoveryPrimitive
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.shield.primitive_compiler import (
    AdjustPosture,
    Backstep,
    HoldAndRequest,
    ReplanWaypoint,
    SetConstraint,
    SwitchGait,
    UpdateTopology,
)
from kino_vla.utils.config import load_config
from kino_vla.vla.output import REJECT_SCHEMA, parse_vla_decision, to_compiler_primitive


@pytest.fixture(scope="module")
def vocab():
    tax = FailureTaxonomy(load_config("data/hindsight.yaml"))
    return tax.synonyms, tax.valid_categories


def _action(attribution: str, primitive: str, params: dict) -> str:
    action = json.dumps({"attribution": attribution, "primitive": primitive, "params": params})
    return f"<Thought>reasoning here</Thought>\n<Action>{action}</Action>"


def test_valid_thought_action_parses(vocab):
    syn, cats = vocab
    out = parse_vla_decision(
        _action("low_friction", "Set_Constraint", {"max_speed": 0.4, "stiffness": 0.5}),
        synonyms=syn,
        valid_categories=cats,
    )
    assert out.ok
    assert out.attribution == "low_friction"
    assert out.primitive_name == "Set_Constraint"
    assert out.thought == "reasoning here"
    assert out.reject_code == ""


def test_bare_json_also_parses(vocab):
    syn, cats = vocab
    bare = json.dumps(
        {
            "thought": "t",
            "attribution": "adhesion",
            "action": {"primitive": "Backstep", "params": {"distance_m": 0.5}},
        }
    )
    out = parse_vla_decision(bare, synonyms=syn, valid_categories=cats)
    assert out.ok and out.primitive_name == "Backstep"


def test_synonym_normalized(vocab):
    syn, cats = vocab
    out = parse_vla_decision(
        _action("ice", "Set_Constraint", {"max_speed": 0.3, "stiffness": 0.5}),
        synonyms=syn,
        valid_categories=cats,
    )
    assert out.ok and out.attribution == "low_friction"  # "ice" → low_friction


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty
        "not json at all",
        "<Thought>only thought, no action</Thought>",
        _action("low_friction", "Teleport", {}),  # primitive not in §5 library
        _action("banana", "Set_Constraint", {"max_speed": 0.3}),  # off-vocabulary attribution
        _action("low_friction", "Backstep", {}),  # missing distance_m
        _action("low_friction", "Backstep", {"distance_m": -1.0}),  # non-positive
        _action("low_friction", "Switch_Gait", {"mode": "moonwalk"}),  # bad gait
        _action("low_friction", "Replan_Waypoint", {"point_px": [1, 2, 3]}),  # not 2D
        _action("low_friction", "Set_Constraint", {}),  # missing max_speed
        '<Action>{"attribution": "low_friction"}</Action>',  # missing action object
        "{not: valid, json}",
    ],
)
def test_malformed_is_rejected_never_raises(vocab, text):
    """Every malformed output is a structured reject — never an exception, never silently ok."""
    syn, cats = vocab
    out = parse_vla_decision(text, synonyms=syn, valid_categories=cats)
    assert out.ok is False
    assert out.annotation is None
    assert out.reject_code.startswith(REJECT_SCHEMA)


def test_parse_or_reject_is_total(vocab):
    """100% of outputs either parse (ok) or carry a reject code — the exit-criterion-2 invariant."""
    syn, cats = vocab
    fuzz = [
        "",
        "garbage",
        _action("low_friction", "Set_Constraint", {"max_speed": 0.4, "stiffness": 0.5}),
        "<Action>not-json</Action>",
        json.dumps(
            {
                "attribution": "overload",
                "action": {"primitive": "Hold_and_Request", "params": {"reason": "x"}},
            }
        ),
        "```json\n{}\n```",
        'note {"attribution": "adhesion", "action": '
        '{"primitive": "Backstep", "params": {"distance_m": 0.6}}} end',
    ]
    for t in fuzz:
        out = parse_vla_decision(t, synonyms=syn, valid_categories=cats)
        assert out.ok ^ (out.reject_code != "")  # exactly one of ok / reject


# --------------------------------------------------------------- compiler mapping
def test_to_compiler_primitive_context_free():
    assert to_compiler_primitive(RecoveryPrimitive("Backstep", {"distance_m": 0.5})) == Backstep(
        0.5
    )
    assert to_compiler_primitive(RecoveryPrimitive("Switch_Gait", {"mode": "crawl"})) == SwitchGait(
        "crawl"
    )
    assert to_compiler_primitive(
        RecoveryPrimitive("Adjust_Posture", {"body_height_m": 0.25, "pitch_deg": 1.0})
    ) == AdjustPosture(0.25, 1.0)
    assert to_compiler_primitive(
        RecoveryPrimitive("Set_Constraint", {"max_speed": 0.4, "stiffness": 0.6})
    ) == SetConstraint(0.4, 0.6)
    assert to_compiler_primitive(
        RecoveryPrimitive("Hold_and_Request", {"reason": "saturated"})
    ) == HoldAndRequest("saturated")


def test_replan_needs_unprojected_point():
    rp = RecoveryPrimitive("Replan_Waypoint", {"point_px": [480, 360]})
    with pytest.raises(CoTParseError):
        to_compiler_primitive(rp)  # pixel coord present, but no odometry point supplied
    assert to_compiler_primitive(rp, point_xy=(1.5, 0.2)) == ReplanWaypoint((1.5, 0.2))


def test_update_topology_uses_params_or_context():
    rp = RecoveryPrimitive(
        "Update_Topology", {"region_xy": [2.0, 0.0], "radius_m": 0.6, "status": "untraversable"}
    )
    got = to_compiler_primitive(rp)
    assert got == UpdateTopology((2.0, 0.0), 0.6, "untraversable")
    # planner-supplied odometry overrides
    got2 = to_compiler_primitive(rp, region_xy=(3.0, 1.0), region_radius_m=0.8)
    assert got2 == UpdateTopology((3.0, 1.0), 0.8, "untraversable")
