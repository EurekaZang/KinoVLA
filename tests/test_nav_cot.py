"""Nominal-nav CoT truth-consistency filter (the §10 filter for nav) — golden cases.

Mirrors tests/test_hindsight_dataops.py's recovery-filter golden: each Oracle nav CoT is judged
against the privileged geometric teacher, and the verdict reason is a single, asserted code.
"""

from __future__ import annotations

import pytest

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import load_config
from kino_vla.vla.nav_cot import (
    DROP_DIRECTION,
    DROP_KIND,
    DROP_PIXEL_SIDE,
    DROP_SCHEMA,
    KEEP,
    extract_thought,
    filter_nav_cot,
    format_nav_target,
    teacher_action_brief,
    thought_coherent,
)
from kino_vla.vla.output import parse_nav_decision


@pytest.fixture(scope="module")
def tax():
    return FailureTaxonomy(load_config("data/hindsight.yaml"))


def _turn(d: float) -> str:
    return (
        "<Thought>The patch fills the view ahead; I rotate to bring the clear corridor into "
        f'view.</Thought>\n<Action>{{"attribution":"nominal","primitive":"Turn",'
        f'"params":{{"yaw_deg":{d}}}}}</Action>'
    )


def _wp(u: int, v: int) -> str:
    return (
        "<Thought>Clear ground continues toward the goal on the open side.</Thought>\n"
        f'<Action>{{"attribution":"nominal","primitive":"Replan_Waypoint",'
        f'"params":{{"point_px":[{u},{v}]}}}}</Action>'
    )


T_TURN_LEFT = {"kind": "turn", "yaw_deg": 45.0}
T_WP_LEFT = {"kind": "waypoint", "point_px": [250, 400]}


def _judge(text, teacher, tax):
    return filter_nav_cot(
        text, teacher, synonyms=tax.synonyms, valid_categories=tax.valid_categories
    )


def test_keep_turn_same_direction(tax):
    _, v = _judge(_turn(40), T_TURN_LEFT, tax)
    assert v.keep and v.reason == KEEP


def test_keep_waypoint_same_side(tax):
    _, v = _judge(_wp(300, 380), T_WP_LEFT, tax)
    assert v.keep and v.reason == KEEP


def test_drop_direction_wrong_way_turn(tax):
    """The dangerous confabulation: the Oracle turns RIGHT where the teacher routes LEFT."""
    _, v = _judge(_turn(-40), T_TURN_LEFT, tax)
    assert not v.keep and v.reason == DROP_DIRECTION


def test_drop_kind_turn_vs_waypoint(tax):
    _, v = _judge(_wp(250, 400), T_TURN_LEFT, tax)
    assert not v.keep and v.reason == DROP_KIND


def test_drop_pixel_side_routes_through_patch(tax):
    """The Oracle drives a pixel to the RIGHT (toward/through the patch) where the teacher skirts
    LEFT — a route-around-side confabulation."""
    _, v = _judge(_wp(780, 400), T_WP_LEFT, tax)
    assert not v.keep and v.reason == DROP_PIXEL_SIDE


def test_drop_schema_unparseable(tax):
    _, v = _judge("I think we should go left, probably.", T_TURN_LEFT, tax)
    assert not v.keep and v.reason == DROP_SCHEMA


def test_drop_schema_non_nav_primitive(tax):
    """A recovery primitive (Backstep) is not a nominal-nav action ⇒ DROP_SCHEMA."""
    text = (
        '<Thought>back out</Thought>\n<Action>{"attribution":"nominal",'
        '"primitive":"Backstep","params":{"distance_m":0.5}}</Action>'
    )
    _, v = _judge(text, T_TURN_LEFT, tax)
    assert not v.keep and v.reason == DROP_SCHEMA


def test_near_centre_teacher_does_not_over_reject(tax):
    """A near-straight teacher turn/waypoint must not drop on a small direction disagreement."""
    _, v = _judge(_turn(-5), {"kind": "turn", "yaw_deg": 4.0}, tax)
    assert v.keep, "teacher turn within the deadband ⇒ direction not enforced"
    _, v2 = _judge(_wp(560, 400), {"kind": "waypoint", "point_px": [490, 400]}, tax)
    assert v2.keep, "teacher waypoint near centre ⇒ side not enforced"


def test_teacher_action_brief_describes_action():
    assert "TURN" in teacher_action_brief(T_TURN_LEFT) and "left" in teacher_action_brief(
        T_TURN_LEFT
    )
    assert "Replan_Waypoint" in teacher_action_brief(T_WP_LEFT)


def test_thought_coherent_keeps_turn_reasoning_drops_contradictions():
    # turn rationale with turn language ⇒ coherent; "straight ahead" with no turn word ⇒ incoherent
    assert thought_coherent("Rotate left to bring the clear corridor into view.", "turn")
    assert thought_coherent("The patch blocks the way; turn to the side.", "turn")
    assert not thought_coherent("Clear ground is straight ahead, continue forward.", "turn")
    assert not thought_coherent("", "turn")
    # waypoint rationales are permissive
    assert thought_coherent("Skirt the patch on the clear right side toward the goal.", "waypoint")


def test_extract_thought_reads_reasoning_or_defaults():
    assert "rotate" in extract_thought(_turn(40)).lower()
    assert extract_thought("no tags here") == "Route toward the goal on clear ground."  # default


def test_format_nav_target_round_trips(tax):
    """The canonical target (Oracle thought + teacher action) parses back to the teacher action."""
    tgt_turn = format_nav_target("Rotate left to reveal the route.", T_TURN_LEFT)
    d = parse_nav_decision(tgt_turn, synonyms=tax.synonyms, valid_categories=tax.valid_categories)
    assert d.ok and d.nav_turn_deg == pytest.approx(45.0)

    tgt_wp = format_nav_target("Skirt the patch on the open side.", T_WP_LEFT)
    d2 = parse_nav_decision(tgt_wp, synonyms=tax.synonyms, valid_categories=tax.valid_categories)
    assert d2.ok and d2.primitive_name == "Replan_Waypoint"
    assert list(d2.annotation.primitive.params["point_px"]) == [250, 400]
