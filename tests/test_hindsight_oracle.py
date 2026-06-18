"""Oracle annotation client tests (spec §10 PHASE 3).

Covers the English prompt template (the QA 5.2 paper artifact), the structured-output parser
(the spec's atomic-action / 2D-pixel-coordinate constraints as schema validation), the
deterministic offline ScriptedOracle, and the external-LLM ApiOracle (with an injected stub
completion, so its transport contract is covered without a network call).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from kino_vla.data import (
    ApiOracle,
    CoTAnnotation,
    CoTParseError,
    FailureTaxonomy,
    RecoveryPrimitive,
    ScriptedOracle,
    build_prompt,
)
from kino_vla.data.schema import Snapshot
from kino_vla.tokens.features import N_FEATURES
from kino_vla.utils.config import load_config


def _snapshot(op: str = "O1_mu_field", appearance: str = "ice_sheet") -> Snapshot:
    return Snapshot(
        operator_name=op,
        appearance_class=appearance,
        t=1.0,
        pose_xy=np.zeros(2),
        heading=0.0,
        rgb=np.zeros((5, 4, 4, 3)),
        depth=np.zeros((5, 4, 4)),
        proprio_window=np.zeros((25, N_FEATURES)),
        prior_outputs=[],
        privileged_theta={"mu": 0.1, "payload_kg": 0.0, "effort_scale": 1.0, "support_ratio": 1.0},
        monitor_channel="slip_ratio",
    )


@pytest.fixture(scope="module")
def cfg():
    return load_config("data/hindsight.yaml")


@pytest.fixture(scope="module")
def taxonomy(cfg) -> FailureTaxonomy:
    return FailureTaxonomy(cfg)


def test_prompt_is_english_only(cfg):
    """All text sent to the Oracle LLM must be English (user directive; paper artifact)."""
    messages = build_prompt(_snapshot(), cfg)
    for m in messages:
        assert m["content"].isascii(), f"non-ASCII in the {m['role']} prompt"


def test_prompt_encodes_spec_constraints(cfg):
    system = next(m["content"] for m in build_prompt(_snapshot(), cfg) if m["role"] == "system")
    # The §5 primitive library and the category vocabulary are listed.
    for primitive in cfg.oracle.prompt.primitive_library:
        assert primitive in system
    for category in cfg.oracle.prompt.category_vocabulary:
        assert category in system
    # The PHASE 3 prompt constraints: single JSON, one atomic action, 2D pixel coord, safety rule.
    assert "SINGLE JSON" in system
    assert "one atomic action" in system
    assert "PIXEL coordinate" in system
    assert "sudden trap" in system


def test_scripted_oracle_output_parses(cfg, taxonomy):
    oracle = ScriptedOracle(cfg, taxonomy, seed=0)
    for op, appr in [("O1_mu_field", "ice_sheet"), ("O5_payload", "solid_ground")]:
        text = oracle.annotate(_snapshot(op, appr))
        ann = CoTAnnotation.from_oracle_text(
            text, synonyms=taxonomy.synonyms, valid_categories=taxonomy.valid_categories
        )
        assert ann.attribution in taxonomy.valid_categories
        assert ann.primitive.name in cfg.oracle.prompt.primitive_library


def test_scripted_oracle_is_deterministic(cfg, taxonomy):
    snaps = [_snapshot("O4_tether", "yellow_adhesive") for _ in range(8)]
    a = [ScriptedOracle(cfg, taxonomy, seed=3).annotate(s) for s in snaps]
    b = [ScriptedOracle(cfg, taxonomy, seed=3).annotate(s) for s in snaps]
    assert a == b


def test_scripted_oracle_fooled_by_decoupled_appearance(cfg, taxonomy):
    """A genuinely-decoupled appearance (ice physics shown as mud) misleads the vision-led read."""
    oracle = ScriptedOracle(cfg, taxonomy, seed=0)
    text = oracle.annotate(_snapshot("O1_mu_field", "brown_mud"))  # ice physics, mud appearance
    ann = CoTAnnotation.from_oracle_text(
        text, synonyms=taxonomy.synonyms, valid_categories=taxonomy.valid_categories
    )
    assert ann.attribution == "compliant_terrain"  # fooled away from the true low_friction


def test_api_oracle_passes_prompt_and_images_and_returns_completion(cfg):
    captured = {}

    def stub_complete(messages, images):
        captured["messages"] = messages
        captured["images"] = images
        return json.dumps(
            {
                "thought": "t",
                "attribution": "low_friction",
                "action": {
                    "primitive": "Set_Constraint",
                    "params": {"max_speed": 0.4, "stiffness": 0.5},
                },
            }
        )

    oracle = ApiOracle(stub_complete, cfg, n_images=2)
    text = oracle.annotate(_snapshot())
    assert captured["messages"][0]["role"] == "system"
    # Multimodal: the rendered RGB frames are attached, and the material class is NOT leaked.
    assert len(captured["images"]) == 2
    assert "ice_sheet" not in captured["messages"][1]["content"]
    assert json.loads(text)["attribution"] == "low_friction"


def test_api_oracle_from_config_requires_provider(cfg, monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="no LLM provider"):
        ApiOracle.from_config(cfg)


# ----------------------------------------------------------------- parser schema
def test_parser_handles_fenced_json(taxonomy):
    body = json.dumps(
        {
            "thought": "x",
            "attribution": "ice",  # synonym, normalized to low_friction
            "action": {
                "primitive": "Set_Constraint",
                "params": {"max_speed": 0.4, "stiffness": 0.5},
            },
        }
    )
    ann = CoTAnnotation.from_oracle_text(
        f"```json\n{body}\n```",
        synonyms=taxonomy.synonyms,
        valid_categories=taxonomy.valid_categories,
    )
    assert ann.attribution == "low_friction"  # 'ice' normalized via synonyms


def test_parser_handles_prose_with_leading_braces(taxonomy):
    """m4: an Oracle reply with brace-containing prose BEFORE the JSON still parses (greedy
    find/rfind used to span both and fail → mis-counted as schema_invalid)."""
    text = "Reasoning {the surface looks icy}; my final answer:\n" + json.dumps(
        {
            "thought": "x",
            "attribution": "low_friction",
            "action": {
                "primitive": "Set_Constraint",
                "params": {"max_speed": 0.4, "stiffness": 0.5},
            },
        }
    )
    ann = CoTAnnotation.from_oracle_text(
        text, synonyms=taxonomy.synonyms, valid_categories=taxonomy.valid_categories
    )
    assert ann.attribution == "low_friction"
    assert ann.primitive.name == "Set_Constraint"


def test_parser_rejects_non_pixel_replan(taxonomy):
    text = json.dumps(
        {
            "thought": "x",
            "attribution": "invisible_obstacle",
            "action": {"primitive": "Replan_Waypoint", "params": {"point_px": [1.0]}},
        }
    )
    with pytest.raises(CoTParseError):
        CoTAnnotation.from_oracle_text(
            text, synonyms=taxonomy.synonyms, valid_categories=taxonomy.valid_categories
        )


def test_recovery_primitive_validation():
    with pytest.raises(CoTParseError):
        RecoveryPrimitive("Backstep", {"distance_m": -1.0})
    with pytest.raises(CoTParseError):
        RecoveryPrimitive("Switch_Gait", {"mode": "moonwalk"})
    with pytest.raises(CoTParseError):
        RecoveryPrimitive("NotAPrimitive", {})
    # a valid one constructs fine
    assert (
        RecoveryPrimitive("Set_Constraint", {"max_speed": 0.4, "stiffness": 0.5}).name
        == "Set_Constraint"
    )
