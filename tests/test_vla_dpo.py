"""Embodied-DPO preference-pair construction + loss (spec §11 Stage 2). Pure logic, CI."""

from __future__ import annotations

import pytest

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import load_config
from kino_vla.vla import scenarios as S
from kino_vla.vla.dpo import build_preference_pairs, dpo_loss, pair_stats, save_pairs
from kino_vla.vla.planner import StubVlaPolicy
from kino_vla.vla.rollout import run_vla_rollout


@pytest.fixture(scope="module")
def cfg():
    return load_config("data/hindsight.yaml")


@pytest.fixture(scope="module")
def tax(cfg):
    return FailureTaxonomy(cfg)


def _node_rollouts(cfg, tax):
    """One success (correct) and one failure (wrong) rollout at the same O3 node."""
    scn = S.o3_collapse()
    good = run_vla_rollout(scn, StubVlaPolicy(cfg, tax), seed=0, backend="surrogate")
    bad = run_vla_rollout(
        scn, StubVlaPolicy(cfg, tax, error_mode="sibling"), seed=0, backend="surrogate"
    )
    return [good, bad]


def test_pairs_chosen_is_success_rejected_is_failure(cfg, tax):
    results = _node_rollouts(cfg, tax)
    pairs = build_preference_pairs(results, max_pairs=4)
    assert pairs, "a success+failure node must yield at least one pair"
    p = pairs[0]
    assert p.chosen_attr == "region_collapse"  # the successful (correct) attribution
    assert p.rejected_attr == "low_friction"  # the failed (wrong) attribution
    assert p.reason == "ambiguity"  # differing attributions ⇒ the §11 ambiguity-pair Rejected
    assert "region_collapse" in p.chosen_text and "low_friction" in p.rejected_text


def test_no_pairs_when_all_succeed(cfg, tax):
    """No failure ⇒ no preference signal (the sampler must produce a loss to pair)."""
    good = run_vla_rollout(S.o3_collapse(), StubVlaPolicy(cfg, tax), seed=0, backend="surrogate")
    assert build_preference_pairs([good, good], max_pairs=4) == []


def test_pair_stats_and_save(cfg, tax, tmp_path):
    pairs = build_preference_pairs(_node_rollouts(cfg, tax), max_pairs=4)
    stats = pair_stats(pairs)
    assert stats["n_pairs"] == len(pairs)
    assert "O3_collapse" in stats["by_operator"]
    out = save_pairs(pairs, tmp_path)
    assert (out / "pairs.jsonl").exists() and (out / "pairs.npz").exists()


def test_dpo_loss_rewards_preferring_chosen():
    """DPO loss decreases as the policy raises chosen above rejected relative to the reference."""
    import torch

    ref_c, ref_r = torch.tensor(-5.0), torch.tensor(-5.0)
    # policy prefers chosen (chosen logprob up, rejected down) ⇒ lower loss
    good = dpo_loss(torch.tensor(-3.0), torch.tensor(-7.0), ref_c, ref_r, beta=0.5)
    bad = dpo_loss(torch.tensor(-7.0), torch.tensor(-3.0), ref_c, ref_r, beta=0.5)
    assert float(good) < float(bad)
    assert float(good) > 0.0  # a proper loss is positive
