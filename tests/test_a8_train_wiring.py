from kino_vla.eval.a7_ablation import load_yaml
from scripts.a8_train import ALL_ARMS, resolve_arm


def test_resolve_arm_failcot_and_a8b():
    cfg = load_yaml("configs/eval/a8.yaml")
    ds, ov, out, label = resolve_arm(cfg, "failcot_sft", seed=0, dataset=None, out=None)
    assert "failcot_sft" in out
    assert ov["route"] == "text"
    assert ov["data.proprio_detail"] == "none"
    assert label == "vision_only"

    ds2, ov2, out2, label2 = resolve_arm(cfg, "latent_conflict", seed=1, dataset="/tmp/x", out=None)
    assert ds2 == "/tmp/x"
    assert "a8b_latent_conflict" in out2
    assert ov2["route"] == "latent"
    assert label2 == "latent_conflict"
    assert set(ALL_ARMS) >= {"failcot_sft", "v_only", "latent_conflict"}
