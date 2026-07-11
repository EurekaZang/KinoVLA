from pathlib import Path

from kino_vla.eval.a8_state_audit import audit_reflect_state


def test_audit_fails_without_state(tmp_path: Path):
    ep = tmp_path / "ep0"
    ep.mkdir()
    (ep / "rgb.png").write_bytes(b"x")
    out = audit_reflect_state(tmp_path)
    assert out["pass"] is False
    assert out["n_state_episodes"] == 0 or "state" in out["reason"].lower()


def test_audit_passes_with_rgb_and_state(tmp_path: Path):
    for i in range(12):
        ep = tmp_path / f"ep{i}"
        ep.mkdir()
        (ep / "front_rgb.png").write_bytes(b"img")
        (ep / "robot_state.npy").write_bytes(b"\x00\x01")
    out = audit_reflect_state(tmp_path)
    assert out["pass"] is True
    assert out["n_multisensory_episodes"] >= 10
