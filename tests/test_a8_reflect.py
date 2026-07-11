import json
from pathlib import Path

import numpy as np

from kino_vla.eval.a8_reflect import audit_reflect_root, build_a8b_cards_from_reflect_zarr


def _make_fake_episode(root: Path, name: str) -> Path:
    ep = root / name
    color = ep / "videos" / "color"
    color.mkdir(parents=True)
    # write tiny PNG via PIL
    from PIL import Image

    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(color / "1.0.0.0", format="PNG")
    Image.fromarray(np.ones((16, 16, 3), dtype=np.uint8) * 10).save(color / "2.0.0.0", format="PNG")

    zdata = ep / "replay_buffer.zarr" / "data"
    for key, arr in {
        "robot_joint": np.linspace(0, 1, 50 * 7, dtype=np.float32).reshape(50, 7),
        "gripper_pos": np.linspace(0.1, 0.01, 50, dtype=np.float32),
        "gripper_state": np.array([0] * 25 + [1] * 25, dtype=np.float32),
        "gripper_force": np.random.randn(50).astype(np.float32),
    }.items():
        d = zdata / key
        d.mkdir(parents=True)
        flat = np.asarray(arr).reshape(-1)
        (d / "0").write_bytes(flat.astype(np.float32).tobytes())
        shape = list(arr.shape) if arr.ndim > 0 else [arr.size]
        (d / ".zarray").write_text(
            json.dumps(
                {
                    "dtype": "<f4",
                    "shape": shape,
                    "chunks": shape,
                    "compressor": None,
                    "fill_value": 0,
                    "order": "C",
                    "zarr_format": 2,
                }
            )
        )
    return ep


def test_audit_and_cards_from_fake_reflect(tmp_path: Path):
    for i in range(12):
        _make_fake_episode(tmp_path, f"task{i}")
    audit = audit_reflect_root(tmp_path)
    assert audit["pass"] is True
    assert audit["n_multisensory_episodes"] >= 10
    assert "robot_joint" in audit["state_fields"]

    cards = build_a8b_cards_from_reflect_zarr(tmp_path, limit=5)
    assert len(cards) == 5
    assert all(c.get("robot_state") for c in cards)
    assert all(c.get("stratum") in {"E1", "E2", "E3", "E4", "unassigned"} for c in cards)
    assert all(c.get("binary_label") == "failure" for c in cards)
