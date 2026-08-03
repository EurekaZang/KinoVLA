from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from kino_vla.eval.visual_shell_preflight import evaluate_visual_shell_panorama


THRESHOLDS = {
    "minimum_luminance_std": 18.0,
    "minimum_5bit_colors": 64,
    "maximum_black_fraction": 0.25,
    "maximum_white_fraction": 0.25,
    "maximum_horizontal_seam_mae": 35.0,
}


def test_complex_seamless_panorama_passes(tmp_path: Path) -> None:
    x = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    y = np.linspace(0, 1, 64)[:, None]
    base = (127 + 90 * np.sin(x)[None, :] * (0.5 + y / 2)).astype(np.uint8)
    rgb = np.stack([base, np.roll(base, 7, axis=1), np.roll(base, 13, axis=1)], axis=-1)
    rgb[:, -1] = rgb[:, 0]
    path = tmp_path / "pano.png"
    Image.fromarray(rgb).save(path)
    result = evaluate_visual_shell_panorama(
        path,
        expected_width=128,
        expected_height=64,
        thresholds=THRESHOLDS,
    )
    assert result["passed"], result


def test_flat_or_broken_panorama_fails(tmp_path: Path) -> None:
    rgb = np.zeros((64, 128, 3), dtype=np.uint8)
    rgb[:, -1] = 255
    path = tmp_path / "bad.png"
    Image.fromarray(rgb).save(path)
    result = evaluate_visual_shell_panorama(
        path,
        expected_width=128,
        expected_height=64,
        thresholds=THRESHOLDS,
    )
    assert not result["passed"]
    assert not result["checks"]["not_mostly_black"]
    assert not result["checks"]["horizontal_seam"]

