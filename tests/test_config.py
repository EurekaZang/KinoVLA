"""Config loader: YAML loading, attribute/dotted access, overrides, read-only guarantee."""

import pytest

from kino_vla.utils.config import CONFIGS_DIR, Config, load_config


def test_default_config_loads():
    cfg = load_config("default.yaml")
    assert cfg.seed == 42
    assert cfg.sim.physics_dt == pytest.approx(0.001)


def test_go2_flat_config_loads():
    cfg = load_config("sim/go2_flat.yaml")
    assert cfg.stand_check.min_base_height_m < cfg.stand_check.max_base_height_m
    assert cfg.episode.duration_s > 0


def test_dotted_get_and_default():
    cfg = load_config("default.yaml")
    assert cfg.get("sim.physics_dt") == pytest.approx(0.001)
    assert cfg.get("sim.nonexistent", "fallback") == "fallback"
    assert cfg.get("no.such.path") is None


def test_overrides():
    cfg = load_config("default.yaml", overrides={"seed": 7, "sim.device": "cpu"})
    assert cfg.seed == 7
    assert cfg.sim.device == "cpu"
    # Untouched keys survive.
    assert cfg.sim.physics_dt == pytest.approx(0.001)


def test_config_is_readonly():
    cfg = Config({"a": 1})
    with pytest.raises(AttributeError):
        cfg.a = 2


def test_missing_key_raises():
    cfg = Config({"a": 1})
    with pytest.raises(AttributeError):
        _ = cfg.missing


def test_configs_dir_exists():
    assert CONFIGS_DIR.is_dir()
    assert (CONFIGS_DIR / "default.yaml").is_file()
