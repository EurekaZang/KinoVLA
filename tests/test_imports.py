"""M0 exit criterion: the package is empty-but-importable, every layer subpackage exists."""

import importlib

import pytest

SUBPACKAGES = ["sim", "monitor", "shield", "tokens", "map", "vla", "data", "eval", "utils"]


def test_package_imports():
    import kino_vla

    assert kino_vla.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    mod = importlib.import_module(f"kino_vla.{name}")
    assert mod.__doc__, f"kino_vla.{name} must document which spec section it implements"
