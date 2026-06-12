"""Shared pytest fixtures and sim-availability gating."""

from __future__ import annotations

import importlib.util

import pytest

ISAAC_AVAILABLE = importlib.util.find_spec("isaaclab") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip @pytest.mark.sim tests when Isaac Lab is not installed (dev laptop)."""
    if ISAAC_AVAILABLE:
        return
    skip_sim = pytest.mark.skip(reason="isaaclab not installed (CPU-only dev machine)")
    for item in items:
        if item.get_closest_marker("sim") is not None:
            item.add_marker(skip_sim)
