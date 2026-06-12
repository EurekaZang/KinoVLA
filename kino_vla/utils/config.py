"""YAML config loading with dotted-key overrides.

Design decision (M0): plain YAML + a thin typed loader instead of Hydra. Isaac Lab
ships its own dataclass config system for envs; a second framework on top of it adds
composition complexity without supporting any spec claim. All thresholds and
tolerances live in ``configs/`` per QA rule 5.1.5 — never as literals in module code.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Repo root resolved relative to this file: kino_vla/utils/config.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


class Config:
    """Read-only view over a nested dict with attribute and dotted-key access."""

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", copy.deepcopy(data))

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError as err:
            raise AttributeError(f"config has no key {key!r}") from err
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Config is read-only; use overrides at load time")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up ``"a.b.c"`` style keys, returning ``default`` when absent."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _apply_override(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise KeyError(f"cannot override through non-mapping at {part!r} in {dotted_key!r}")
    node[parts[-1]] = value


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load a YAML config file, optionally applying ``{"a.b": value}`` overrides.

    Relative paths are resolved against the repo's ``configs/`` directory.
    """
    path = Path(path)
    if not path.is_absolute():
        path = CONFIGS_DIR / path
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"top level of {path} must be a mapping, got {type(data).__name__}")
    for key, value in (overrides or {}).items():
        _apply_override(data, key, value)
    return Config(data)
