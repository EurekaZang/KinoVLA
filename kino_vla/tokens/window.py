"""500 ms sliding-window logger (spec §4 backbone, M4 scope).

Two access paths over the same fixed-length window of proprioceptive features:

- ``RollingWindow``: an online ring buffer fed one ``Obs`` per control step; used by
  the live μ̂→shield coupler (``kino_vla.tokens.coupler``). ``ready`` once it holds a
  full window; ``window()`` returns the ``(T, F)`` array (oldest → newest).
- ``slice_rollout_to_windows``: offline, turns a logged episode (per-step features +
  privileged targets) into ``(window, target)`` training pairs, the target taken at
  each window's *last* step so time-/location-varying physics is supervised in place.

Pure numpy — torch lives only in the extractor model.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from kino_vla.sim.types import Obs
from kino_vla.tokens.features import N_FEATURES, obs_to_features


def window_length(window_ms: float, control_hz: float) -> int:
    """Number of control steps in a ``window_ms`` window at ``control_hz`` (≥ 1)."""
    return max(1, int(round(window_ms * 1e-3 * control_hz)))


class RollingWindow:
    """Online ``(T, F)`` ring buffer of the most recent ``length`` feature vectors."""

    def __init__(self, length: int) -> None:
        if length < 1:
            raise ValueError(f"window length must be >= 1, got {length}")
        self._length = int(length)
        self._buf: deque[np.ndarray] = deque(maxlen=self._length)

    @property
    def length(self) -> int:
        return self._length

    @property
    def ready(self) -> bool:
        """True once a full window has accumulated (before that, μ̂ is not emitted)."""
        return len(self._buf) == self._length

    def reset(self) -> None:
        self._buf.clear()

    def push(self, obs: Obs) -> None:
        self._buf.append(obs_to_features(obs))

    def window(self) -> np.ndarray:
        """The current ``(length, N_FEATURES)`` window, oldest row first.

        Before ``ready`` the buffer is left-padded with its earliest sample so the
        shape is always fixed (the coupler gates on ``ready`` regardless).
        """
        if not self._buf:
            return np.zeros((self._length, N_FEATURES), dtype=np.float64)
        rows = list(self._buf)
        if len(rows) < self._length:
            rows = [rows[0]] * (self._length - len(rows)) + rows
        return np.stack(rows, axis=0)


def slice_rollout_to_windows(
    features: np.ndarray,
    targets: np.ndarray,
    length: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a per-step rollout into overlapping windows and last-step targets.

    ``features`` is ``(S, F)`` and ``targets`` ``(S, T_dim)`` for an ``S``-step
    episode. Returns ``(windows, win_targets)`` of shape ``(N, length, F)`` and
    ``(N, T_dim)``; empty arrays if the rollout is shorter than one window.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n_steps = features.shape[0]
    if n_steps < length:
        return (
            np.empty((0, length, features.shape[1]), dtype=np.float64),
            np.empty((0, targets.shape[1]), dtype=np.float64),
        )
    starts = range(0, n_steps - length + 1, max(1, stride))
    wins = np.stack([features[s : s + length] for s in starts], axis=0)
    tgts = np.stack([targets[s + length - 1] for s in starts], axis=0)
    return wins, tgts
