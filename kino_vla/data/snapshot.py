"""PHASE 2 failure interception + multimodal snapshot packaging (spec §10).

The Kino-Monitor intercepts the anomaly instant; this recorder packages the spec's snapshot:
``[5×RGB] + [5×depth] + [500 ms proprioceptive window (the Kino-Tokens precursor)] + [prior
VLA outputs] + [privileged physics truth]``.

A live RGB-D camera renders every step, but rendering thousands of frames per episode just to
keep five is wasteful, so the recorder buffers only the last few *poses* (cheap) and the
proprioceptive window, then renders the five RGB-D frames lazily from those buffered poses at
the interception instant — the genuine "five frames before the anomaly". The render is the
real §7 pinhole RGB-D (``kino_vla.map.rgbd``); a live Isaac camera feed drops in unchanged.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np

from kino_vla.data.schema import Snapshot
from kino_vla.map.rgbd import (
    CameraExtrinsics,
    CameraIntrinsics,
    pixel_to_ground,
    render_ground_scene,
)
from kino_vla.map.types import SemanticRegion
from kino_vla.monitor.event import MonitorEvent
from kino_vla.sim.types import Obs
from kino_vla.tokens.window import RollingWindow, window_length
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import Rect


class SnapshotRecorder:
    """Buffers poses + a proprio window and captures one Snapshot at the first monitor event.

    Capture is gated to the failure locus (``gate_rect``): the bang-bang collection drive
    spikes the tracking-error channel on clean ground too, so without the gate the monitor
    would intercept a decel transient before the robot reaches the hazard. Buffering happens
    every step (so the window is full at the interception); only the capture is gated.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        scene: list[SemanticRegion],
        operator_name: str,
        appearance_class: str,
        privileged_fn: Callable[[], dict[str, float]],
        gate_rect: Rect | None = None,
        live_camera: object | None = None,
    ) -> None:
        snap = cfg.snapshot
        self._n_frames = int(snap.n_frames)
        self._intr = CameraIntrinsics.from_hfov(
            int(snap.image_width), int(snap.image_height), float(snap.fov_rad)
        )
        self._mount_h = float(snap.mount_height_m)
        self._pitch = float(snap.pitch_rad)
        self._scene = list(scene)
        self._operator_name = operator_name
        self._appearance_class = appearance_class
        self._gate_rect = gate_rect
        # When set, EVERY snapshot's RGB and the waypoint back-projection come from this live Isaac
        # RTX camera (kino_vla.sim.live_camera.LiveRtxCamera) — the deployed VLA sees real rendered
        # pixels, not a CPU render of the scene (user directive 2026-06-21). None = the procedural
        # render path, kept ONLY for the camera-free CI / M6 dataset build.
        self._live_camera = live_camera
        # The privileged θ truth, sampled at the robot's current state at the interception.
        self._privileged_fn = privileged_fn
        self._poses: deque[tuple[np.ndarray, float]] = deque(maxlen=self._n_frames)
        self._window = RollingWindow(window_length(float(snap.window_ms), float(snap.control_hz)))
        self._prior_outputs: list[str] = []
        self.snapshot: Snapshot | None = None

    def observe(self, obs: Obs, event: MonitorEvent | None) -> None:
        """Buffer this control step; capture the snapshot on the first monitor event in-locus."""
        self._poses.append((obs.pos.copy(), float(obs.heading)))
        self._window.push(obs)
        if event is not None and self.snapshot is None and self._in_locus(obs.pos):
            self.snapshot = self._capture(event)

    def buffer(self, obs: Obs) -> None:
        """Buffer one control step WITHOUT capturing (the runtime closed-loop planner path).

        The deployed VLA planner (M7) re-snapshots on *every* monitor event (multi-round
        reflection, spec §10), so it buffers each step here and calls :meth:`capture` on demand,
        instead of the M6 pipeline's one-shot in-locus capture."""
        self._poses.append((obs.pos.copy(), float(obs.heading)))
        self._window.push(obs)

    def world_from_pixel(
        self, pose_xy: np.ndarray, heading: float, u_frac: float, v_frac: float
    ) -> np.ndarray | None:
        """Back-project a NORMALISED image pixel (u_frac, v_frac ∈ [0,1]) to a ground-plane world
        (x, y), using the SAME body camera the snapshot RGB came from — so the VLA's Replan_Waypoint
        pixel becomes a consistent next nav waypoint (spec §7). With a live RTX camera, that is the
        REAL camera's intrinsics + pose (the frame the VLA just saw); else the procedural one."""
        if self._live_camera is not None:
            return self._live_camera.world_from_pixel(float(u_frac), float(v_frac))
        extr = CameraExtrinsics.look(
            np.asarray(pose_xy, dtype=np.float64), float(heading), self._mount_h, self._pitch
        )
        return pixel_to_ground(
            self._intr, extr, float(u_frac) * self._intr.width, float(v_frac) * self._intr.height
        )

    def capture(self, event: MonitorEvent, *, prior_outputs: list[str] | None = None) -> Snapshot:
        """Build a snapshot from the current buffer at ``event`` (public, multi-round capable)."""
        if prior_outputs is not None:
            self._prior_outputs = list(prior_outputs)
        return self._capture(event)

    def _in_locus(self, pos: np.ndarray) -> bool:
        """True if the robot is at the failure locus (inside the hazard gate, if any)."""
        return self._gate_rect is None or self._gate_rect.contains(pos)

    def _capture(self, event: MonitorEvent) -> Snapshot:
        poses = list(self._poses)
        if not poses:  # an event on the very first step (no buffered pose yet)
            poses = [(event.pos.copy(), 0.0)]
        # Left-pad with the oldest available pose so the frame count is always n_frames.
        while len(poses) < self._n_frames:
            poses.insert(0, poses[0])
        if self._live_camera is not None:
            # REAL RTX path: the VLA sees the live camera frame (one render, replicated to n_frames;
            # the model reads only the last n_images). Back-projection later reuses the same frame.
            rgb1 = self._live_camera.snapshot_rgb()
            if rgb1 is None:
                raise RuntimeError("live RTX camera returned no frame (perception camera off?)")
            rgb = np.stack([rgb1] * self._n_frames, axis=0)
            depth = np.zeros((self._n_frames, rgb1.shape[0], rgb1.shape[1]), dtype=np.float64)
        else:
            rgb_frames: list[np.ndarray] = []
            depth_frames: list[np.ndarray] = []
            for pose_xy, heading in poses[-self._n_frames :]:
                extr = CameraExtrinsics.look(pose_xy, heading, self._mount_h, self._pitch)
                frame = render_ground_scene(self._intr, extr, self._scene)
                rgb_frames.append(frame.rgb)
                depth_frames.append(frame.range_m)
            rgb = np.stack(rgb_frames, axis=0)
            depth = np.stack(depth_frames, axis=0)
        return Snapshot(
            operator_name=self._operator_name,
            appearance_class=self._appearance_class,
            t=float(event.t),
            pose_xy=event.pos.copy(),
            heading=poses[-1][1],
            rgb=rgb,
            depth=depth,
            proprio_window=self._window.window(),
            prior_outputs=list(self._prior_outputs),
            privileged_theta=dict(self._privileged_fn()),
            monitor_channel=event.channel,
        )
