"""Live Isaac RTX body camera → the VLA's eyes (every inference reads REAL rendered pixels).

The deployed VLA planner must SEE through the same real camera the robot carries: its snapshot RGB
and the back-projection of its ``Replan_Waypoint`` / ``Turn`` picks both come from the live Isaac
RTX perception camera, NOT a CPU render of the scene. This wraps :class:`IsaacPolicyBackend`'s
robot-following RGB-D camera (``capture_perception``):

- :meth:`snapshot_rgb` returns the live RTX frame (float [0,1]) for the VLA's image input;
- :meth:`world_from_pixel` back-projects a normalised pixel to the ground plane using the camera's
  REAL intrinsics (``cap["K"]``) and look-at pose (``cap["eye"]``/``cap["target"]``) — the same
  ray∩ground geometry the §7 :class:`LiveRtxSegmenter` map uses;
- :meth:`pixel_from_world` is the inverse (project a world point to a normalised pixel), used to
  label a geometric next-waypoint as the pixel the VLA should emit when building nav SFT data.

All three share ONE captured frame (the camera pose at that instant), so the image the VLA reasons
over and the geometry its pick is grounded in are consistent.
"""

from __future__ import annotations

import math

import numpy as np

from kino_vla.map.rgbd import (
    CameraExtrinsics,
    CameraIntrinsics,
    ground_to_pixel,
    pixel_to_ground,
)


class LiveRtxCamera:
    """Real RTX perception camera as the VLA's image + waypoint-grounding source."""

    def __init__(self, backend: object) -> None:
        self._backend = backend
        self._cap: dict | None = None  # the last captured frame (rgb + K + eye/target)
        self._logged = False  # one-shot diagnostic on the first real frame

    # --------------------------------------------------------------- capture
    def capture(self) -> dict | None:
        """Render + read the live RTX frame; cache it for the projection methods. None if off."""
        cap = self._backend.capture_perception()
        if cap is not None and cap.get("eye") is not None:
            self._cap = cap
            return cap
        return None

    def snapshot_rgb(self) -> np.ndarray | None:
        """The live RTX body-camera frame as (H, W, 3) float in [0, 1] (the VLA's image input).

        Captures a fresh frame (the camera was auto-aimed at the robot this step) and normalises to
        the [0,1] float the model's image path expects (the Isaac camera returns uint8 [0,255])."""
        cap = self.capture()
        if cap is None:
            return None
        rgb = np.asarray(cap["rgb"], dtype=np.float64)[..., :3]
        if float(np.nanmax(rgb)) > 1.5:  # uint8 [0,255] → [0,1]
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        if not self._logged:
            self._logged = True
            print(
                f"[live-rtx] VLA reads REAL RTX frame: shape={rgb.shape} "
                f"range=[{rgb.min():.3f},{rgb.max():.3f}] mean={rgb.mean():.3f} "
                f"eye={np.round(cap['eye'], 2)} K_fx={float(cap['K'][0, 0]):.1f}",
                flush=True,
            )
        return rgb

    # ------------------------------------------------------------ projection
    def _intr_extr(self, cap: dict) -> tuple[CameraIntrinsics, CameraExtrinsics]:
        """Real pinhole intrinsics + look-at extrinsics for a captured frame (start/odom frame)."""
        h, w = cap["rgb"].shape[:2]
        eye = np.asarray(cap["eye"], dtype=np.float64)
        target = np.asarray(cap["target"], dtype=np.float64)
        fwd = target - eye
        heading = float(math.atan2(fwd[1], fwd[0]))
        pitch = float(math.atan2(-fwd[2], float(np.hypot(fwd[0], fwd[1]))))
        k = cap["K"]
        intr = CameraIntrinsics(
            width=int(w),
            height=int(h),
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
        )
        extr = CameraExtrinsics.look(eye[:2], heading, float(eye[2]), pitch)
        return intr, extr

    def world_from_pixel(self, u_frac: float, v_frac: float) -> np.ndarray | None:
        """Back-project a NORMALISED pixel (u_frac, v_frac ∈ [0,1]) to the ground plane, using the
        LAST captured frame's real intrinsics + pose — the camera the VLA just looked through."""
        if self._cap is None:
            return None
        intr, extr = self._intr_extr(self._cap)
        return pixel_to_ground(intr, extr, float(u_frac) * intr.width, float(v_frac) * intr.height)

    def pixel_from_world(self, world_xy: np.ndarray, cap: dict | None = None) -> np.ndarray | None:
        """Project a world ground point to a NORMALISED pixel (u_frac, v_frac ∈ [0,1]) in the given
        (or last) captured frame, or None if it is behind the camera / out of frame (nav labels)."""
        cap = cap if cap is not None else self._cap
        if cap is None:
            return None
        intr, extr = self._intr_extr(cap)
        uv = ground_to_pixel(intr, extr, np.asarray(world_xy, dtype=np.float64))
        if uv is None:
            return None
        return np.array([uv[0] / intr.width, uv[1] / intr.height], dtype=np.float64)
