"""Isaac backend adapter for contact-aware O4 surface adhesion v2."""

from __future__ import annotations

import numpy as np

from kino_vla.sim.adhesion_v2 import (
    SurfaceAdhesionConfig,
    SurfaceAdhesionState,
    SurfaceAdhesionUpdate,
    step_surface_adhesion,
)
from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend


class IsaacPolicyBackendO4V2(IsaacPolicyBackend):
    """Adds peelable stance-cycle adhesion without changing the frozen base backend."""

    def _init_state(self) -> None:
        super()._init_state()
        if isinstance(getattr(self, "_foot_adhesion_cfg", None), SurfaceAdhesionConfig):
            self._foot_adhesion_states = [SurfaceAdhesionState() for _ in self._foot_ids]
            self._foot_adhesion_updates = []
            self._foot_adhesion_contacts_n = [0.0 for _ in self._foot_ids]
            self._surface_adhesion_recovery_mode = False

    def add_foot_adhesion(self, config: object) -> None:
        if not isinstance(config, SurfaceAdhesionConfig):
            super().add_foot_adhesion(config)  # type: ignore[arg-type]
            return
        if self._foot_terrain_cfg is not None:
            raise RuntimeError("surface adhesion and foot terramechanics need a force mixer")
        self._foot_adhesion_cfg = config  # type: ignore[assignment]
        self._foot_adhesion_states = [SurfaceAdhesionState() for _ in self._foot_ids]
        self._foot_adhesion_updates = []
        self._foot_adhesion_contacts_n = [0.0 for _ in self._foot_ids]
        self._surface_adhesion_recovery_mode = False

    def clear_foot_adhesion(self) -> None:
        super().clear_foot_adhesion()
        self._surface_adhesion_recovery_mode = False

    def _apply_foot_adhesion_forces(self, command_world_xy: np.ndarray) -> None:
        config = self._foot_adhesion_cfg
        if not isinstance(config, SurfaceAdhesionConfig):
            super()._apply_foot_adhesion_forces(command_world_xy)
            return
        env_origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        foot_pos = self._robot.data.body_pos_w[0, self._foot_ids].detach().cpu().numpy()
        foot_pos -= env_origin.reshape(1, 3)
        foot_vel = self._robot.data.body_lin_vel_w[0, self._foot_ids].detach().cpu().numpy()
        contact_vec = self._contact.data.net_forces_w[0, self._contact_foot_ids]
        contact_n = self._torch.linalg.norm(contact_vec, dim=1).detach().cpu().numpy()
        commanded_progress = float(
            np.dot(np.asarray(command_world_xy, dtype=np.float64), config.unit_progress_axis_xy)
        )
        if commanded_progress < -0.03:
            self._surface_adhesion_recovery_mode = True
        active = sum(state.attached for state in self._foot_adhesion_states)
        forces = np.zeros((len(self._foot_ids), 3), dtype=np.float32)
        updates: list[SurfaceAdhesionUpdate] = []
        next_states: list[SurfaceAdhesionState] = []
        for index, state in enumerate(self._foot_adhesion_states):
            allow_attach = not self._surface_adhesion_recovery_mode and (
                state.attached or active < config.max_active_feet
            )
            update = step_surface_adhesion(
                config,
                state,
                foot_pos[index],
                foot_vel[index],
                float(contact_n[index]),
                dt_s=float(self.dt),
                allow_attach=allow_attach,
                force_release=self._surface_adhesion_recovery_mode and state.attached,
            )
            if not state.attached and update.state.attached:
                active += 1
            elif state.attached and not update.state.attached:
                active -= 1
            forces[index] = update.force_world_n.astype(np.float32)
            updates.append(update)
            next_states.append(update.state)
        force_tensor = self._torch.from_numpy(forces.reshape(1, len(self._foot_ids), 3)).to(
            self._device
        )
        torque_tensor = self._torch.zeros_like(force_tensor)
        try:
            self._robot.set_external_force_and_torque(
                force_tensor, torque_tensor, body_ids=self._foot_ids, is_global=True
            )
        except TypeError:
            self._robot.set_external_force_and_torque(
                force_tensor, torque_tensor, body_ids=self._foot_ids
            )
        self._foot_adhesion_states = next_states
        self._foot_adhesion_updates = updates
        self._foot_adhesion_contacts_n = [float(value) for value in contact_n]

    def adhesion_telemetry(self) -> dict[str, object]:
        config = self._foot_adhesion_cfg
        if not isinstance(config, SurfaceAdhesionConfig):
            return super().adhesion_telemetry()
        feet = []
        total_applied = 0.0
        for index, state in enumerate(self._foot_adhesion_states):
            applied_n = float(np.linalg.norm(state.force_world_n))
            total_applied += applied_n
            update = (
                self._foot_adhesion_updates[index]
                if index < len(self._foot_adhesion_updates)
                else None
            )
            feet.append(
                {
                    "name": self._foot_names[index],
                    "attached": state.attached,
                    "terminal_release": False,
                    "broken": False,
                    "phase": state.phase,
                    "event": state.last_event,
                    "anchor_xyz": list(state.anchor_xyz) if state.anchor_xyz is not None else None,
                    "contact_force_n": (
                        self._foot_adhesion_contacts_n[index]
                        if index < len(self._foot_adhesion_contacts_n)
                        else 0.0
                    ),
                    "applied_force_n": applied_n,
                    "raw_force_n": (
                        0.0
                        if update is None
                        else float(
                            np.hypot(
                                update.raw_tangential_force_n, update.raw_normal_force_n
                            )
                        )
                    ),
                    "raw_tangential_force_n": (
                        0.0 if update is None else update.raw_tangential_force_n
                    ),
                    "raw_normal_force_n": 0.0 if update is None else update.raw_normal_force_n,
                    "max_force_n": float(
                        np.hypot(state.max_tangential_force_n, state.max_normal_force_n)
                    ),
                    "max_extension_m": state.max_tangential_extension_m,
                    "attachment_count": state.attachment_count,
                    "peel_count": state.peel_count,
                    "tangential_work_j": state.tangential_work_j,
                }
            )
        return {
            "enabled": True,
            "mode": "contact_aware_anisotropic_surface_adhesion_v2",
            "terminal_peel_mode": bool(self._surface_adhesion_recovery_mode),
            "active_feet": sum(state.attached for state in self._foot_adhesion_states),
            "released_feet": sum(state.peel_count > 0 for state in self._foot_adhesion_states),
            "broken_feet": 0,
            "total_applied_force_n": total_applied,
            "total_attachment_cycles": sum(
                state.attachment_count for state in self._foot_adhesion_states
            ),
            "total_peel_cycles": sum(state.peel_count for state in self._foot_adhesion_states),
            "total_tangential_work_j": sum(
                state.tangential_work_j for state in self._foot_adhesion_states
            ),
            "feet": feet,
        }
