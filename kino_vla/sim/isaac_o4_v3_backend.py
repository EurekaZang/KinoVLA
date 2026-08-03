"""Isaac adapter for command-agnostic O4 surface adhesion v3."""

from __future__ import annotations

import numpy as np

from kino_vla.sim.adhesion_v3 import (
    SurfaceAdhesionConfig,
    SurfaceAdhesionState,
    SurfaceAdhesionUpdate,
    step_surface_adhesion,
)
from kino_vla.sim.isaac_o4_v2_backend import IsaacPolicyBackendO4V2


class IsaacPolicyBackendO4V3(IsaacPolicyBackendO4V2):
    """Removes command-conditioned release from the v2 Isaac adapter.

    The action can change the robot trajectory, contact loads and foot lift, but it cannot
    directly switch adhesion off.  This makes action-consequence differences physical rather
    than an implementation shortcut.
    """

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
        active = sum(state.attached for state in self._foot_adhesion_states)
        forces = np.zeros((len(self._foot_ids), 3), dtype=np.float32)
        updates: list[SurfaceAdhesionUpdate] = []
        next_states: list[SurfaceAdhesionState] = []
        for index, state in enumerate(self._foot_adhesion_states):
            allow_attach = state.attached or active < config.max_active_feet
            update = step_surface_adhesion(
                config,
                state,
                foot_pos[index],
                foot_vel[index],
                float(contact_n[index]),
                dt_s=float(self.dt),
                allow_attach=allow_attach,
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
        telemetry = super().adhesion_telemetry()
        if isinstance(self._foot_adhesion_cfg, SurfaceAdhesionConfig):
            telemetry["mode"] = "command_agnostic_contact_surface_adhesion_v3"
            telemetry.pop("terminal_peel_mode", None)
            telemetry["command_conditioned_release"] = False
        return telemetry
