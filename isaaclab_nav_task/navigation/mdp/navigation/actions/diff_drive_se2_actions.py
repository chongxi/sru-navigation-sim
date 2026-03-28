# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Differential-drive SE2 action term with PID yaw hold.

This action term takes 2D policy outputs [vx, yaw_target_rate] and converts them to
wheel velocity targets via:
1. Ramp-limited command shaping on forward speed and yaw-target rate
2. Persistent yaw-target integration from ramped yaw-target rate
3. PID controller at physics frequency for yaw tracking
4. Differential drive kinematics to wheel velocities

The PID yaw hold compensates for the xlerobot's physical asymmetry that causes
yaw drift during straight-line driving.

        The architecture mirrors the fly brain's navigation circuit:
        - Policy (fan-shaped body) outputs how to move and how to slew target yaw
        - PID (motor circuits) handles HOW to get there (wheel velocities)
        - This naturally accommodates the phasor network in Phase 3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from .diff_drive_se2_actions_cfg import DiffDriveNavigationSE2ActionCfg


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))


class DiffDriveNavigationSE2Action(ActionTerm):
    """Direct SE2 velocity control for differential-drive robots with PID yaw hold.

        The policy outputs 2D actions [vx, yaw_target_rate]. Internally, 3D buffers
        are maintained for compatibility with existing event functions that assume
        3-dimensional velocity commands (vx, vy, omega).

        Control flow:
        process_actions (policy rate): policy [vx, yaw_target_rate] -> scale/store targets
        apply_actions (physics rate): integrate yaw_target -> PID -> diff kinematics -> wheels
    """

    cfg: DiffDriveNavigationSE2ActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: DiffDriveNavigationSE2ActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Create wheel velocity action term
        self._wheel_action_term: ActionTerm = cfg.wheel_velocity_action.class_type(
            cfg.wheel_velocity_action, env
        )

        # Action dimensions
        self._action_dim = 2  # policy sees [vx, yaw_target_rate]
        self._internal_dim = 3  # internal [vx, vy=0, yaw_target_rate] for event compat

        # Kinematic parameters
        self._wheel_radius = cfg.wheel_radius
        self._half_track = cfg.wheel_track / 2.0
        self._max_wheel_vel = cfg.max_wheel_velocity

        # PID parameters
        self._kp = cfg.yaw_pid_kp
        self._ki = cfg.yaw_pid_ki
        self._kd = cfg.yaw_pid_kd
        self._integral_limit = cfg.yaw_pid_integral_limit
        self._max_correction = cfg.yaw_hold_max_correction
        self._yaw_rate_filter_time = cfg.yaw_rate_filter_time

        # Physics timing
        self._physics_dt = env.physics_dt

        # Find wheel joint IDs for direct velocity target setting
        self._wheel_joint_ids, _ = self._asset.find_joints(
            ["wheel_left_joint", "wheel_right_joint"], preserve_order=True
        )

        # Initialize all buffers
        self._init_buffers()

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Policy action dimension (2D: vx, yaw_target_rate)."""
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw navigation actions (3D internal buffer)."""
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed navigation actions after scaling and filtering (3D internal buffer)."""
        return self._processed_navigation_velocity_actions

    @property
    def filtered_velocity_commands(self) -> torch.Tensor:
        """Current ramp-limited velocity commands."""
        return self._ramped_navigation_velocity_commands

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        """Process policy actions at policy frequency (e.g., 10Hz).

        Maps 2D policy output [vx, yaw_target_rate] to 3D internal representation.
        The second channel slews the persistent yaw target in world frame. Zero keeps
        the current target fixed so the PID can continue converging even at vx=0.

        Args:
            actions: Policy actions of shape [num_envs, 2].
        """
        # 1. Map 2D -> 3D: [vx, yaw_target_rate] -> [vx, 0, yaw_target_rate]
        self._raw_navigation_velocity_actions[:, 0] = actions[:, 0]  # vx
        self._raw_navigation_velocity_actions[:, 1] = 0.0  # vy always zero
        self._raw_navigation_velocity_actions[:, 2] = actions[:, 1]  # yaw_target_rate
        self._debug_last_policy_actions[:, :] = actions

        # 2. Apply affine transform or use raw
        if not self.cfg.use_raw_actions:
            self._processed_navigation_velocity_actions[:] = (
                self._raw_navigation_velocity_actions * self._scale + self._offset
            )
        else:
            self._processed_navigation_velocity_actions[:] = self._raw_navigation_velocity_actions

        # 3. Apply distribution-dependent squashing
        if self.cfg.policy_distr_type == "gaussian":
            self._processed_navigation_velocity_actions = torch.tanh(
                self._processed_navigation_velocity_actions
            )
        elif self.cfg.policy_distr_type == "beta":
            self._processed_navigation_velocity_actions = (
                self._processed_navigation_velocity_actions - 0.5
            ) * 2.0
        else:
            raise ValueError(f"Unknown policy distribution type: {self.cfg.policy_distr_type}")

        # 4. Apply policy scaling and bias (3D, compatible with randomize_action_scale event)
        self._processed_navigation_velocity_actions = (
            self._processed_navigation_velocity_actions * self._policy_scaling
            + self._policy_bias * 0.0  # bias unused, kept for event compat
        )

        # 5. Store desired physical commands. These are ramped at physics rate in apply_actions().
        self._target_vx[:] = self._processed_navigation_velocity_actions[:, 0]
        self._target_yaw_target_rate[:] = self._processed_navigation_velocity_actions[:, 2]
        self._debug_apply_count_since_process = 0

    @torch.inference_mode()
    def apply_actions(self):
        """Apply actions at physics frequency.

        Runs the PID yaw controller and computes wheel velocity targets via
        differential drive kinematics.
        """
        # 1. Read current yaw from robot
        _, _, current_yaw = euler_xyz_from_quat(self._asset.data.root_quat_w)
        self._debug_apply_count_since_process += 1

        # 1b. Deferred yaw_target initialization: capture actual yaw after
        #     physics has stepped (avoids stale data during reset sequence)
        needs_init = self._yaw_target_needs_init
        if needs_init.any():
            self._yaw_target[needs_init] = current_yaw[needs_init]
            self._yaw_target_needs_init[needs_init] = False

        # 1c. Ramp the held commands toward the latest policy targets at physics rate.
        self._stored_vx[:] = self._ramp_command(self._stored_vx, self._target_vx)
        self._stored_yaw_target_rate[:] = self._ramp_command(
            self._stored_yaw_target_rate, self._target_yaw_target_rate
        )
        self._ramped_navigation_velocity_commands[:, 0] = self._stored_vx
        self._ramped_navigation_velocity_commands[:, 1] = 0.0
        self._ramped_navigation_velocity_commands[:, 2] = self._stored_yaw_target_rate

        # 1d. Integrate the persistent yaw target at physics frequency.
        self._yaw_target[:] = _wrap_to_pi(
            self._yaw_target + self._stored_yaw_target_rate * self._physics_dt
        )

        # 2. Read and filter yaw rate
        raw_yaw_rate = self._asset.data.root_ang_vel_b[:, 2]
        if self._yaw_rate_filter_time > 0.0:
            alpha = min(1.0, self._physics_dt / self._yaw_rate_filter_time)
            self._filtered_yaw_rate += alpha * (raw_yaw_rate - self._filtered_yaw_rate)
        else:
            self._filtered_yaw_rate[:] = raw_yaw_rate

        # 3. PID computation (vectorized over all envs)
        yaw_error = _wrap_to_pi(self._yaw_target - current_yaw)

        self._yaw_error_integral += yaw_error * self._physics_dt
        self._yaw_error_integral.clamp_(
            -self._integral_limit, self._integral_limit
        )

        correction = (
            self._kp * yaw_error
            + self._ki * self._yaw_error_integral
            - self._kd * self._filtered_yaw_rate
        )
        correction.clamp_(-self._max_correction, self._max_correction)

        # 4. Differential drive kinematics.
        # Unlike the forward-only controller, we intentionally allow correction
        # when vx=0 so the base can pivot in place and converge to yaw_target.
        left_vel = (self._stored_vx - correction * self._half_track) / self._wheel_radius
        right_vel = (self._stored_vx + correction * self._half_track) / self._wheel_radius

        left_vel.clamp_(-self._max_wheel_vel, self._max_wheel_vel)
        right_vel.clamp_(-self._max_wheel_vel, self._max_wheel_vel)

        self._debug_current_yaw[:] = current_yaw
        self._debug_raw_yaw_rate[:] = raw_yaw_rate
        self._debug_filtered_yaw_rate[:] = self._filtered_yaw_rate
        self._debug_yaw_error[:] = yaw_error
        self._debug_correction[:] = correction
        self._debug_left_wheel_velocity[:] = left_vel
        self._debug_right_wheel_velocity[:] = right_vel

        # 5. Set wheel velocity targets
        wheel_targets = torch.stack([left_vel, right_vel], dim=-1)
        self._asset.set_joint_velocity_target(wheel_targets, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset PID state and filter buffers for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Mark yaw_target as needing re-capture on next apply_actions.
        # We can't read root_quat_w here because Isaac Lab's reset sequence
        # may not have stepped physics yet after reset_base event — the pose
        # data would be stale (pre-reset yaw). Instead, flag these envs so
        # apply_actions captures the correct yaw on its first call.
        self._yaw_target_needs_init[env_ids] = True

        # Reset PID state
        self._yaw_error_integral[env_ids] = 0.0
        self._filtered_yaw_rate[env_ids] = 0.0

        # Reset command buffers
        self._stored_vx[env_ids] = 0.0
        self._stored_yaw_target_rate[env_ids] = 0.0
        self._target_vx[env_ids] = 0.0
        self._target_yaw_target_rate[env_ids] = 0.0
        self._ramped_navigation_velocity_commands[env_ids] = 0.0
        self._raw_navigation_velocity_actions[env_ids] = 0.0
        self._processed_navigation_velocity_actions[env_ids] = 0.0
        self._debug_last_policy_actions[env_ids] = 0.0
        self._debug_current_yaw[env_ids] = 0.0
        self._debug_raw_yaw_rate[env_ids] = 0.0
        self._debug_filtered_yaw_rate[env_ids] = 0.0
        self._debug_yaw_error[env_ids] = 0.0
        self._debug_correction[env_ids] = 0.0
        self._debug_left_wheel_velocity[env_ids] = 0.0
        self._debug_right_wheel_velocity[env_ids] = 0.0
        self._debug_apply_count_since_process = 0

    def reset_low_pass_filter(self, env_ids: torch.Tensor):
        """Reset ramped-command state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._ramped_navigation_velocity_commands[env_ids] = 0.0

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _ramp_command(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Ramp commands toward their targets using separate up/down time constants."""
        if self._command_ramp_up_time <= 0.0 and self._command_ramp_down_time <= 0.0:
            return target

        increasing = torch.abs(target) > torch.abs(current)
        ramp_time = torch.full_like(current, self._command_ramp_down_time)
        ramp_time[increasing] = self._command_ramp_up_time

        alpha = torch.ones_like(current)
        positive_ramp = ramp_time > 0.0
        alpha[positive_ramp] = torch.clamp(
            torch.full_like(current[positive_ramp], self._physics_dt) / ramp_time[positive_ramp],
            max=1.0,
        )
        return current + alpha * (target - current)

    def _init_buffers(self):
        """Initialize all internal buffers."""
        N = self.num_envs
        D = self._internal_dim  # 3

        # 3D internal buffers for event function compatibility
        self._raw_navigation_velocity_actions = torch.zeros(N, D, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros(N, D, device=self.device)
        self._ramped_navigation_velocity_commands = torch.zeros(N, D, device=self.device)

        # Policy scaling and bias (3D, written by randomize_action_scale event)
        self._policy_scaling = torch.tensor(
            self.cfg.policy_scaling, device=self.device
        ).unsqueeze(0).expand(N, -1).clone()
        self._policy_bias = torch.zeros(N, D, device=self.device)

        # Scale and offset
        self._scale = torch.tensor(self.cfg.scale, device=self.device)
        self._offset = torch.tensor(self.cfg.offset, device=self.device)

        # Command ramp configuration
        self._command_ramp_up_time = float(self.cfg.command_ramp_up_time)
        self._command_ramp_down_time = float(self.cfg.command_ramp_down_time)

        # PID state buffers (per-env)
        self._yaw_target = torch.zeros(N, device=self.device)
        self._yaw_target_needs_init = torch.ones(N, dtype=torch.bool, device=self.device)
        self._yaw_error_integral = torch.zeros(N, device=self.device)
        self._filtered_yaw_rate = torch.zeros(N, device=self.device)
        self._target_vx = torch.zeros(N, device=self.device)
        self._target_yaw_target_rate = torch.zeros(N, device=self.device)
        self._stored_vx = torch.zeros(N, device=self.device)
        self._stored_yaw_target_rate = torch.zeros(N, device=self.device)

        # Debug state from the most recent physics sub-step.
        self._debug_last_policy_actions = torch.zeros(N, self._action_dim, device=self.device)
        self._debug_current_yaw = torch.zeros(N, device=self.device)
        self._debug_raw_yaw_rate = torch.zeros(N, device=self.device)
        self._debug_filtered_yaw_rate = torch.zeros(N, device=self.device)
        self._debug_yaw_error = torch.zeros(N, device=self.device)
        self._debug_correction = torch.zeros(N, device=self.device)
        self._debug_left_wheel_velocity = torch.zeros(N, device=self.device)
        self._debug_right_wheel_velocity = torch.zeros(N, device=self.device)
        self._debug_apply_count_since_process = 0

        # Curriculum compatibility (checked by disable_backward_penalty_after_steps)
        self.disable_backward_penalty = torch.zeros(N, dtype=torch.bool, device=self.device)
