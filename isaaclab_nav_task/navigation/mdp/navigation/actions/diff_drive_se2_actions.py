# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Differential-drive SE2 action term with PID yaw hold.

This action term takes 2D policy outputs [vx, heading_offset] and converts them to
wheel velocity targets via:
1. Low-pass filtering (for smooth commands)
2. Yaw target computation (current_yaw + heading_offset)
3. PID controller at physics frequency (200Hz) for yaw tracking
4. Differential drive kinematics to wheel velocities

The PID yaw hold compensates for the xlerobot's physical asymmetry that causes
yaw drift during straight-line driving.

The architecture mirrors the fly brain's navigation circuit:
- Policy (fan-shaped body) outputs WHERE to face (heading_offset)
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

    The policy outputs 2D actions [vx, heading_offset]. Internally, 3D buffers
    are maintained for compatibility with existing event functions that assume
    3-dimensional velocity commands (vx, vy, omega).

    Control flow:
        process_actions (20Hz): policy [vx, heading_offset] -> scale/filter -> yaw_target
        apply_actions (200Hz): PID(yaw_target, current_yaw) -> diff kinematics -> wheels
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
        self._action_dim = 2  # policy sees [vx, heading_offset]
        self._internal_dim = 3  # internal [vx, vy=0, heading_offset] for event compat

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
        self._yaw_hold_engage_speed = cfg.yaw_hold_engage_speed
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
        """Policy action dimension (2D: vx, heading_offset)."""
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
        """Current filtered velocity commands."""
        return self._prev_filtered_velocity_commands

    @property
    def low_pass_alpha_values(self) -> torch.Tensor:
        """Per-env per-dim low-pass filter alpha values [num_envs, 3]."""
        return self._per_env_per_dim_low_pass_alpha

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        """Process policy actions at policy frequency (e.g., 10Hz).

        Maps 2D policy output [vx, heading_relative] to 3D internal representation.
        heading_relative is relative to current heading: yaw_target = current_yaw + output.
        Network outputs 0 → go straight. Outputs +0.5 → turn ~90° right.

        Args:
            actions: Policy actions of shape [num_envs, 2].
        """
        # 1. Map 2D -> 3D: [vx, heading_relative] -> [vx, 0, heading_relative]
        self._raw_navigation_velocity_actions[:, 0] = actions[:, 0]  # vx
        self._raw_navigation_velocity_actions[:, 1] = 0.0  # vy always zero
        self._raw_navigation_velocity_actions[:, 2] = actions[:, 1]  # heading_relative

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

        # 5. Apply low-pass filter to vx only
        vx_filtered = (
            self._per_env_per_dim_low_pass_alpha[:, 0] * self._prev_filtered_velocity_commands[:, 0]
            + (1.0 - self._per_env_per_dim_low_pass_alpha[:, 0]) * self._processed_navigation_velocity_actions[:, 0]
        )
        self._prev_filtered_velocity_commands[:, 0] = vx_filtered
        self._stored_vx[:] = vx_filtered

        # 6. Set yaw_target relative to current heading (no accumulation, no drift)
        #    Network output 0 → go straight. Output ±1 → turn ±π from current heading.
        _, _, current_yaw = euler_xyz_from_quat(self._asset.data.root_quat_w)
        heading_relative = self._processed_navigation_velocity_actions[:, 2]
        self._yaw_target[:] = _wrap_to_pi(current_yaw + heading_relative)

    @torch.inference_mode()
    def apply_actions(self):
        """Apply actions at physics frequency (200Hz).

        Runs the PID yaw controller and computes wheel velocity targets via
        differential drive kinematics.
        """
        # 1. Read current yaw from robot
        _, _, current_yaw = euler_xyz_from_quat(self._asset.data.root_quat_w)

        # 1b. Deferred yaw_target initialization: capture actual yaw after
        #     physics has stepped (avoids stale data during reset sequence)
        needs_init = self._yaw_target_needs_init
        if needs_init.any():
            self._yaw_target[needs_init] = current_yaw[needs_init]
            self._yaw_target_needs_init[needs_init] = False

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

        # Only engage PID when moving forward (matches drive_terrain_pid.py).
        # Limit correction so the slower wheel stays above 30% of the faster
        # wheel — this prevents the robot from trying to pivot in place,
        # which lifts one wheel and causes loss of traction.
        moving = torch.abs(self._stored_vx) > self._yaw_hold_engage_speed
        # At max_omega, the slow wheel = (1-0.7)*vx = 0.3*vx (30% of nominal)
        max_omega = 0.7 * torch.abs(self._stored_vx) / (self._half_track + 1e-6)
        correction = torch.where(moving, correction.clamp(-max_omega, max_omega), torch.zeros_like(correction))

        # 4. Differential drive kinematics
        left_vel = (self._stored_vx - correction * self._half_track) / self._wheel_radius
        right_vel = (self._stored_vx + correction * self._half_track) / self._wheel_radius

        left_vel.clamp_(-self._max_wheel_vel, self._max_wheel_vel)
        right_vel.clamp_(-self._max_wheel_vel, self._max_wheel_vel)

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
        self._prev_filtered_velocity_commands[env_ids] = 0.0
        self._raw_navigation_velocity_actions[env_ids] = 0.0
        self._processed_navigation_velocity_actions[env_ids] = 0.0

    def reset_low_pass_filter(self, env_ids: torch.Tensor):
        """Reset low-pass filter state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._prev_filtered_velocity_commands[env_ids] = 0.0

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _apply_low_pass_filter(self, velocity_commands: torch.Tensor) -> torch.Tensor:
        """Apply exponential smoothing low-pass filter.

        filtered = alpha * prev + (1 - alpha) * new
        Supports per-env per-dim alpha values for event randomization.

        Args:
            velocity_commands: New commands [num_envs, 3].

        Returns:
            Filtered commands [num_envs, 3].
        """
        if not self.cfg.enable_low_pass_filter:
            return velocity_commands

        alpha = self._per_env_per_dim_low_pass_alpha
        filtered = (
            alpha * self._prev_filtered_velocity_commands
            + (1.0 - alpha) * velocity_commands
        )
        self._prev_filtered_velocity_commands.copy_(filtered)
        return filtered

    def _init_buffers(self):
        """Initialize all internal buffers."""
        N = self.num_envs
        D = self._internal_dim  # 3

        # 3D internal buffers for event function compatibility
        self._raw_navigation_velocity_actions = torch.zeros(N, D, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros(N, D, device=self.device)
        self._prev_filtered_velocity_commands = torch.zeros(N, D, device=self.device)

        # Policy scaling and bias (3D, written by randomize_action_scale event)
        self._policy_scaling = torch.tensor(
            self.cfg.policy_scaling, device=self.device
        ).unsqueeze(0).expand(N, -1).clone()
        self._policy_bias = torch.zeros(N, D, device=self.device)

        # Low-pass filter alpha (3D, written by randomize_low_pass_filter_alpha event)
        self._per_env_per_dim_low_pass_alpha = torch.full(
            (N, D), self.cfg.low_pass_filter_alpha, device=self.device
        )

        # Scale and offset
        self._scale = torch.tensor(self.cfg.scale, device=self.device)
        self._offset = torch.tensor(self.cfg.offset, device=self.device)

        # PID state buffers (per-env)
        self._yaw_target = torch.zeros(N, device=self.device)
        self._yaw_target_needs_init = torch.ones(N, dtype=torch.bool, device=self.device)
        self._yaw_error_integral = torch.zeros(N, device=self.device)
        self._filtered_yaw_rate = torch.zeros(N, device=self.device)
        self._stored_vx = torch.zeros(N, device=self.device)

        # Curriculum compatibility (checked by disable_backward_penalty_after_steps)
        self.disable_backward_penalty = torch.zeros(N, dtype=torch.bool, device=self.device)
