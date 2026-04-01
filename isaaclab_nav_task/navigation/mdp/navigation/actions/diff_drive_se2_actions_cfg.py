# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for the differential-drive SE2 action term with PID yaw hold."""

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .diff_drive_se2_actions import DiffDriveNavigationSE2Action


@configclass
class DiffDriveNavigationSE2ActionCfg(ActionTermCfg):
    """Configuration for differential-drive navigation with PID yaw hold.

    The policy outputs 2D actions [vx, yaw_target_rate]:
    - vx: desired forward speed
    - yaw_target_rate: rate command that slews the persistent yaw target

    Internally, 3D buffers [vx, vy=0, yaw_target_rate] are used for compatibility
    with existing event functions that hardcode 3D indexing.

    A PID controller runs at physics frequency to track the yaw target,
    compensating for the robot's physical asymmetry that causes yaw drift.
    """

    class_type: type[ActionTerm] = DiffDriveNavigationSE2Action
    """Class of the action term."""

    wheel_velocity_action: ActionTermCfg = MISSING
    """Configuration for the wheel joint velocity action term."""

    # --- Action scaling ---
    # 3D for event function compatibility: [vx, vy(ignored), yaw_target_rate]
    # After tanh [-1,1], multiplied by policy_scaling to get physical units.
    scale: list[float] = [1.0, 1.0, 1.0]
    """Scale for the raw actions [vx, vy, heading]."""

    offset: list[float] = [0.0, 0.0, 0.0]
    """Offset for the raw actions [vx, vy, heading]."""

    policy_scaling: list[float] = [2.5, 0.0, 3.0]
    """Policy-dependent scaling: [vx in m/s, vy ignored, yaw-target-rate in rad/s]."""

    use_raw_actions: bool = True
    """Whether to use raw actions (skip affine transform)."""

    policy_distr_type: str = "gaussian"
    """Policy distribution type: 'gaussian' (tanh squash) or 'beta'."""

    # --- Command ramping ---
    command_ramp_up_time: float = 0.8
    """Time constant for ramping command magnitude up toward the target (seconds)."""

    command_ramp_down_time: float = 0.15
    """Time constant for ramping command magnitude down toward the target (seconds)."""

    # --- Differential drive kinematics ---
    wheel_radius: float = 0.08
    """Wheel radius in meters."""

    wheel_track: float = 0.56
    """Distance between left and right wheel centers in meters."""

    max_wheel_velocity: float = 25.0
    """Maximum wheel angular velocity in rad/s (25 * 0.08m = 2.0 m/s max)."""

    # --- PID yaw hold ---
    # Gains from tuned keyboard script (xlerobot_keyboard_drive_yaw_pid.py)
    yaw_pid_kp: float = 40.0
    """Proportional gain for yaw PID controller."""

    yaw_pid_ki: float = 0.0
    """Integral gain for yaw PID controller."""

    yaw_pid_kd: float = 3.0
    """Derivative gain for yaw PID controller."""

    yaw_pid_integral_limit: float = 1.5
    """Clamp for the integrated yaw error term."""

    yaw_hold_max_correction: float = 5.0
    """Maximum yaw rate correction from PID in rad/s."""

    yaw_rate_filter_time: float = 0.3
    """Low-pass filter time constant for measured yaw rate (seconds). 0 = no filtering."""
