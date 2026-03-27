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

    The policy outputs 2D actions [vx, heading_offset]:
    - vx: desired forward speed
    - heading_offset: desired heading change relative to current yaw

    Internally, 3D buffers [vx, vy=0, heading_offset] are used for compatibility
    with existing event functions (randomize_action_scale, randomize_low_pass_filter_alpha)
    that hardcode 3D indexing.

    A PID controller runs at physics frequency (200Hz) to track the yaw target,
    compensating for the robot's physical asymmetry that causes yaw drift.
    """

    class_type: type[ActionTerm] = DiffDriveNavigationSE2Action
    """Class of the action term."""

    wheel_velocity_action: ActionTermCfg = MISSING
    """Configuration for the wheel joint velocity action term."""

    # --- Action scaling ---
    # 3D for event function compatibility: [vx, vy(ignored), heading_offset]
    # After tanh [-1,1], multiplied by policy_scaling to get physical units.
    scale: list[float] = [1.0, 1.0, 1.0]
    """Scale for the raw actions [vx, vy, heading]."""

    offset: list[float] = [0.0, 0.0, 0.0]
    """Offset for the raw actions [vx, vy, heading]."""

    policy_scaling: list[float] = [2.5, 1.0, math.pi]
    """Policy-dependent scaling: [vx in m/s, vy ignored, heading in radians].
    vx=2.5 matches the keyboard PID script's linear_speed, ensuring the PID
    yaw correction (kp=40) doesn't dominate forward velocity."""

    use_raw_actions: bool = True
    """Whether to use raw actions (skip affine transform)."""

    policy_distr_type: str = "gaussian"
    """Policy distribution type: 'gaussian' (tanh squash) or 'beta'."""

    # --- Low-pass filter ---
    enable_low_pass_filter: bool = True
    """Whether to enable low-pass filtering for smoothing."""

    low_pass_filter_alpha: float = 0.5
    """Default smoothing factor (0=no smoothing, 1=max smoothing)."""

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

    yaw_hold_engage_speed: float = 0.10
    """Minimum |vx| (m/s) to engage yaw PID. Below this, correction is zero."""

    yaw_rate_filter_time: float = 0.3
    """Low-pass filter time constant for measured yaw rate (seconds). 0 = no filtering."""
