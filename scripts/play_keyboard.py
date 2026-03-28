#!/usr/bin/env python3
# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Play a navigation task with keyboard actions instead of a learned policy.

This script uses the exact environment step path as training/play:
keyboard -> env.step(actions) -> action manager -> reward/termination/reset.

For the DiffDrive task, the keyboard replaces the policy's 10 Hz action output.
"""

from __future__ import annotations

import argparse
import math
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a navigation task with keyboard actions.")
parser.add_argument("--task", type=str, required=True, help="Task name, e.g. Isaac-Nav-MDPO-DiffDrive-Play-v0.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--linear_speed", type=float, default=2.5, help="Desired vx command magnitude in m/s.")
parser.add_argument(
    "--yaw_speed",
    type=float,
    default=2.0,
    help="Desired yaw-target slew-rate magnitude in rad/s.",
)
parser.add_argument("--print_every", type=int, default=1, help="Print reward/debug info every N env steps.")
parser.add_argument(
    "--command_ramp_up_time",
    type=float,
    default=None,
    help="Override DiffDrive command ramp-up time in seconds.",
)
parser.add_argument(
    "--command_ramp_down_time",
    type=float,
    default=None,
    help="Override DiffDrive command ramp-down time in seconds.",
)
parser.add_argument("--yaw_pid_kp", type=float, default=None, help="Override DiffDrive yaw PID Kp.")
parser.add_argument("--yaw_pid_ki", type=float, default=None, help="Override DiffDrive yaw PID Ki.")
parser.add_argument("--yaw_pid_kd", type=float, default=None, help="Override DiffDrive yaw PID Kd.")
parser.add_argument(
    "--yaw_pid_integral_limit",
    type=float,
    default=None,
    help="Override DiffDrive yaw PID integral clamp.",
)
parser.add_argument(
    "--yaw_hold_max_correction",
    type=float,
    default=None,
    help="Override DiffDrive max yaw correction in rad/s.",
)
parser.add_argument(
    "--yaw_rate_filter_time",
    type=float,
    default=None,
    help="Override DiffDrive yaw-rate filter time constant in seconds.",
)
parser.add_argument(
    "--debug_controller",
    action="store_true",
    help="Print detailed diff-drive controller state each env step.",
)
parser.add_argument(
    "--disable_action_scale_randomization",
    action="store_true",
    help="Disable reset-time action-scale randomization for easier controller debugging.",
)
parser.add_argument(
    "--disable_low_pass_randomization",
    action="store_true",
    help="Disable reset-time low-pass alpha randomization for easier controller debugging.",
)
parser.add_argument(
    "--disable_friction_randomization",
    action="store_true",
    help="Disable startup physics-material randomization for easier controller debugging.",
)
parser.add_argument(
    "--disable_early_goal_termination",
    action="store_true",
    help="Disable success-based early termination so episodes run to timeout unless another failure occurs.",
)
parser.add_argument(
    "--enable_pose_goal_reward",
    action="store_true",
    help="Enable a constant per-step bonus when the robot is within both XY and yaw goal thresholds.",
)
parser.add_argument(
    "--enable_pose_goal_proximity_reward",
    action="store_true",
    help="Enable a smooth dense reward that increases as the robot approaches the target pose.",
)
parser.add_argument(
    "--pose_goal_reward_weight",
    type=float,
    default=1.0,
    help="Weight for the pose-hold bonus reward when enabled.",
)
parser.add_argument(
    "--pose_goal_proximity_reward_weight",
    type=float,
    default=1.0,
    help="Weight for the dense pose-proximity reward when enabled.",
)
parser.add_argument(
    "--pose_goal_xy_threshold",
    type=float,
    default=0.35,
    help="XY threshold in meters for the pose-hold bonus.",
)
parser.add_argument(
    "--pose_goal_proximity_xy_scale",
    type=float,
    default=1.0,
    help="XY scale in meters for the dense pose-proximity reward.",
)
parser.add_argument(
    "--pose_goal_proximity_activation_xy_threshold",
    type=float,
    default=0.5,
    help="Activate the dense pose-proximity reward only when XY error is below this threshold in meters.",
)
parser.add_argument(
    "--pose_goal_yaw_threshold_deg",
    type=float,
    default=15.0,
    help="Yaw threshold in degrees for the pose-hold bonus.",
)
parser.add_argument(
    "--pose_goal_proximity_yaw_scale_deg",
    type=float,
    default=45.0,
    help="Yaw scale in degrees for the dense pose-proximity reward.",
)
parser.add_argument(
    "--blue_only_arrow",
    action="store_true",
    help="Hide the green DiffDrive current-velocity arrow and keep only the blue heading arrow.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg


class DifferentialDriveKeyboard(Se2Keyboard):
    """Keyboard mapping that mirrors the diff-drive action space."""

    def __str__(self) -> str:
        msg = f"Keyboard Controller for SE(2): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset all commands: L\n"
        msg += "\tMove forward:  Arrow Up\n"
        msg += "\tMove backward: Arrow Down\n"
        msg += "\tYaw left:      Arrow Left\n"
        msg += "\tYaw right:     Arrow Right\n"
        msg += "\tReset env:     R"
        return msg

    def _create_key_bindings(self):
        self._INPUT_KEY_MAPPING = {
            "NUMPAD_8": (self.v_x_sensitivity, 0.0, 0.0),
            "UP": (self.v_x_sensitivity, 0.0, 0.0),
            "NUMPAD_2": (-self.v_x_sensitivity, 0.0, 0.0),
            "DOWN": (-self.v_x_sensitivity, 0.0, 0.0),
            "NUMPAD_4": (0.0, 0.0, self.omega_z_sensitivity),
            "LEFT": (0.0, 0.0, self.omega_z_sensitivity),
            "NUMPAD_6": (0.0, 0.0, -self.omega_z_sensitivity),
            "RIGHT": (0.0, 0.0, -self.omega_z_sensitivity),
        }


def physical_to_raw_policy_action(command: float, scale: float) -> float:
    """Invert tanh-squash scaling so keyboard inputs match env action semantics."""
    if scale <= 0.0:
        return 0.0
    normalized = max(-0.999999, min(0.999999, command / scale))
    return math.atanh(normalized)


def clamp_to_scale(command: float, scale: float) -> tuple[float, bool]:
    """Clamp a physical command to the currently available runtime action scale."""
    if scale <= 0.0:
        return 0.0, abs(command) > 0.0
    clamped = max(-scale, min(scale, command))
    return clamped, abs(clamped - command) > 1e-6


def get_runtime_action_scales(action_term, env_idx: int = 0) -> tuple[float, float]:
    """Return runtime action scales after reset-time randomization."""
    return (
        float(action_term._policy_scaling[env_idx, 0].item()),
        float(action_term._policy_scaling[env_idx, 2].item()),
    )


def format_runtime_controller_config(action_term, env_idx: int = 0) -> str:
    """Summarize runtime action scaling and ramp settings."""
    vx_scale, yaw_scale = get_runtime_action_scales(action_term, env_idx)
    return (
        f"scale(vx={vx_scale:.3f}, yaw_rate={yaw_scale:.3f}) "
        f"ramp(up={action_term._command_ramp_up_time:.3f}, "
        f"down={action_term._command_ramp_down_time:.3f})"
    )


def format_pid_config(action_term) -> str:
    """Summarize the active PID configuration from the action term."""
    return (
        f"kp={action_term._kp:.3f} ki={action_term._ki:.3f} kd={action_term._kd:.3f} "
        f"i_limit={action_term._integral_limit:.3f} "
        f"max_corr={action_term._max_correction:.3f} "
        f"yaw_rate_tau={action_term._yaw_rate_filter_time:.3f}"
    )


def format_reward_terms(env, env_idx: int = 0) -> str:
    terms = env.reward_manager.get_active_iterable_terms(env_idx)
    return " ".join(f"{name}={values[0]:+.3f}" for name, values in terms)


def get_reward_terms_dict(env, env_idx: int = 0) -> dict[str, float]:
    """Return current step reward terms as a dictionary."""
    terms = env.reward_manager.get_active_iterable_terms(env_idx)
    return {name: float(values[0]) for name, values in terms}


def active_termination_terms(env, env_idx: int = 0) -> list[str]:
    terms = env.termination_manager.get_active_iterable_terms(env_idx)
    return [name for name, values in terms if values[0] > 0.5]


def format_episode_log(extras: dict) -> str:
    log = extras.get("log", {})
    if not log:
        return ""
    parts = []
    for key, value in sorted(log.items()):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                continue
        if isinstance(value, float):
            parts.append(f"{key}={value:+.3f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def format_episode_reward_breakdown(episode_reward_terms: dict[str, float]) -> str:
    """Format accumulated per-term reward contributions for the episode."""
    if not episode_reward_terms:
        return ""
    parts = []
    for name, value in sorted(episode_reward_terms.items()):
        parts.append(f"{name}={value:+.3f}")
    return " ".join(parts)


def format_done_diagnostics(goal_cmd_term, contact_sensor=None, env_idx: int = 0) -> str:
    """Format cached pre-reset termination diagnostics."""
    if goal_cmd_term is None:
        return ""
    parts = []
    if hasattr(goal_cmd_term, "last_roll"):
        parts.append(f"roll_deg={math.degrees(goal_cmd_term.last_roll[env_idx].item()):+.1f}")
    if hasattr(goal_cmd_term, "last_pitch"):
        parts.append(f"pitch_deg={math.degrees(goal_cmd_term.last_pitch[env_idx].item()):+.1f}")
    if hasattr(goal_cmd_term, "last_contact_body_id"):
        body_id = int(goal_cmd_term.last_contact_body_id[env_idx].item())
        if contact_sensor is not None and 0 <= body_id < len(contact_sensor.body_names):
            parts.append(f"contact_body={contact_sensor.body_names[body_id]}")
        elif body_id >= 0:
            parts.append(f"contact_body_id={body_id}")
    if hasattr(goal_cmd_term, "last_base_contact_force_w"):
        base_force_w = goal_cmd_term.last_base_contact_force_w[env_idx]
        parts.append(
            f"contact_force_w=({base_force_w[0].item():+.1f},{base_force_w[1].item():+.1f},{base_force_w[2].item():+.1f})"
        )
    if hasattr(goal_cmd_term, "last_base_contact_force_xy"):
        parts.append(f"contact_force_xy={goal_cmd_term.last_base_contact_force_xy[env_idx].item():+.1f}")
    if hasattr(goal_cmd_term, "last_base_contact_force"):
        parts.append(f"contact_force={goal_cmd_term.last_base_contact_force[env_idx].item():+.1f}")
    return " ".join(parts)


def format_base_force_step(contact_sensor=None, base_body_id: int | None = None, env_idx: int = 0) -> str:
    """Format the current world-frame base contact force for the step log."""
    if contact_sensor is None or base_body_id is None:
        return "base_force_w=(n/a)"
    base_force_w = contact_sensor.data.net_forces_w[env_idx, base_body_id]
    return f"base_force_w=({base_force_w[0].item():+.1f},{base_force_w[1].item():+.1f},{base_force_w[2].item():+.1f})"


def format_controller_debug(action_term, robot, contact_sensor=None, base_body_id: int | None = None, env_idx: int = 0) -> str:
    """Format controller state captured during the most recent physics sub-step."""
    root_lin_vel_b = robot.data.root_lin_vel_b[env_idx]
    root_ang_vel_b = robot.data.root_ang_vel_b[env_idx]
    wheel_joint_ids = action_term._wheel_joint_ids
    measured_wheel_vel = robot.data.joint_vel[env_idx, wheel_joint_ids]
    base_force_text = "base_force_w=(n/a) |F|=n/a"
    if contact_sensor is not None and base_body_id is not None:
        base_force_w = contact_sensor.data.net_forces_w[env_idx, base_body_id]
        base_force_norm = torch.linalg.vector_norm(base_force_w).item()
        base_force_text = (
            f"base_force_w=({base_force_w[0].item():+.1f},{base_force_w[1].item():+.1f},{base_force_w[2].item():+.1f}) "
            f"|F|={base_force_norm:+.1f}"
        )
    return (
        f"apply_count={action_term._debug_apply_count_since_process} "
        f"target(vx={action_term._target_vx[env_idx].item():+.3f}, "
        f"yaw_rate={action_term._target_yaw_target_rate[env_idx].item():+.3f}) "
        f"held(vx={action_term._stored_vx[env_idx].item():+.3f}, "
        f"yaw_rate={action_term._stored_yaw_target_rate[env_idx].item():+.3f}) "
        f"yaw(cur={action_term._debug_current_yaw[env_idx].item():+.3f}, "
        f"tgt={action_term._yaw_target[env_idx].item():+.3f}, "
        f"err={action_term._debug_yaw_error[env_idx].item():+.3f}) "
        f"rate(raw={action_term._debug_raw_yaw_rate[env_idx].item():+.3f}, "
        f"filt={action_term._debug_filtered_yaw_rate[env_idx].item():+.3f}) "
        f"corr={action_term._debug_correction[env_idx].item():+.3f} "
        f"wheel_tgt=({action_term._debug_left_wheel_velocity[env_idx].item():+.3f},"
        f"{action_term._debug_right_wheel_velocity[env_idx].item():+.3f}) "
        f"wheel_meas=({measured_wheel_vel[0].item():+.3f},{measured_wheel_vel[1].item():+.3f}) "
        f"body_vel=({root_lin_vel_b[0].item():+.3f},{root_lin_vel_b[1].item():+.3f}) "
        f"body_wz={root_ang_vel_b[2].item():+.3f} "
        f"{base_force_text}"
    )


def format_pose_goal_debug(
    goal_cmd_term,
    robot,
    xy_threshold: float,
    yaw_threshold_rad: float,
    proximity_xy_scale: float | None = None,
    proximity_yaw_scale_rad: float | None = None,
    proximity_activation_xy_threshold: float | None = None,
    env_idx: int = 0,
) -> str:
    """Summarize pose-goal errors relative to the target position and heading."""
    xy_error = torch.norm(
        robot.data.root_pos_w[env_idx, :2] - goal_cmd_term.goal_position_world[env_idx, :2],
        dim=0,
    ).item()
    current_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[env_idx : env_idx + 1])[2][0]
    yaw_error = math_utils.wrap_to_pi(goal_cmd_term.goal_heading_world[env_idx] - current_yaw).item()
    in_pose = (xy_error < xy_threshold) and (abs(yaw_error) < yaw_threshold_rad)
    pose_score_text = ""
    if (
        proximity_xy_scale is not None
        and proximity_yaw_scale_rad is not None
        and proximity_activation_xy_threshold is not None
    ):
        xy_score = 1.0 / (1.0 + (xy_error / max(proximity_xy_scale, 1.0e-6)) ** 2)
        yaw_score = 1.0 / (1.0 + (abs(yaw_error) / max(proximity_yaw_scale_rad, 1.0e-6)) ** 2)
        pose_score = xy_score * yaw_score if xy_error < proximity_activation_xy_threshold else 0.0
        pose_score_text = f", pose_score={pose_score:+.3f}"
    return (
        f"goal(xy_err={xy_error:+.3f}, "
        f"yaw_tgt={goal_cmd_term.goal_heading_world[env_idx].item():+.3f}, "
        f"yaw_err_deg={math.degrees(yaw_error):+.1f}, "
        f"in_pose={int(in_pose)}"
        f"{pose_score_text})"
    )


def main():
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    if env_cfg_class is None:
        raise ValueError(f"Task '{args_cli.task}' does not define an env config entry point.")

    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.disable_friction_randomization:
        env_cfg.events.physics_material = None
    if args_cli.disable_early_goal_termination:
        env_cfg.terminations.early_termination = None
    if args_cli.disable_action_scale_randomization:
        env_cfg.events.randomize_action_scale = None
    if args_cli.disable_low_pass_randomization:
        env_cfg.events.randomize_low_pass_filter_alpha = None
    action_cfg = env_cfg.actions.velocity_command
    if args_cli.command_ramp_up_time is not None:
        action_cfg.command_ramp_up_time = args_cli.command_ramp_up_time
    if args_cli.command_ramp_down_time is not None:
        action_cfg.command_ramp_down_time = args_cli.command_ramp_down_time
    if args_cli.yaw_pid_kp is not None:
        action_cfg.yaw_pid_kp = args_cli.yaw_pid_kp
    if args_cli.yaw_pid_ki is not None:
        action_cfg.yaw_pid_ki = args_cli.yaw_pid_ki
    if args_cli.yaw_pid_kd is not None:
        action_cfg.yaw_pid_kd = args_cli.yaw_pid_kd
    if args_cli.yaw_pid_integral_limit is not None:
        action_cfg.yaw_pid_integral_limit = args_cli.yaw_pid_integral_limit
    if args_cli.yaw_hold_max_correction is not None:
        action_cfg.yaw_hold_max_correction = args_cli.yaw_hold_max_correction
    if args_cli.yaw_rate_filter_time is not None:
        action_cfg.yaw_rate_filter_time = args_cli.yaw_rate_filter_time
    pose_goal_xy_threshold = args_cli.pose_goal_xy_threshold
    pose_goal_yaw_threshold_rad = math.radians(args_cli.pose_goal_yaw_threshold_deg)
    pose_goal_proximity_xy_scale = args_cli.pose_goal_proximity_xy_scale
    pose_goal_proximity_activation_xy_threshold = args_cli.pose_goal_proximity_activation_xy_threshold
    pose_goal_proximity_yaw_scale_rad = math.radians(args_cli.pose_goal_proximity_yaw_scale_deg)
    if args_cli.enable_pose_goal_reward:
        pose_goal_reward_cfg = getattr(env_cfg.rewards, "pose_goal_hold_bonus", None)
        if pose_goal_reward_cfg is None:
            raise ValueError("The selected task does not define a pose_goal_hold_bonus reward term.")
        pose_goal_reward_cfg.weight = args_cli.pose_goal_reward_weight
        pose_goal_reward_cfg.params["xy_threshold"] = pose_goal_xy_threshold
        pose_goal_reward_cfg.params["yaw_threshold"] = pose_goal_yaw_threshold_rad
    if args_cli.enable_pose_goal_proximity_reward:
        pose_goal_proximity_cfg = getattr(env_cfg.rewards, "pose_goal_proximity", None)
        if pose_goal_proximity_cfg is None:
            raise ValueError("The selected task does not define a pose_goal_proximity reward term.")
        pose_goal_proximity_cfg.weight = args_cli.pose_goal_proximity_reward_weight
        pose_goal_proximity_cfg.params["xy_scale"] = pose_goal_proximity_xy_scale
        pose_goal_proximity_cfg.params["yaw_scale"] = pose_goal_proximity_yaw_scale_rad
        pose_goal_proximity_cfg.params["activation_xy_threshold"] = pose_goal_proximity_activation_xy_threshold

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env_unwrapped = env.unwrapped

    action_term = env_unwrapped.action_manager._terms["velocity_command"]
    robot = action_term._asset
    goal_cmd_term = env_unwrapped.command_manager._terms.get("robot_goal")
    contact_sensor = env_unwrapped.scene.sensors.get("contact_forces")
    base_body_id = None
    if contact_sensor is not None:
        try:
            base_body_id = contact_sensor.body_names.index("base_link")
        except ValueError:
            base_body_id = None
    if not hasattr(action_term, "_yaw_target"):
        raise ValueError(
            f"Task '{args_cli.task}' does not appear to use the DiffDrive action term."
        )
    if args_cli.blue_only_arrow and goal_cmd_term is not None:
        goal_cmd_term._show_diff_drive_current_velocity_marker = False

    if getattr(action_term.cfg, "policy_distr_type", None) != "gaussian":
        raise ValueError("play_keyboard.py currently assumes gaussian/tanh diff-drive actions.")

    cfg_vx_scale = float(action_term.cfg.policy_scaling[0])
    cfg_yaw_scale = float(action_term.cfg.policy_scaling[2])

    keyboard = DifferentialDriveKeyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.linear_speed,
            v_y_sensitivity=0.0,
            omega_z_sensitivity=args_cli.yaw_speed,
            sim_device=env_unwrapped.device,
        )
    )

    pending_reset = False

    def handle_reset():
        nonlocal pending_reset
        pending_reset = True

    keyboard.add_callback("R", handle_reset)

    print(keyboard)
    print(
        f"[INFO] Task={args_cli.task} physics_dt={env_unwrapped.physics_dt:.5f}s "
        f"step_dt={env_unwrapped.step_dt:.5f}s action_decimation={env_unwrapped.cfg.decimation}"
    )
    print(
        f"[INFO] Keyboard request range: desired_vx in [-{args_cli.linear_speed:.2f}, {args_cli.linear_speed:.2f}] m/s, "
        f"desired_yaw_rate in [-{args_cli.yaw_speed:.2f}, {args_cli.yaw_speed:.2f}] rad/s"
    )
    print(
        f"[INFO] Base policy scaling from config: vx={cfg_vx_scale:.2f} yaw_rate={cfg_yaw_scale:.2f}"
    )
    print(f"[INFO] PID config: {format_pid_config(action_term)}")
    if args_cli.enable_pose_goal_reward:
        print(
            f"[INFO] Pose-goal reward enabled: weight={args_cli.pose_goal_reward_weight:.3f} "
            f"xy_threshold={pose_goal_xy_threshold:.3f}m "
            f"yaw_threshold={args_cli.pose_goal_yaw_threshold_deg:.1f}deg"
        )
    if args_cli.enable_pose_goal_proximity_reward:
        print(
            f"[INFO] Pose-goal proximity reward enabled: weight={args_cli.pose_goal_proximity_reward_weight:.3f} "
            f"xy_scale={pose_goal_proximity_xy_scale:.3f}m "
            f"activation_xy_threshold={pose_goal_proximity_activation_xy_threshold:.3f}m "
            f"yaw_scale={args_cli.pose_goal_proximity_yaw_scale_deg:.1f}deg"
        )

    obs, extras = env.reset()
    print(f"[INFO] Runtime controller config: {format_runtime_controller_config(action_term)}")
    step_count = 0
    episode_reward_sum = 0.0
    episode_step_count = 0
    episode_reward_terms: dict[str, float] = {}

    while simulation_app.is_running():
        if pending_reset:
            obs, extras = env.reset()
            keyboard.reset()
            pending_reset = False
            episode_reward_sum = 0.0
            episode_step_count = 0
            episode_reward_terms = {}
            print("[INFO] Environment reset.")
            print(f"[INFO] Runtime controller config: {format_runtime_controller_config(action_term)}")

        command = keyboard.advance()
        desired_vx = float(command[0].item())
        desired_yaw_rate = float(command[2].item())

        runtime_vx_scale, runtime_yaw_scale = get_runtime_action_scales(action_term)
        used_vx, vx_saturated = clamp_to_scale(desired_vx, runtime_vx_scale)
        used_yaw_rate, yaw_saturated = clamp_to_scale(desired_yaw_rate, runtime_yaw_scale)
        raw_vx = physical_to_raw_policy_action(used_vx, runtime_vx_scale)
        raw_yaw = physical_to_raw_policy_action(used_yaw_rate, runtime_yaw_scale)
        actions = torch.tensor([[raw_vx, raw_yaw]], device=env_unwrapped.device, dtype=torch.float32)
        actions = actions.repeat(env_unwrapped.num_envs, 1)

        obs, rew, terminated, truncated, extras = env.step(actions)
        step_count += 1
        episode_step_count += 1
        episode_reward_sum += float(rew[0].item())
        for name, value in get_reward_terms_dict(env_unwrapped, env_idx=0).items():
            episode_reward_terms[name] = episode_reward_terms.get(name, 0.0) + value * env_unwrapped.step_dt

        done0 = bool((terminated[0] | truncated[0]).item())
        should_print = (step_count % max(1, args_cli.print_every) == 0) or done0
        if should_print:
            reward_terms = format_reward_terms(env_unwrapped, env_idx=0)
            yaw_target = float(action_term._yaw_target[0].item())
            stored_vx = float(action_term._stored_vx[0].item())
            pose_goal_text = ""
            if goal_cmd_term is not None and hasattr(goal_cmd_term, "goal_heading_world"):
                pose_goal_text = format_pose_goal_debug(
                    goal_cmd_term,
                    robot,
                    xy_threshold=pose_goal_xy_threshold,
                    yaw_threshold_rad=pose_goal_yaw_threshold_rad,
                    proximity_xy_scale=pose_goal_proximity_xy_scale if args_cli.enable_pose_goal_proximity_reward else None,
                    proximity_yaw_scale_rad=pose_goal_proximity_yaw_scale_rad if args_cli.enable_pose_goal_proximity_reward else None,
                    proximity_activation_xy_threshold=(
                        pose_goal_proximity_activation_xy_threshold if args_cli.enable_pose_goal_proximity_reward else None
                    ),
                    env_idx=0,
                )
            print(
                f"[STEP {step_count:05d}] "
                f"cmd(vx={desired_vx:+.3f}, yaw_rate={desired_yaw_rate:+.3f}) "
                f"used(vx={used_vx:+.3f}, yaw_rate={used_yaw_rate:+.3f}) "
                f"raw=({raw_vx:+.3f},{raw_yaw:+.3f}) "
                f"{'SAT ' if (vx_saturated or yaw_saturated) else ''}"
                f"reward={rew[0].item():+.3f} "
                f"stored_vx={stored_vx:+.3f} yaw_target={yaw_target:+.3f} "
                f"{reward_terms} "
                f"{pose_goal_text} "
                f"{format_base_force_step(contact_sensor, base_body_id, env_idx=0)}"
            )
            if args_cli.debug_controller:
                print(
                    f"[CTRL] {format_controller_debug(action_term, robot, contact_sensor, base_body_id, env_idx=0)}"
                )

        if done0:
            term_names = active_termination_terms(env_unwrapped, env_idx=0)
            episode_log = format_episode_log(extras)
            episode_reward_text = format_episode_reward_breakdown(episode_reward_terms)
            done_diag_text = format_done_diagnostics(goal_cmd_term, contact_sensor, env_idx=0)
            print(
                f"[DONE] terminated={bool(terminated[0].item())} "
                f"truncated={bool(truncated[0].item())} "
                f"terms={term_names if term_names else ['none']}"
            )
            if done_diag_text:
                print(f"[DONE DIAG] {done_diag_text}")
            print(
                f"[EPISODE RETURN] total={episode_reward_sum:+.3f} "
                f"steps={episode_step_count} duration={episode_step_count * env_unwrapped.step_dt:.1f}s"
            )
            if episode_reward_text:
                print(f"[EPISODE REWARDS] {episode_reward_text}")
            if episode_log:
                print(f"[EPISODE] {episode_log}")
            print("[INFO] Pausing 5s before continuing...")
            time.sleep(5.0)
            try:
                input("[INFO] Press Enter to continue...")
            except EOFError:
                pass
            print(f"[INFO] Next-episode controller config: {format_runtime_controller_config(action_term)}")
            episode_reward_sum = 0.0
            episode_step_count = 0
            episode_reward_terms = {}

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
