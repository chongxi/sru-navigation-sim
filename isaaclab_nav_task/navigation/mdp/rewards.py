# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Reward functions for navigation tasks.

These functions can be passed to :class:`isaaclab.managers.RewardTermCfg`
to specify the reward function and its parameters.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l1(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize the rate of change of the actions using L1 kernel."""
    return torch.sum(torch.abs(env.action_manager.action - env.action_manager.prev_action), dim=1)


def lateral_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for moving laterally using L1-Kernel.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the lateral velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    lateral_velocity = asset.data.root_lin_vel_b[:, 1]
    reward = torch.abs(lateral_velocity)
    return reward


def rot_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for rotating around the z-axis using an L2-Kernel.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the rotational velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rot_vel_norm = torch.norm(asset.data.root_ang_vel_b, dim=1)
    return rot_vel_norm


def reach_goal_xyz(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigmoid: float,
    T_r: float,
    probability: float,
    flat: bool,
    ratio: bool,
) -> torch.Tensor:
    """Reward goal reaching with configurable sigmoid shaping.

    Args:
        env: The learning environment.
        command_name: Name of the goal command.
        sigmoid: Sigmoid parameter for shaping.
        T_r: Time reward scaling factor.
        probability: Probability of random sampling.
        flat: Whether to only consider xy error (ignore z).
        ratio: Whether to scale by travel distance ratio.

    Returns:
        Dense reward based on distance to goal.
    """
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[command_name]

    t = env.episode_length_buf
    T = env.max_episode_length

    if flat:
        xyz_error = torch.norm(goal_cmd_generator._get_unscaled_command()[:, :2], dim=1)
    else:
        xyz_error = torch.norm(goal_cmd_generator._get_unscaled_command(), dim=1)

    reward = 1 / (1 + torch.square(xyz_error / sigmoid)) / T_r

    timeup_mask = t > (T - goal_cmd_generator.required_time_at_goal_in_steps)
    random_mask = torch.rand_like(t.float()) < probability
    timeup_mask = torch.logical_or(timeup_mask, random_mask)

    arrive_mask = goal_cmd_generator.time_at_goal > 0.0
    reward_mask = torch.logical_or(timeup_mask, arrive_mask)

    if ratio:
        # Calculate the travel distance ratio relative to the initial goal distance
        travel_distance = torch.max(
            goal_cmd_generator.distance_traveled, goal_cmd_generator.initial_distance_to_goal
        )
        travel_distance_ratio = goal_cmd_generator.initial_distance_to_goal / (travel_distance + 1e-6)
    else:
        travel_distance_ratio = torch.ones_like(reward)

    reward = reward * reward_mask.float() * travel_distance_ratio

    return reward


def goal_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Potential-based dense reward: decrease in distance to goal.

    Returns (prev_distance - current_distance) each step. Positive when
    getting closer, negative when moving away, zero when stationary or
    pushing into a wall. Returns zero on episode reset / goal resampling
    (detected when distance jumps by more than physically possible in one step).
    """
    goal_cmd: RobotNavigationGoalCommand = env.command_manager._terms[command_name]
    # Initialize prev_distance buffer on first call
    if not hasattr(goal_cmd, "_prev_distance_to_goal"):
        goal_cmd._prev_distance_to_goal = goal_cmd.distance_to_goal.clone()
    delta = goal_cmd._prev_distance_to_goal - goal_cmd.distance_to_goal
    goal_cmd._prev_distance_to_goal = goal_cmd.distance_to_goal.clone()
    # Zero out reset spikes: robot can't move more than ~0.5m in one step
    # at 10Hz/2.5m/s, so |delta| > 1.0 means goal was resampled
    delta = torch.where(delta.abs() > 1.0, torch.zeros_like(delta), delta)
    return delta


def backward_movement_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Small penalty for backward movement as a regularization term.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Penalty [0, +1] based on backward velocity (to be used with negative weight).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the penalty
    forward_velocity = asset.data.root_lin_vel_b[:, 0]
    # Only penalize negative forward velocity (backward movement)
    backward_velocity = torch.clamp(-forward_velocity, min=0.0, max=1.0)
    return backward_velocity


def pose_goal_hold_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    xy_threshold: float,
    yaw_threshold: float,
    lin_speed_threshold: float,
    yaw_rate_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Constant per-step bonus for being at the target pose and nearly stationary."""
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd: RobotNavigationGoalCommand = env.command_manager._terms[command_name]

    xy_error = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd.pos_command_w[:, :2], dim=1)
    current_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    yaw_error = torch.abs(math_utils.wrap_to_pi(goal_cmd.goal_heading_world - current_yaw))
    lin_speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])

    in_pose = torch.logical_and(xy_error < xy_threshold, yaw_error < yaw_threshold)
    settled = torch.logical_and(lin_speed < lin_speed_threshold, yaw_rate < yaw_rate_threshold)
    return torch.logical_and(in_pose, settled).float()


def pose_goal_proximity(
    env: ManagerBasedRLEnv,
    command_name: str,
    xy_scale: float,
    yaw_scale: float,
    activation_xy_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Smooth dense reward for approaching the target pose.

    The reward is high only when both the XY position error and the yaw error
    are small, but it stays dense everywhere so the policy gets directional
    signal before it reaches the final pose.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd: RobotNavigationGoalCommand = env.command_manager._terms[command_name]

    xy_error = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd.pos_command_w[:, :2], dim=1)
    current_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    yaw_error = torch.abs(math_utils.wrap_to_pi(goal_cmd.goal_heading_world - current_yaw))

    xy_scale = max(float(xy_scale), 1.0e-6)
    yaw_scale = max(float(yaw_scale), 1.0e-6)

    xy_score = 1.0 / (1.0 + torch.square(xy_error / xy_scale))
    yaw_score = 1.0 / (1.0 + torch.square(yaw_error / yaw_scale))
    active = xy_error < float(activation_xy_threshold)
    return xy_score * yaw_score * active.float()
