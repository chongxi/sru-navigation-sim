# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_inv, yaw_quat, quat_mul, euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand


def euler_xyz_from_quat_wrapped(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternion to Euler angles (XYZ convention) with wrapping to [-pi, pi].

    Args:
        quat: Quaternion tensor of shape (..., 4) in (w, x, y, z) format.

    Returns:
        Tuple of (roll, pitch, yaw) tensors.
    """
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    # Wrap to [-pi, pi]
    roll = torch.remainder(roll + torch.pi, 2 * torch.pi) - torch.pi
    pitch = torch.remainder(pitch + torch.pi, 2 * torch.pi) - torch.pi
    yaw = torch.remainder(yaw + torch.pi, 2 * torch.pi) - torch.pi
    return roll, pitch, yaw


def _goal_pose_mask(
    asset: Articulation,
    goal_cmd_generator: "RobotNavigationGoalCommand",
    distance_threshold: float,
    yaw_threshold: float | None = None,
    lin_speed_threshold: float | None = None,
    yaw_rate_threshold: float | None = None,
) -> torch.Tensor:
    """Return a boolean mask for being close enough to the target pose."""
    xy_error = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_generator.pos_command_w[:, :2], dim=1)
    at_goal = xy_error < distance_threshold

    if yaw_threshold is not None:
        current_yaw = euler_xyz_from_quat(asset.data.root_quat_w)[2]
        yaw_error = torch.abs(wrap_to_pi(goal_cmd_generator.goal_heading_world - current_yaw))
        at_goal = torch.logical_and(at_goal, yaw_error < float(yaw_threshold))

    if lin_speed_threshold is not None:
        lin_speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        at_goal = torch.logical_and(at_goal, lin_speed < float(lin_speed_threshold))

    if yaw_rate_threshold is not None:
        yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
        at_goal = torch.logical_and(at_goal, yaw_rate < float(yaw_rate_threshold))

    return at_goal


def _record_failure(
    goal_cmd_generator: "RobotNavigationGoalCommand",
    termination: torch.Tensor,
):
    """Record a failed episode result for terminated environments."""
    env_ids = torch.where(termination)[0]
    if env_ids.numel() > 0:
        goal_cmd_generator.goal_reached_buffer.add(torch.zeros_like(termination, dtype=torch.float), env_ids)


def _record_success(
    goal_cmd_generator: "RobotNavigationGoalCommand",
    termination: torch.Tensor,
):
    """Record a successful episode result for terminated environments."""
    env_ids = torch.where(termination)[0]
    if env_ids.numel() > 0:
        goal_cmd_generator.goal_reached_buffer.add(torch.ones_like(termination, dtype=torch.float), env_ids)


def _ensure_trap_history_buffers(
    goal_cmd_generator: "RobotNavigationGoalCommand",
    window_steps: int,
    num_cells: int,
):
    """Allocate or resize trapped-occupancy buffers on demand."""
    needs_init = (
        not hasattr(goal_cmd_generator, "trap_cell_history")
        or goal_cmd_generator.trap_cell_history.shape[1] != window_steps
        or goal_cmd_generator.trap_cell_counts.shape[1] != num_cells
    )
    if not needs_init:
        return

    num_envs = goal_cmd_generator.num_envs
    device = goal_cmd_generator.device
    goal_cmd_generator.trap_cell_history = torch.full(
        (num_envs, window_steps), -1, device=device, dtype=torch.long
    )
    goal_cmd_generator.trap_cell_counts = torch.zeros(
        (num_envs, num_cells), device=device, dtype=torch.int16
    )
    goal_cmd_generator.trap_cell_history_head = torch.zeros(
        num_envs, device=device, dtype=torch.long
    )
    goal_cmd_generator.trap_cell_history_len = torch.zeros(
        num_envs, device=device, dtype=torch.long
    )


def _terrain_cell_metadata(
    env: "ManagerBasedRLEnv",
) -> tuple[float, int, int]:
    """Read the current coarse-cell size and grid dimensions from the terrain config."""
    terrain_cfg = env.scene.terrain.cfg.terrain_generator
    default_grid = max(1, int(round(terrain_cfg.size[0] / 2.0)))
    default_cell_size = float(terrain_cfg.size[0]) / float(default_grid)

    cell_size = default_cell_size
    grid_size = (default_grid, default_grid)
    if getattr(terrain_cfg, "sub_terrains", None):
        base = next(iter(terrain_cfg.sub_terrains.values()))
        cell_size = float(getattr(base, "cell_size", default_cell_size))
        grid_size = tuple(getattr(base, "grid_size", (default_grid, default_grid)))

    grid_x = max(1, int(grid_size[0]))
    grid_y = max(1, int(grid_size[1]))
    return cell_size, grid_x, grid_y


def _current_coarse_cell_ids(
    env: "ManagerBasedRLEnv",
    asset: Articulation,
) -> tuple[torch.Tensor, int]:
    """Map each robot pose to a coarse terrain-cell id within its current tile."""
    terrain = env.scene.terrain
    cell_size, grid_x, grid_y = _terrain_cell_metadata(env)
    half_size = float(terrain.cfg.terrain_generator.size[0]) * 0.5

    levels = terrain.terrain_levels
    types = terrain.terrain_types
    terrain_origins = terrain.terrain_origins[levels, types]
    local_xy = asset.data.root_pos_w[:, :2] - terrain_origins[:, :2] + half_size
    cell_xy = torch.floor(local_xy / max(cell_size, 1.0e-6)).to(dtype=torch.long)
    cell_xy[:, 0] = cell_xy[:, 0].clamp(0, grid_x - 1)
    cell_xy[:, 1] = cell_xy[:, 1].clamp(0, grid_y - 1)
    cell_ids = cell_xy[:, 0] * grid_y + cell_xy[:, 1]
    return cell_ids, grid_x * grid_y


def time_out_navigation(
    env: "ManagerBasedRLEnv",
    goal_cmd_name: str = "robot_goal",
    distance_threshold: float = 0.5,
    yaw_threshold: float | None = None,
    lin_speed_threshold: float | None = None,
    yaw_rate_threshold: float | None = None,
) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length.

    This also tracks success metrics by checking if the robot reached the goal before timeout.
    """
    from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[goal_cmd_name]
    asset: Articulation = env.scene["robot"]
    at_goal = _goal_pose_mask(
        asset,
        goal_cmd_generator,
        distance_threshold=distance_threshold,
        yaw_threshold=yaw_threshold,
        lin_speed_threshold=lin_speed_threshold,
        yaw_rate_threshold=yaw_rate_threshold,
    )
    goal_cmd_generator.time_at_goal[at_goal] += env.step_dt
    goal_cmd_generator.time_at_goal[~at_goal] = 0.0

    termination = env.episode_length_buf >= env.max_episode_length

    env_ids = torch.where(termination)[0]

    if env_ids.numel() > 0:  # Check if env_ids is not empty
        success_masks = goal_cmd_generator.time_at_goal >= (
            goal_cmd_generator.required_time_at_goal_in_steps * env.step_dt
        )
        value_buffer = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        value_buffer[success_masks] = 1.0  # Success
        goal_cmd_generator.goal_reached_buffer.add(value_buffer, env_ids)

    return termination


def illegal_contact_navigation(
    env: "ManagerBasedRLEnv",
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    goal_cmd_name: str = "robot_goal",
    horizontal_only: bool = False,
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[goal_cmd_name]
    contact_force_vectors = net_contact_forces[:, :, sensor_cfg.body_ids]
    if horizontal_only:
        contact_force_norms = torch.norm(contact_force_vectors[..., :2], dim=-1)
    else:
        contact_force_norms = torch.norm(contact_force_vectors, dim=-1)
    contact_force_max_per_body = torch.max(contact_force_norms, dim=1)[0]
    num_selected_bodies = contact_force_norms.shape[-1]
    flat_contact_force_norms = contact_force_norms.reshape(contact_force_norms.shape[0], -1)
    max_contact_indices = torch.argmax(flat_contact_force_norms, dim=1)
    flat_contact_force_vectors = contact_force_vectors.reshape(contact_force_vectors.shape[0], -1, 3)
    selected_body_ids = torch.as_tensor(sensor_cfg.body_ids, device=flat_contact_force_vectors.device, dtype=torch.long)
    selected_body_index = torch.remainder(max_contact_indices, num_selected_bodies)
    goal_cmd_generator.last_base_contact_force_w[:] = flat_contact_force_vectors[
        torch.arange(flat_contact_force_vectors.shape[0], device=flat_contact_force_vectors.device),
        max_contact_indices,
    ]
    goal_cmd_generator.last_base_contact_force_xy[:] = torch.norm(goal_cmd_generator.last_base_contact_force_w[:, :2], dim=-1)
    goal_cmd_generator.last_base_contact_force[:] = torch.max(contact_force_max_per_body, dim=1)[0]
    goal_cmd_generator.last_contact_body_id[:] = selected_body_ids[selected_body_index]
    termination = torch.any(contact_force_max_per_body > threshold, dim=1)

    _record_failure(goal_cmd_generator, termination)
    return termination


def large_angle_termination_navigation(
    env: "ManagerBasedRLEnv",
    threshold: float,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate when the robot exceeds a pitch or roll angle threshold."""
    from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[goal_cmd_name]

    # degree to rad
    threshold_rad = threshold * torch.pi / 180.0

    robot = env.scene["robot"]
    yaw_q = yaw_quat(robot.data.root_quat_w)
    base_quat_b = quat_mul(quat_inv(yaw_q), robot.data.root_quat_w)
    robot_roll, robot_pitch, _ = euler_xyz_from_quat_wrapped(base_quat_b)
    goal_cmd_generator.last_roll[:] = robot_roll
    goal_cmd_generator.last_pitch[:] = robot_pitch

    termination = torch.logical_or(torch.abs(robot_pitch) > threshold_rad, torch.abs(robot_roll) > threshold_rad)

    _record_failure(goal_cmd_generator, termination)
    return termination


def in_goal_navigation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
    goal_cmd_name: str = "robot_goal",
    yaw_threshold: float | None = None,
    lin_speed_threshold: float | None = None,
    yaw_rate_threshold: float | None = None,
) -> torch.Tensor:
    """Terminate the episode when the goal is reached.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.
        goal_cmd_name: The name of the goal command term.

    Returns:
        Boolean tensor indicating whether the goal is reached.
    """
    from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

    # Extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms.get(goal_cmd_name)

    # Require the robot to remain continuously inside the goal region. Merely
    # touching the goal once should not guarantee termination a few seconds later.
    at_goal = _goal_pose_mask(
        asset,
        goal_cmd_generator,
        distance_threshold=distance_threshold,
        yaw_threshold=yaw_threshold,
        lin_speed_threshold=lin_speed_threshold,
        yaw_rate_threshold=yaw_rate_threshold,
    )
    goal_cmd_generator.time_at_goal_in_steps[at_goal] += 1
    goal_cmd_generator.time_at_goal_in_steps[~at_goal] = 0

    # Determine if termination condition is met
    termination = goal_cmd_generator.time_at_goal_in_steps >= goal_cmd_generator.required_time_at_goal_in_steps

    _record_success(goal_cmd_generator, termination)
    return termination


def near_goal_navigation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.6,
    goal_cmd_name: str = "robot_goal",
    yaw_threshold: float | None = None,
    lin_speed_threshold: float | None = None,
    yaw_rate_threshold: float | None = None,
    in_goal_distance_threshold: float = 0.35,
    in_goal_yaw_threshold: float | None = None,
    in_goal_lin_speed_threshold: float | None = None,
    in_goal_yaw_rate_threshold: float | None = None,
    hold_time_s: float = 8.0,
) -> torch.Tensor:
    """Terminate when the robot spends too long inside a looser near-goal region."""
    from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms.get(goal_cmd_name)

    near_goal = _goal_pose_mask(
        asset,
        goal_cmd_generator,
        distance_threshold=distance_threshold,
        yaw_threshold=yaw_threshold,
        lin_speed_threshold=lin_speed_threshold,
        yaw_rate_threshold=yaw_rate_threshold,
    )
    in_goal = _goal_pose_mask(
        asset,
        goal_cmd_generator,
        distance_threshold=in_goal_distance_threshold,
        yaw_threshold=in_goal_yaw_threshold,
        lin_speed_threshold=in_goal_lin_speed_threshold,
        yaw_rate_threshold=in_goal_yaw_rate_threshold,
    )

    goal_cmd_generator.near_goal_accumulated_steps[near_goal] += 1
    required_steps = max(1, int(round(float(hold_time_s) / env.step_dt)))
    termination = near_goal & ~in_goal & (
        goal_cmd_generator.near_goal_accumulated_steps >= required_steps
    )

    _record_failure(goal_cmd_generator, termination)
    return termination


def trapped_navigation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cmd_name: str = "robot_goal",
    near_goal_distance_threshold: float = 0.6,
    near_goal_yaw_threshold: float | None = None,
    near_goal_lin_speed_threshold: float | None = None,
    near_goal_yaw_rate_threshold: float | None = None,
    window_s: float = 20.0,
    single_cell_time_s: float = 4.0,
    top_two_cell_time_s: float = 8.0,
) -> torch.Tensor:
    """Terminate when the robot is locally trapped outside the near-goal region.

    A rolling occupancy window is maintained over coarse terrain cells. While the robot is
    near the goal, the buffer stores a sentinel instead of the current cell, so goal-region
    lingering never contributes to the trap counters.
    """
    from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms.get(goal_cmd_name)

    near_goal = _goal_pose_mask(
        asset,
        goal_cmd_generator,
        distance_threshold=near_goal_distance_threshold,
        yaw_threshold=near_goal_yaw_threshold,
        lin_speed_threshold=near_goal_lin_speed_threshold,
        yaw_rate_threshold=near_goal_yaw_rate_threshold,
    )

    window_steps = max(1, int(round(float(window_s) / env.step_dt)))
    single_cell_steps = max(1, int(round(float(single_cell_time_s) / env.step_dt)))
    top_two_steps = max(1, int(round(float(top_two_cell_time_s) / env.step_dt)))

    cell_ids, num_cells = _current_coarse_cell_ids(env, asset)
    _ensure_trap_history_buffers(goal_cmd_generator, window_steps, num_cells)

    history = goal_cmd_generator.trap_cell_history
    counts = goal_cmd_generator.trap_cell_counts
    head = goal_cmd_generator.trap_cell_history_head
    hist_len = goal_cmd_generator.trap_cell_history_len

    write_values = torch.where(near_goal, torch.full_like(cell_ids, -1), cell_ids)
    env_indices = torch.arange(env.num_envs, device=env.device)
    old_values = history[env_indices, head]
    old_valid = old_values >= 0
    if old_valid.any():
        counts[env_indices[old_valid], old_values[old_valid]] -= 1

    history[env_indices, head] = write_values
    new_valid = write_values >= 0
    if new_valid.any():
        counts[env_indices[new_valid], write_values[new_valid]] += 1

    head[:] = (head + 1) % window_steps
    hist_len[:] = torch.clamp(hist_len + 1, max=window_steps)

    max_count = counts.max(dim=1).values.to(dtype=torch.long)
    if counts.shape[1] >= 2:
        top_two_sum = counts.to(dtype=torch.long).topk(k=2, dim=1).values.sum(dim=1)
    else:
        top_two_sum = max_count

    latent_trapped = (max_count >= single_cell_steps) | (top_two_sum >= top_two_steps)
    window_ready = hist_len >= window_steps
    termination = latent_trapped & window_ready & ~near_goal

    _record_failure(goal_cmd_generator, termination)
    return termination


def at_goal_navigation(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible alias for the renamed in-goal termination."""
    return in_goal_navigation(*args, **kwargs)


def terrain_fall(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fall_height_threshold: float = -1.0,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate when the robot falls below a certain height threshold.

    This termination is triggered when the robot's z-position falls below a
    specified threshold, indicating that the robot has fallen off the terrain
    or into a deep pit.

    Args:
        env: The learning environment.
        asset_cfg: The configuration for the robot asset.
        fall_height_threshold: The z-height below which the robot is considered fallen (in meters).
            Default is -1.0m to account for pits which can be ~1.5m deep.
        goal_cmd_name: The name of the goal command term.

    Returns:
        Boolean tensor indicating whether the robot has fallen.
    """
    # Direct tensor access for z-coordinate (avoids intermediate variable allocation)
    termination = env.scene[asset_cfg.name].data.root_pos_w[:, 2] < fall_height_threshold

    goal_cmd = env.command_manager._terms.get(goal_cmd_name)
    if goal_cmd is not None and termination.any():
        _record_failure(goal_cmd, termination)

    return termination
