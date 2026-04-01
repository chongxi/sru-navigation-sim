#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Play a trained navigation policy with the real task env, night visuals, and live depth display.

This script intentionally follows the same env -> wrapper -> runner -> policy path as `scripts/play.py`.
Unlike the previous standalone reimplementation, it does not rebuild observations, actions, PID control,
or resets manually. That keeps behavior aligned with the task while still exposing the raw depth camera
stream and using a night-style visual setup.

Usage:
    python scripts/play_with_depth_night.py \
      --checkpoint logs/rsl_rl/diff_drive_navigation_mdpo/2026-03-30_22-01-17/model_900.pt
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections.abc import Sequence
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Play the real task env with night visuals and live depth display."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Nav-MDPO-DiffDrive-Play-v0",
    help="Task name. Defaults to the DiffDrive Play task.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use last checkpoint from logs.")
parser.add_argument("--export_jit", action="store_true", default=False, help="Export policy as JIT module.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--terrain_rows", type=int, default=None, help="Terrain grid rows (difficulty levels).")
parser.add_argument("--terrain_cols", type=int, default=None, help="Terrain grid columns (variations).")
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    choices=["maze", "non_maze", "both", "flat"],
    help="Terrain type: maze, non_maze, both (default from config), or flat (no walls).",
)
parser.add_argument(
    "--difficulty",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Terrain difficulty range, e.g. --difficulty 0.3 0.8",
)
parser.add_argument("--cell_size", type=float, default=None, help="Maze cell size in meters.")
parser.add_argument("--wall_ratio", type=float, default=None, help="Random wall ratio (0=no walls, 1=max walls).")
parser.add_argument("--terrain_size", type=float, default=None, help="Terrain tile size in meters (default 30).")
parser.add_argument("--grid_size", type=int, default=None, help="Maze grid dimension, e.g. 15 for 15x15.")
parser.add_argument("--episode_length", type=float, default=None, help="Episode length in seconds.")
parser.add_argument(
    "--policy_scale_yaw",
    type=float,
    default=None,
    help="Override the third entry of actions.velocity_command.policy_scaling (yaw/heading scale).",
)
parser.add_argument(
    "--decimation",
    type=int,
    default=None,
    help="Override policy decimation. For diff-drive with dt=1/60, use 6 for 10Hz or 1 for 60Hz.",
)
parser.add_argument(
    "--debug_raycast",
    action="store_true",
    default=True,
    help="Enable raycast camera debug visualization.",
)
parser.add_argument(
    "--show_depth",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show a live OpenCV depth window.",
)
parser.add_argument("--depth_env_idx", type=int, default=0, help="Environment index used for depth visualization.")
parser.add_argument(
    "--depth_vis_every",
    type=int,
    default=1,
    help="Refresh the OpenCV depth window every N env steps.",
)
parser.add_argument(
    "--depth_display_scale",
    type=int,
    default=0,
    help="Integer upscaling factor for the displayed depth image.",
)
parser.add_argument("--depth_min", type=float, default=0.3, help="Minimum displayed depth in meters.")
parser.add_argument("--depth_max", type=float, default=8.0, help="Maximum displayed depth in meters.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401
import isaaclab_nav_task.navigation.mdp as nav_mdp

from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import PositionSampler, RobotNavigationGoalCommand
from isaaclab_nav_task.navigation.mdp.navigation.goal_commands_cfg import RobotNavigationGoalCommandCfg
from isaaclab_nav_task.terrains import HfMazeTerrainCfg
from isaaclab_nav_task.vecenv_wrapper import SruRslRlVecEnvWrapper


try:
    import cv2
except ImportError:
    cv2 = None


def find_latest_checkpoint(log_path: str, checkpoint_pattern: str = "model_.*.pt") -> str:
    """Find the latest checkpoint file in the log directory."""
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

    run_dirs = []
    for entry in os.scandir(log_path):
        if entry.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", entry.name):
            run_dirs.append(entry.name)

    if not run_dirs:
        raise ValueError(f"No run directories found in: {log_path}")

    run_dirs.sort()
    latest_run = run_dirs[-1]
    run_path = os.path.join(log_path, latest_run)

    checkpoint_files = []
    for file_name in os.listdir(run_path):
        if re.match(checkpoint_pattern, file_name):
            checkpoint_files.append(file_name)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files matching '{checkpoint_pattern}' found in: {run_path}")

    checkpoint_files.sort(key=lambda item: f"{item:0>15}")
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(run_path, latest_checkpoint)


def load_checkpoint_with_fallback(runner: OnPolicyRunner, checkpoint_path: str, load_optimizer: bool = True):
    """Load checkpoint with fallback for PyTorch compatibility issues."""
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    loaded_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if runner.is_mdpo:
        runner.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    else:
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)

    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

    if load_optimizer:
        if runner.is_mdpo:
            runner.alg.optimizer_1.load_state_dict(loaded_dict["optimizer_state_dict"])
        else:
            runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

    runner.current_learning_iteration = loaded_dict["iter"]
    print(f"[INFO] Loaded checkpoint from iteration {loaded_dict['iter']}")


def export_policy_jit(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as JIT module."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")
    actor_critic = runner.alg.actor_critic_1 if runner.is_mdpo else runner.alg.actor_critic
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None
    print(f"[INFO] Exporting JIT policy to: {export_dir}")
    actor_critic.export_jit(path=export_dir, filename="policy.pt", normalizer=normalizer)
    print("[INFO] JIT export complete!")


def export_policy_onnx(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as ONNX model."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")
    actor_critic = runner.alg.actor_critic_1 if runner.is_mdpo else runner.alg.actor_critic
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    if not hasattr(actor_critic, "export_onnx"):
        raise NotImplementedError(
            f"ONNX export not implemented for {type(actor_critic).__name__}. "
            "Please add an export_onnx method to this module."
        )

    print(f"[INFO] Exporting ONNX policy to: {export_dir}")
    actor_critic.export_onnx(path=export_dir, filename="policy.onnx", normalizer=normalizer)
    print("[INFO] ONNX export complete!")


def configure_night_scene(env_cfg: ManagerBasedRLEnvCfg):
    """Apply visual-only night settings without changing task physics or action/obs behavior."""
    env_cfg.scene.sky_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
    env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.01, 0.01, 0.01),
    )


def _make_local_subterrain_cfg(
    base_cfg: HfMazeTerrainCfg,
    *,
    proportion: float | None = None,
    force_flat: bool = False,
) -> HfMazeTerrainCfg:
    """Create a fresh local sub-terrain config instead of reusing the task-owned one."""
    grid_size = (
        (args_cli.grid_size, args_cli.grid_size)
        if args_cli.grid_size is not None
        else tuple(getattr(base_cfg, "grid_size", (15, 15)))
    )
    cell_size = args_cli.cell_size if args_cli.cell_size is not None else float(getattr(base_cfg, "cell_size", 2.0))
    wall_ratio = (
        args_cli.wall_ratio if args_cli.wall_ratio is not None else float(getattr(base_cfg, "random_wall_ratio", 0.5))
    )

    cfg = HfMazeTerrainCfg(
        proportion=float(getattr(base_cfg, "proportion", 1.0) if proportion is None else proportion),
        open_probability=float(getattr(base_cfg, "open_probability", 0.9)),
        grid_size=grid_size,
        cell_size=cell_size,
        wall_height=float(getattr(base_cfg, "wall_height", 1.5)),
        goal_padding_cells=getattr(base_cfg, "goal_padding_cells", None),
        spawn_padding_cells=getattr(base_cfg, "spawn_padding_cells", None),
        add_goal=True,
        add_noise_to_flat=bool(getattr(base_cfg, "add_noise_to_flat", False)),
        randomize_wall=bool(getattr(base_cfg, "randomize_wall", True)),
        random_wall_ratio=wall_ratio,
        non_maze_terrain=bool(getattr(base_cfg, "non_maze_terrain", False)),
        stairs=bool(getattr(base_cfg, "stairs", False)),
        add_stairs_to_maze=bool(getattr(base_cfg, "add_stairs_to_maze", False)),
        dynamic_obstacles=bool(getattr(base_cfg, "dynamic_obstacles", False)),
    )

    if force_flat:
        cfg.open_probability = 1.0
        cfg.randomize_wall = False
        cfg.random_wall_ratio = 0.0
        cfg.non_maze_terrain = True
        cfg.stairs = False
        cfg.add_stairs_to_maze = False
        cfg.dynamic_obstacles = False

    return cfg


def _build_local_subterrain_map(base_tg: TerrainGeneratorCfg) -> dict[str, HfMazeTerrainCfg]:
    """Create local terrain generator entries matching the current CLI selection."""
    base_sub_terrains = base_tg.sub_terrains
    default_base = next(iter(base_sub_terrains.values()))

    def _base_or_default(name: str) -> HfMazeTerrainCfg:
        return base_sub_terrains.get(name, default_base)

    if args_cli.terrain_type in (None, "both"):
        local_sub_terrains: dict[str, HfMazeTerrainCfg] = {}
        for name, base_cfg in base_sub_terrains.items():
            local_sub_terrains[name] = _make_local_subterrain_cfg(base_cfg)
        return local_sub_terrains

    if args_cli.terrain_type == "flat":
        return {"flat": _make_local_subterrain_cfg(_base_or_default("non_maze"), proportion=1.0, force_flat=True)}

    return {
        args_cli.terrain_type: _make_local_subterrain_cfg(_base_or_default(args_cli.terrain_type), proportion=1.0)
    }


def build_standalone_terrain_cfg(env_cfg: ManagerBasedRLEnvCfg) -> TerrainImporterCfg:
    """Build the terrain locally in this script while keeping the rest of the task env unchanged."""
    base_terrain_cfg = env_cfg.scene.terrain
    base_tg = base_terrain_cfg.terrain_generator
    if base_tg is None:
        raise ValueError("Standalone terrain override requires a generator-based terrain config.")

    terrain_size = args_cli.terrain_size if args_cli.terrain_size is not None else float(base_tg.size[0])
    terrain_size_xy = (terrain_size, terrain_size)
    num_rows = args_cli.terrain_rows if args_cli.terrain_rows is not None else int(base_tg.num_rows)
    num_cols = args_cli.terrain_cols if args_cli.terrain_cols is not None else int(base_tg.num_cols)
    difficulty_range = (
        tuple(args_cli.difficulty)
        if args_cli.difficulty is not None
        else tuple(getattr(base_tg, "difficulty_range", (0.5, 1.0)))
    )

    max_init_terrain_level = base_terrain_cfg.max_init_terrain_level
    if max_init_terrain_level is not None:
        max_init_terrain_level = min(int(max_init_terrain_level), max(0, num_rows - 1))

    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode=base_terrain_cfg.physics_material.friction_combine_mode,
        restitution_combine_mode=base_terrain_cfg.physics_material.restitution_combine_mode,
        restitution=base_terrain_cfg.physics_material.restitution,
        static_friction=base_terrain_cfg.physics_material.static_friction,
        dynamic_friction=base_terrain_cfg.physics_material.dynamic_friction,
        compliant_contact_stiffness=base_terrain_cfg.physics_material.compliant_contact_stiffness,
        compliant_contact_damping=base_terrain_cfg.physics_material.compliant_contact_damping,
    )

    return TerrainImporterCfg(
        prim_path=base_terrain_cfg.prim_path,
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=terrain_size_xy,
            border_width=float(base_tg.border_width),
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_scale=float(base_tg.horizontal_scale),
            vertical_scale=float(base_tg.vertical_scale),
            slope_threshold=float(base_tg.slope_threshold),
            use_cache=False,
            curriculum=bool(getattr(base_tg, "curriculum", False)),
            difficulty_range=difficulty_range,
            sub_terrains=_build_local_subterrain_map(base_tg),
        ),
        max_init_terrain_level=max_init_terrain_level,
        collision_group=base_terrain_cfg.collision_group,
        physics_material=physics_material,
        visual_material=base_terrain_cfg.visual_material,
        debug_vis=base_terrain_cfg.debug_vis,
    )


class StandaloneRobotGoalCommand(RobotNavigationGoalCommand):
    """Local goal/spawn sampler that teleports the robot to the sampled spawn immediately on reset."""

    def _initialize_position_sampling(self):
        if self._sampling_initialized:
            return

        terrain = self.env.scene.terrain
        heights_raw = getattr(terrain, "_height_field_visual", None)
        valid_mask_raw = getattr(terrain, "_height_field_valid_mask", None)
        platform_mask_raw = getattr(terrain, "_height_field_platform_mask", None)
        spawn_mask_raw = getattr(terrain, "_height_field_spawn_mask", None)

        if heights_raw is None or valid_mask_raw is None:
            raise ValueError(
                "Standalone terrain did not publish height-field sampling data. "
                "Ensure add_goal=True is enabled in the local terrain config."
            )

        heights = heights_raw.to(self.device)
        valid_mask = valid_mask_raw.to(self.device)
        platform_mask = platform_mask_raw.to(self.device) if platform_mask_raw is not None else torch.zeros_like(valid_mask)
        spawn_mask = spawn_mask_raw.to(self.device) if spawn_mask_raw is not None else valid_mask

        terrain_cfg = terrain.cfg.terrain_generator
        self._position_sampler = PositionSampler(
            heights=heights,
            valid_mask=valid_mask,
            platform_mask=platform_mask,
            terrain_size=self.terrain_size,
            horizontal_scale=float(terrain_cfg.horizontal_scale),
            device=self.device,
            spawn_mask=spawn_mask,
            border_width=0.0,
        )
        self._sampling_initialized = True

    def _write_robot_spawn_state(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Write the sampled spawn immediately so reset uses the current episode's spawn, not the previous one."""
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        spawn_positions = default_root_state[:, :3] + self.env.scene.env_origins[env_ids]
        spawn_yaw = self.spawn_heading_world[env_ids]
        zero = torch.zeros_like(spawn_yaw)
        yaw_delta = math_utils.quat_from_euler_xyz(zero, zero, spawn_yaw)
        orientations = math_utils.quat_mul(default_root_state[:, 3:7], yaw_delta)
        velocities = default_root_state[:, 7:13]

        self.robot.write_root_pose_to_sim(torch.cat([spawn_positions, orientations], dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)
        return spawn_positions

    def _resample_command(self, env_ids: Sequence[int]):
        self._initialize_position_sampling()

        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.clone().to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if env_ids_tensor.numel() == 0:
            return

        self._reset_tracking_state(env_ids_tensor)
        terrain_indices = self._get_terrain_indices(env_ids_tensor)

        goal_x, goal_y, goal_z = self._position_sampler.sample(terrain_indices)
        spawn_x, spawn_y, spawn_z = self._position_sampler.sample_spawn(terrain_indices)

        min_spawn_goal_distance = getattr(self.cfg, "min_spawn_goal_distance", 0.0)
        if min_spawn_goal_distance > 0.0:
            attempts = max(1, int(getattr(self.cfg, "spawn_goal_resample_attempts", 32)))
            invalid = torch.sqrt((goal_x - spawn_x) ** 2 + (goal_y - spawn_y) ** 2) < min_spawn_goal_distance
            for _ in range(attempts):
                if not invalid.any():
                    break
                invalid_indices = invalid.nonzero(as_tuple=False).squeeze(-1)
                resampled_goal_x, resampled_goal_y, resampled_goal_z = self._position_sampler.sample(
                    terrain_indices[invalid_indices]
                )
                goal_x[invalid_indices] = resampled_goal_x
                goal_y[invalid_indices] = resampled_goal_y
                goal_z[invalid_indices] = resampled_goal_z
                invalid = torch.sqrt((goal_x - spawn_x) ** 2 + (goal_y - spawn_y) ** 2) < min_spawn_goal_distance

        terrain = self.env.scene.terrain
        levels = terrain.terrain_levels[env_ids_tensor]
        types = terrain.terrain_types[env_ids_tensor]
        terrain_origins = terrain.terrain_origins[levels, types]

        self.goal_position_world[env_ids_tensor, 0] = terrain_origins[:, 0] + goal_x
        self.goal_position_world[env_ids_tensor, 1] = terrain_origins[:, 1] + goal_y
        height_offset = torch.rand(len(env_ids_tensor), device=self.device) * 0.6 + 0.2
        self.goal_position_world[env_ids_tensor, 2] = goal_z + height_offset
        self.goal_heading_world[env_ids_tensor] = torch.empty(len(env_ids_tensor), device=self.device).uniform_(
            -math.pi, math.pi
        )

        spawn_offset = 0.05
        terrain.env_origins[env_ids_tensor, 0] = terrain_origins[:, 0] + spawn_x
        terrain.env_origins[env_ids_tensor, 1] = terrain_origins[:, 1] + spawn_y
        terrain.env_origins[env_ids_tensor, 2] = spawn_z + spawn_offset

        self.spawn_position_world[env_ids_tensor, 0] = terrain_origins[:, 0] + spawn_x
        self.spawn_position_world[env_ids_tensor, 1] = terrain_origins[:, 1] + spawn_y
        self.spawn_position_world[env_ids_tensor, 2] = spawn_z + spawn_offset
        self.spawn_heading_world[env_ids_tensor] = torch.empty(len(env_ids_tensor), device=self.device).uniform_(
            -math.pi, math.pi
        )

        spawn_positions = self._write_robot_spawn_state(env_ids_tensor)
        self.previous_position[env_ids_tensor] = spawn_positions.clone()
        self.initial_distance_to_goal[env_ids_tensor] = torch.norm(
            spawn_positions - self.goal_position_world[env_ids_tensor], dim=1
        )
        self.distance_to_goal[env_ids_tensor] = self.initial_distance_to_goal[env_ids_tensor]
        self.closest_distance_to_goal[env_ids_tensor] = self.initial_distance_to_goal[env_ids_tensor]


@configclass
class StandaloneRobotGoalCommandCfg(RobotNavigationGoalCommandCfg):
    class_type: type = StandaloneRobotGoalCommand


def standalone_time_out_navigation(
    env,
    goal_cmd_name: str = "robot_goal",
    distance_threshold: float = 0.5,
    yaw_threshold: float | None = None,
    lin_speed_threshold: float | None = None,
    yaw_rate_threshold: float | None = None,
):
    return nav_mdp.time_out_navigation(
        env,
        goal_cmd_name=goal_cmd_name,
        distance_threshold=distance_threshold,
        yaw_threshold=yaw_threshold,
        lin_speed_threshold=lin_speed_threshold,
        yaw_rate_threshold=yaw_rate_threshold,
    )


def standalone_illegal_contact_navigation(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    goal_cmd_name: str = "robot_goal",
    horizontal_only: bool = False,
):
    return nav_mdp.illegal_contact_navigation(
        env,
        threshold=threshold,
        sensor_cfg=sensor_cfg,
        goal_cmd_name=goal_cmd_name,
        horizontal_only=horizontal_only,
    )


def standalone_large_angle_termination_navigation(
    env,
    threshold: float,
    goal_cmd_name: str = "robot_goal",
):
    return nav_mdp.large_angle_termination_navigation(
        env,
        threshold=threshold,
        goal_cmd_name=goal_cmd_name,
    )


def standalone_in_goal_navigation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
    goal_cmd_name: str = "robot_goal",
    yaw_threshold: float | None = None,
    lin_speed_threshold: float | None = None,
    yaw_rate_threshold: float | None = None,
):
    return nav_mdp.in_goal_navigation(
        env,
        asset_cfg=asset_cfg,
        distance_threshold=distance_threshold,
        goal_cmd_name=goal_cmd_name,
        yaw_threshold=yaw_threshold,
        lin_speed_threshold=lin_speed_threshold,
        yaw_rate_threshold=yaw_rate_threshold,
    )


def standalone_near_goal_navigation(
    env,
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
):
    return nav_mdp.near_goal_navigation(
        env,
        asset_cfg=asset_cfg,
        distance_threshold=distance_threshold,
        goal_cmd_name=goal_cmd_name,
        yaw_threshold=yaw_threshold,
        lin_speed_threshold=lin_speed_threshold,
        yaw_rate_threshold=yaw_rate_threshold,
        in_goal_distance_threshold=in_goal_distance_threshold,
        in_goal_yaw_threshold=in_goal_yaw_threshold,
        in_goal_lin_speed_threshold=in_goal_lin_speed_threshold,
        in_goal_yaw_rate_threshold=in_goal_yaw_rate_threshold,
        hold_time_s=hold_time_s,
    )


def standalone_trapped_navigation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cmd_name: str = "robot_goal",
    near_goal_distance_threshold: float = 0.6,
    near_goal_yaw_threshold: float | None = None,
    near_goal_lin_speed_threshold: float | None = None,
    near_goal_yaw_rate_threshold: float | None = None,
    window_s: float = 20.0,
    single_cell_time_s: float = 4.0,
    top_two_cell_time_s: float = 8.0,
):
    return nav_mdp.trapped_navigation(
        env,
        asset_cfg=asset_cfg,
        goal_cmd_name=goal_cmd_name,
        near_goal_distance_threshold=near_goal_distance_threshold,
        near_goal_yaw_threshold=near_goal_yaw_threshold,
        near_goal_lin_speed_threshold=near_goal_lin_speed_threshold,
        near_goal_yaw_rate_threshold=near_goal_yaw_rate_threshold,
        window_s=window_s,
        single_cell_time_s=single_cell_time_s,
        top_two_cell_time_s=top_two_cell_time_s,
    )


def standalone_terrain_fall(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fall_height_threshold: float = -1.0,
    goal_cmd_name: str = "robot_goal",
):
    return nav_mdp.terrain_fall(
        env,
        asset_cfg=asset_cfg,
        fall_height_threshold=fall_height_threshold,
        goal_cmd_name=goal_cmd_name,
    )


def apply_local_navigation_overrides(env_cfg: ManagerBasedRLEnvCfg):
    """Replace terrain generation, spawn/goal sampling, and termination wiring with local standalone versions."""
    env_cfg.scene.terrain = build_standalone_terrain_cfg(env_cfg)
    env_cfg.sim.physics_material = env_cfg.scene.terrain.physics_material

    current_goal_cfg = env_cfg.commands.robot_goal
    env_cfg.commands.robot_goal = StandaloneRobotGoalCommandCfg(
        asset_name=current_goal_cfg.asset_name,
        resampling_time_range=current_goal_cfg.resampling_time_range,
        debug_vis=current_goal_cfg.debug_vis,
        robot_to_goal_line_vis=getattr(current_goal_cfg, "robot_to_goal_line_vis", True),
        min_spawn_goal_distance=getattr(current_goal_cfg, "min_spawn_goal_distance", 0.0),
        spawn_goal_resample_attempts=getattr(current_goal_cfg, "spawn_goal_resample_attempts", 32),
    )

    termination_overrides = {
        "time_out": standalone_time_out_navigation,
        "base_contact": standalone_illegal_contact_navigation,
        "large_pitch_angle": standalone_large_angle_termination_navigation,
        "in_goal": standalone_in_goal_navigation,
        "near_goal": standalone_near_goal_navigation,
        "trapped": standalone_trapped_navigation,
        "terrain_fall": standalone_terrain_fall,
    }
    for term_name, term_func in termination_overrides.items():
        term_cfg = getattr(env_cfg.terminations, term_name, None)
        if term_cfg is not None:
            term_cfg.func = term_func


def apply_cli_overrides(env_cfg: ManagerBasedRLEnvCfg):
    """Apply the same task config overrides supported by play.py/play_night.py."""
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.episode_length is not None:
        env_cfg.episode_length_s = args_cli.episode_length

    if args_cli.policy_scale_yaw is not None:
        action_cfg = getattr(env_cfg.actions, "velocity_command", None)
        if action_cfg is None or not hasattr(action_cfg, "policy_scaling"):
            raise AttributeError("This task does not expose actions.velocity_command.policy_scaling")
        policy_scaling = list(action_cfg.policy_scaling)
        if len(policy_scaling) < 3:
            raise ValueError(
                f"actions.velocity_command.policy_scaling must have at least 3 entries, got {policy_scaling}"
            )
        old_yaw_scale = policy_scaling[2]
        policy_scaling[2] = args_cli.policy_scale_yaw
        action_cfg.policy_scaling = policy_scaling
        print(f"[INFO] Overriding yaw policy scale from {old_yaw_scale} to {args_cli.policy_scale_yaw}")

    if args_cli.decimation is not None:
        if args_cli.decimation <= 0:
            raise ValueError(f"--decimation must be >= 1, got {args_cli.decimation}")
        old_decimation = env_cfg.decimation
        env_cfg.decimation = args_cli.decimation
        if hasattr(env_cfg.scene, "height_scanner_critic") and env_cfg.scene.height_scanner_critic is not None:
            env_cfg.scene.height_scanner_critic.update_period = env_cfg.decimation * env_cfg.sim.dt
        if hasattr(env_cfg.scene, "raycast_camera") and env_cfg.scene.raycast_camera is not None:
            env_cfg.scene.raycast_camera.update_period = env_cfg.decimation * env_cfg.sim.dt
        old_policy_hz = 1.0 / (env_cfg.sim.dt * old_decimation)
        new_policy_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
        print(
            f"[INFO] Overriding policy decimation from {old_decimation} to {env_cfg.decimation} "
            f"({old_policy_hz:.1f}Hz -> {new_policy_hz:.1f}Hz)"
        )

    if args_cli.debug_raycast and hasattr(env_cfg.scene, "raycast_camera") and env_cfg.scene.raycast_camera is not None:
        env_cfg.scene.raycast_camera.debug_vis = True

    apply_local_navigation_overrides(env_cfg)


def depth_to_colormap(depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    """Convert depth to a colorized image for OpenCV display."""
    clean = np.nan_to_num(depth, nan=depth_max, posinf=depth_max, neginf=0.0)
    depth_norm = np.clip((clean - depth_min) / max(depth_max - depth_min, 1e-6), 0.0, 1.0)
    depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    if cv2 is None:
        return depth_u8
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def setup_depth_window():
    """Create the OpenCV window if depth visualization is enabled."""
    if not args_cli.show_depth:
        return
    if cv2 is None:
        print("[WARN] OpenCV is not installed; disabling depth window.")
        args_cli.show_depth = False
        return
    if getattr(args_cli, "headless", False):
        print("[WARN] Headless mode requested; disabling depth window.")
        args_cli.show_depth = False
        return

    display_scale = max(1, int(args_cli.depth_display_scale) or 8)
    cv2.namedWindow("Depth Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Camera", 64 * display_scale, 40 * display_scale)
    cv2.moveWindow("Depth Camera", 50, 50)


def update_depth_window(depth_camera: Any, env_idx: int):
    """Render the selected raw depth image in OpenCV."""
    if not args_cli.show_depth or cv2 is None:
        return

    depth_data = depth_camera.data.output.get("distance_to_image_plane")
    if depth_data is None or depth_data.numel() == 0:
        return

    if env_idx < 0 or env_idx >= depth_data.shape[0]:
        return

    depth = depth_data[env_idx].detach().cpu().numpy()
    if depth.ndim == 3:
        depth = depth.squeeze(-1)

    color = depth_to_colormap(depth, args_cli.depth_min, args_cli.depth_max)
    if args_cli.depth_display_scale and args_cli.depth_display_scale > 0:
        display_scale = int(args_cli.depth_display_scale)
    else:
        display_scale = max(1, min(20, int(round(720 / max(color.shape[0], 1)))))
    if display_scale > 1:
        color = cv2.resize(
            color,
            (color.shape[1] * display_scale, color.shape[0] * display_scale),
            interpolation=cv2.INTER_NEAREST,
        )
    cv2.imshow("Depth Camera", color)
    cv2.waitKey(1)


def main():
    """Play the real task env with night visuals and live depth display."""
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")

    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    agent_cfg = agent_cfg_class()

    configure_night_scene(env_cfg)
    apply_cli_overrides(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = SruRslRlVecEnvWrapper(env)

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = find_latest_checkpoint(log_root_path, checkpoint_pattern="model_.*.pt")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_checkpoint_with_fallback(runner, resume_path)

    if args_cli.export_jit:
        export_policy_jit(runner, resume_path)
    if args_cli.export_onnx:
        export_policy_onnx(runner, resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.get_observations()

    raycast_camera = env.unwrapped.scene.sensors.get("raycast_camera")
    robot_goal_term = env.unwrapped.command_manager._terms.get("robot_goal")
    terrain_generator = env_cfg.scene.terrain.terrain_generator
    setup_depth_window()

    physics_hz = 1.0 / env_cfg.sim.dt
    policy_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    camera_period = None if raycast_camera is None else raycast_camera.cfg.update_period
    camera_hz = None if camera_period in (None, 0.0) else 1.0 / camera_period
    print("[INFO] Using the real task env/action/observation pipeline from play.py")
    print("[INFO] Terrain/spawn/termination: local standalone overrides")
    if terrain_generator is not None:
        print(
            f"[INFO] Terrain grid: {terrain_generator.num_rows}x{terrain_generator.num_cols}; "
            f"sub-terrains: {', '.join(terrain_generator.sub_terrains.keys())}"
        )
    if robot_goal_term is not None:
        print(f"[INFO] Goal command term: {type(robot_goal_term).__name__}")
    print(f"[INFO] Physics: {physics_hz:.1f} Hz")
    print(f"[INFO] Policy:  {policy_hz:.1f} Hz")
    if camera_period is None:
        print("[INFO] Camera:  unavailable")
    elif camera_period == 0.0:
        print("[INFO] Camera:  every render/update (update_period=0.0)")
    else:
        print(f"[INFO] Camera:  {camera_hz:.1f} Hz (update_period={camera_period:.3f}s)")

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        if raycast_camera is not None and step_count % max(args_cli.depth_vis_every, 1) == 0:
            update_depth_window(raycast_camera, args_cli.depth_env_idx)
        step_count += 1

    env.close()
    if cv2 is not None and args_cli.show_depth:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
