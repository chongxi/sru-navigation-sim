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


    python scripts/play_with_depth_night.py   \
      --checkpoint logs/rsl_rl/diff_drive_navigation_mdpo/2026-03-30_22-01-17/model_1800.pt \
      --usd /home/chongxi/Work/Astera/navigation/sru-navigation-sim/Astera_upd.usd \
      --robot_scale 0.95 \
      --decimation 6      
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
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
parser.add_argument(
    "--usd",
    type=str,
    default=None,
    help="Optional USD scene path. If omitted, keep the current generated-terrain flow.",
)
parser.add_argument(
    "--robot_scale",
    type=float,
    default=1.0,
    help="Uniform scale factor for the robot before env creation. 0.9 makes it 10% smaller.",
)
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use last checkpoint from logs.")
parser.add_argument("--export_jit", action="store_true", default=False, help="Export policy as JIT module.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--terrain_rows", type=int, default=None, help="Terrain grid rows (difficulty levels).")
parser.add_argument("--terrain_cols", type=int, default=None, help="Terrain grid columns (variations).")
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    choices=["maze", "non_maze", "stairs", "pits", "mixed", "both", "flat"],
    help="Drive-style terrain preset. Defaults to maze when omitted. Use mixed/both for the full preset mix.",
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
parser.add_argument("--terrain_border_width", type=float, default=20.0, help="Outer border width around the terrain grid.")
parser.add_argument("--open_probability", type=float, default=0.9, help="Open-cell probability for maze generation.")
parser.add_argument("--wall_height", type=float, default=1.5, help="Wall height in meters.")
parser.add_argument(
    "--goal_padding_cells",
    type=int,
    default=None,
    help="Optional obstacle padding for goal sampling in height-field cells.",
)
parser.add_argument(
    "--spawn_padding_cells",
    type=int,
    default=None,
    help="Optional obstacle padding for spawn sampling in height-field cells.",
)
parser.add_argument("--ground_static_friction", type=float, default=1.0, help="Ground static friction.")
parser.add_argument("--ground_dynamic_friction", type=float, default=0.9, help="Ground dynamic friction.")
parser.add_argument("--ground_restitution", type=float, default=0.0, help="Ground restitution.")
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
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster.ray_caster import RayCaster
from isaaclab.utils import configclass
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape
from isaaclab.utils.warp import convert_to_warp_mesh
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401
import isaaclab_nav_task.navigation.mdp as nav_mdp

from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import PositionSampler, RobotNavigationGoalCommand
from isaaclab_nav_task.navigation.mdp.navigation.goal_commands_cfg import RobotNavigationGoalCommandCfg
from isaaclab_nav_task.terrains import HfMazeTerrainCfg
from isaaclab_nav_task.terrains.terrain_constants import HORIZONTAL_SCALE, PADDING
from isaaclab_nav_task.vecenv_wrapper import SruRslRlVecEnvWrapper

from generate_terrain_usd import build_navigation_terrain_from_usd_stage


try:
    import cv2
except ImportError:
    cv2 = None


RAYCAST_MERGED_MESH_PATH = "/World/RaycastSceneMerged"
RAYCAST_GEOM_TYPES = set(PRIMITIVE_MESH_TYPES) | {"Mesh"}
ASTERA_UPD_GOAL_LOCATIONS = (
    (2.01, -4.33, 0.04),
    (9.68, -0.626, 0.04),
    (-12.72, -3.69, 0.04),
    (-0.45, 4.69, 0.04),
)


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


def _terrain_grid_size() -> tuple[int, int]:
    grid_dim = args_cli.grid_size if args_cli.grid_size is not None else 15
    return (grid_dim, grid_dim)


def _default_wall_ratio(terrain_name: str) -> float:
    if args_cli.wall_ratio is not None:
        return args_cli.wall_ratio
    return {
        "maze": 0.5,
        "non_maze": 1.0,
        "pits": 1.0,
        "stairs": 1.0,
        "flat": 0.0,
    }.get(terrain_name, 1.0)


def _build_drive_style_subterrain(terrain_name: str, proportion: float) -> HfMazeTerrainCfg:
    """Build one standalone terrain preset using the same style as drive_terrain_pid_depth.py."""
    grid_size = _terrain_grid_size()
    cell_size = args_cli.cell_size if args_cli.cell_size is not None else 2.0
    wall_ratio = _default_wall_ratio(terrain_name)

    cfg = HfMazeTerrainCfg(
        proportion=proportion,
        open_probability=args_cli.open_probability,
        grid_size=grid_size,
        cell_size=cell_size,
        wall_height=args_cli.wall_height,
        goal_padding_cells=args_cli.goal_padding_cells,
        spawn_padding_cells=args_cli.spawn_padding_cells,
        add_goal=True,
        add_noise_to_flat=False,
        randomize_wall=True,
        random_wall_ratio=wall_ratio,
        non_maze_terrain=False,
        stairs=False,
        add_stairs_to_maze=False,
        dynamic_obstacles=False,
    )

    if terrain_name == "non_maze":
        cfg.non_maze_terrain = True
    elif terrain_name == "stairs":
        cfg.randomize_wall = False
        cfg.stairs = True
    elif terrain_name == "pits":
        cfg.non_maze_terrain = True
        cfg.dynamic_obstacles = True
    elif terrain_name == "flat":
        cfg.open_probability = 1.0
        cfg.randomize_wall = False
        cfg.random_wall_ratio = 0.0
        cfg.non_maze_terrain = True

    return cfg


def _build_local_subterrain_map() -> tuple[str, dict[str, HfMazeTerrainCfg]]:
    """Create drive-style sub-terrain presets for the standalone play script."""
    selection = args_cli.terrain_type or "maze"

    if selection in ("mixed", "both"):
        return (
            "mixed",
            {
                "maze": _build_drive_style_subterrain("maze", 0.3),
                "non_maze": _build_drive_style_subterrain("non_maze", 0.2),
                "stairs": _build_drive_style_subterrain("stairs", 0.3),
                "pits": _build_drive_style_subterrain("pits", 0.2),
            },
        )

    return selection, {selection: _build_drive_style_subterrain(selection, 1.0)}


def _build_ground_physics_material() -> sim_utils.RigidBodyMaterialCfg:
    """Build the floor material shared by generated terrain and USD terrain."""
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="multiply",
        restitution=args_cli.ground_restitution,
        static_friction=args_cli.ground_static_friction,
        dynamic_friction=args_cli.ground_dynamic_friction,
        compliant_contact_stiffness=5e5,
        compliant_contact_damping=300.0,
    )


def build_standalone_terrain_cfg(env_cfg: ManagerBasedRLEnvCfg) -> TerrainImporterCfg:
    """Build the terrain locally in this script while keeping the rest of the task env unchanged."""
    base_terrain_cfg = env_cfg.scene.terrain
    base_tg = base_terrain_cfg.terrain_generator
    if base_tg is None:
        raise ValueError("Standalone terrain override requires a generator-based terrain config.")

    terrain_size = args_cli.terrain_size if args_cli.terrain_size is not None else 30.0
    terrain_size_xy = (terrain_size, terrain_size)
    num_rows = args_cli.terrain_rows if args_cli.terrain_rows is not None else 1
    num_cols = args_cli.terrain_cols if args_cli.terrain_cols is not None else 2
    difficulty_range = (
        tuple(args_cli.difficulty)
        if args_cli.difficulty is not None
        else (0.5, 1.0)
    )
    terrain_label, sub_terrains = _build_local_subterrain_map()

    max_init_terrain_level = base_terrain_cfg.max_init_terrain_level
    if max_init_terrain_level is not None:
        max_init_terrain_level = min(int(max_init_terrain_level), max(0, num_rows - 1))

    physics_material = _build_ground_physics_material()

    print(
        f"[INFO] Local terrain preset: {terrain_label}; grid={num_rows}x{num_cols}; "
        f"tile_size={terrain_size_xy[0]:.1f}m; difficulty={difficulty_range}"
    )

    return TerrainImporterCfg(
        prim_path=base_terrain_cfg.prim_path,
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=terrain_size_xy,
            border_width=args_cli.terrain_border_width,
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            difficulty_range=difficulty_range,
            sub_terrains=sub_terrains,
        ),
        max_init_terrain_level=max_init_terrain_level,
        collision_group=base_terrain_cfg.collision_group,
        physics_material=physics_material,
        visual_material=base_terrain_cfg.visual_material,
        debug_vis=base_terrain_cfg.debug_vis,
    )


def build_standalone_usd_terrain_cfg(env_cfg: ManagerBasedRLEnvCfg) -> TerrainImporterCfg:
    """Reference a USD scene while keeping enough terrain metadata alive for manager construction."""
    if args_cli.usd is None:
        raise ValueError("USD terrain config requested without --usd.")

    base_terrain_cfg = env_cfg.scene.terrain
    physics_material = _build_ground_physics_material()
    usd_path = Path(args_cli.usd).expanduser()
    if not usd_path.is_absolute():
        usd_path = Path.cwd() / usd_path
    if not usd_path.exists():
        raise FileNotFoundError(f"USD scene not found: {usd_path}")

    placeholder_size = args_cli.terrain_size if args_cli.terrain_size is not None else 30.0
    placeholder_cell_size = args_cli.cell_size if args_cli.cell_size is not None else 2.0
    placeholder_grid_dim = max(1, int(math.ceil(placeholder_size / max(placeholder_cell_size, 1.0e-6))))

    print(f"[INFO] Loading scene USD from {usd_path}")

    return TerrainImporterCfg(
        prim_path=base_terrain_cfg.prim_path,
        terrain_type="usd",
        usd_path=str(usd_path),
        terrain_generator=TerrainGeneratorCfg(
            size=(placeholder_size, placeholder_size),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            difficulty_range=(0.0, 0.0),
            sub_terrains={
                "usd_scene": HfMazeTerrainCfg(
                    proportion=1.0,
                    open_probability=1.0,
                    grid_size=(placeholder_grid_dim, placeholder_grid_dim),
                    cell_size=placeholder_cell_size,
                    wall_height=args_cli.wall_height,
                    goal_padding_cells=args_cli.goal_padding_cells,
                    spawn_padding_cells=args_cli.spawn_padding_cells,
                    add_goal=True,
                    add_noise_to_flat=False,
                    randomize_wall=False,
                    random_wall_ratio=0.0,
                    non_maze_terrain=True,
                    stairs=False,
                    add_stairs_to_maze=False,
                    dynamic_obstacles=False,
                )
            },
        ),
        max_init_terrain_level=0,
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

    def _sample_goal_positions_world(
        self,
        env_ids_tensor: torch.Tensor,
        terrain_indices: torch.Tensor,
        terrain_origins: torch.Tensor,
        spawn_positions_world: torch.Tensor,
    ) -> torch.Tensor:
        """Sample goal positions in world coordinates without mutating spawn state."""
        fixed_goal_positions_world = getattr(self, "fixed_goal_positions_world", None)
        if fixed_goal_positions_world is not None:
            fixed_goal_positions_world = fixed_goal_positions_world.to(device=self.device, dtype=torch.float32)
            goal_indices = torch.randint(
                0, fixed_goal_positions_world.shape[0], (len(env_ids_tensor),), device=self.device
            )
            goal_positions_world = fixed_goal_positions_world[goal_indices].clone()

            min_spawn_goal_distance = getattr(self.cfg, "min_spawn_goal_distance", 0.0)
            if min_spawn_goal_distance > 0.0:
                attempts = max(1, int(getattr(self.cfg, "spawn_goal_resample_attempts", 32)))
                invalid = (
                    torch.norm(goal_positions_world[:, :2] - spawn_positions_world[:, :2], dim=1)
                    < min_spawn_goal_distance
                )
                for _ in range(attempts):
                    if not invalid.any():
                        break
                    invalid_indices = invalid.nonzero(as_tuple=False).squeeze(-1)
                    goal_indices[invalid_indices] = torch.randint(
                        0, fixed_goal_positions_world.shape[0], (len(invalid_indices),), device=self.device
                    )
                    goal_positions_world[invalid_indices] = fixed_goal_positions_world[goal_indices[invalid_indices]]
                    invalid = (
                        torch.norm(goal_positions_world[:, :2] - spawn_positions_world[:, :2], dim=1)
                        < min_spawn_goal_distance
                    )
            return goal_positions_world

        goal_x, goal_y, goal_z = self._position_sampler.sample(terrain_indices)
        goal_positions_world = torch.stack(
            [
                terrain_origins[:, 0] + goal_x,
                terrain_origins[:, 1] + goal_y,
                goal_z + (torch.rand(len(env_ids_tensor), device=self.device) * 0.6 + 0.2),
            ],
            dim=-1,
        )

        min_spawn_goal_distance = getattr(self.cfg, "min_spawn_goal_distance", 0.0)
        if min_spawn_goal_distance > 0.0:
            attempts = max(1, int(getattr(self.cfg, "spawn_goal_resample_attempts", 32)))
            invalid = torch.norm(goal_positions_world[:, :2] - spawn_positions_world[:, :2], dim=1) < min_spawn_goal_distance
            for _ in range(attempts):
                if not invalid.any():
                    break
                invalid_indices = invalid.nonzero(as_tuple=False).squeeze(-1)
                resampled_goal_x, resampled_goal_y, resampled_goal_z = self._position_sampler.sample(
                    terrain_indices[invalid_indices]
                )
                goal_positions_world[invalid_indices, 0] = terrain_origins[invalid_indices, 0] + resampled_goal_x
                goal_positions_world[invalid_indices, 1] = terrain_origins[invalid_indices, 1] + resampled_goal_y
                goal_positions_world[invalid_indices, 2] = (
                    resampled_goal_z + (torch.rand(len(invalid_indices), device=self.device) * 0.6 + 0.2)
                )
                invalid = (
                    torch.norm(goal_positions_world[:, :2] - spawn_positions_world[:, :2], dim=1)
                    < min_spawn_goal_distance
                )

        return goal_positions_world

    def resample_goal_only(self, env_ids: Sequence[int]):
        """Sample a new goal while keeping the robot at its current pose."""
        self._initialize_position_sampling()

        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.clone().to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if env_ids_tensor.numel() == 0:
            return

        self._reset_tracking_state(env_ids_tensor)
        terrain = self.env.scene.terrain
        terrain_indices = self._get_terrain_indices(env_ids_tensor)
        levels = terrain.terrain_levels[env_ids_tensor]
        types = terrain.terrain_types[env_ids_tensor]
        terrain_origins = terrain.terrain_origins[levels, types]
        robot_positions_world = self.robot.data.root_pos_w[env_ids_tensor].clone()
        goal_positions_world = self._sample_goal_positions_world(
            env_ids_tensor=env_ids_tensor,
            terrain_indices=terrain_indices,
            terrain_origins=terrain_origins,
            spawn_positions_world=robot_positions_world,
        )

        self.goal_position_world[env_ids_tensor] = goal_positions_world
        self.goal_heading_world[env_ids_tensor] = torch.empty(len(env_ids_tensor), device=self.device).uniform_(
            -math.pi, math.pi
        )
        self.spawn_position_world[env_ids_tensor] = robot_positions_world
        self.spawn_heading_world[env_ids_tensor] = math_utils.euler_xyz_from_quat(
            self.robot.data.root_quat_w[env_ids_tensor]
        )[2]

        self.time_left[env_ids_tensor] = self.time_left[env_ids_tensor].uniform_(*self.cfg.resampling_time_range)
        self.command_counter[env_ids_tensor] += 1
        self.previous_position[env_ids_tensor] = robot_positions_world

        self.initial_distance_to_goal[env_ids_tensor] = torch.norm(
            robot_positions_world - self.goal_position_world[env_ids_tensor], dim=1
        )
        self.distance_to_goal[env_ids_tensor] = self.initial_distance_to_goal[env_ids_tensor]
        self.closest_distance_to_goal[env_ids_tensor] = self.initial_distance_to_goal[env_ids_tensor]
        self._update_command()

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

        spawn_x, spawn_y, spawn_z = self._position_sampler.sample_spawn(terrain_indices)

        terrain = self.env.scene.terrain
        levels = terrain.terrain_levels[env_ids_tensor]
        types = terrain.terrain_types[env_ids_tensor]
        terrain_origins = terrain.terrain_origins[levels, types]
        spawn_positions_world = torch.stack(
            [
                terrain_origins[:, 0] + spawn_x,
                terrain_origins[:, 1] + spawn_y,
                spawn_z,
            ],
            dim=-1,
        )
        goal_positions_world = self._sample_goal_positions_world(
            env_ids_tensor=env_ids_tensor,
            terrain_indices=terrain_indices,
            terrain_origins=terrain_origins,
            spawn_positions_world=spawn_positions_world,
        )

        self.goal_position_world[env_ids_tensor] = goal_positions_world
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


def standalone_ignore_in_goal_reset(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
    goal_cmd_name: str = "robot_goal",
    yaw_threshold: float | None = None,
    lin_speed_threshold: float | None = None,
    yaw_rate_threshold: float | None = None,
):
    """Keep the termination term present for rewards/config lookup, but never reset on success."""
    del asset_cfg, distance_threshold, goal_cmd_name, yaw_threshold, lin_speed_threshold, yaw_rate_threshold
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


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
    if args_cli.usd:
        env_cfg.scene.terrain = build_standalone_usd_terrain_cfg(env_cfg)
    else:
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
        "in_goal": standalone_ignore_in_goal_reset,
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

    if args_cli.robot_scale <= 0.0:
        raise ValueError(f"--robot_scale must be > 0, got {args_cli.robot_scale}")

    if not math.isclose(args_cli.robot_scale, 1.0):
        scale = float(args_cli.robot_scale)
        robot_cfg = getattr(env_cfg.scene, "robot", None)
        if robot_cfg is None or getattr(robot_cfg, "spawn", None) is None:
            raise AttributeError("This task does not expose scene.robot.spawn for scaling.")

        old_spawn_scale = robot_cfg.spawn.scale if robot_cfg.spawn.scale is not None else (1.0, 1.0, 1.0)
        robot_cfg.spawn.scale = tuple(float(value) * scale for value in old_spawn_scale)

        if getattr(robot_cfg, "init_state", None) is not None and getattr(robot_cfg.init_state, "pos", None) is not None:
            robot_cfg.init_state.pos = tuple(float(value) * scale for value in robot_cfg.init_state.pos)

        action_cfg = getattr(env_cfg.actions, "velocity_command", None)
        if action_cfg is not None:
            if hasattr(action_cfg, "wheel_radius"):
                action_cfg.wheel_radius = float(action_cfg.wheel_radius) * scale
            if hasattr(action_cfg, "wheel_track"):
                action_cfg.wheel_track = float(action_cfg.wheel_track) * scale

        raycast_camera_cfg = getattr(env_cfg.scene, "raycast_camera", None)
        if raycast_camera_cfg is not None and getattr(raycast_camera_cfg, "offset", None) is not None:
            raycast_camera_cfg.offset.pos = tuple(float(value) * scale for value in raycast_camera_cfg.offset.pos)

        print(
            f"[INFO] Scaling robot by {scale:.3f}x: "
            f"spawn.scale={robot_cfg.spawn.scale}"
        )

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


def _path_has_name_token(path: str, token: str) -> bool:
    return token in path.strip("/").split("/")


def _should_skip_raycast_geom(path: str) -> bool:
    return (
        _path_has_name_token(path, "Robot")
        or _path_has_name_token(path, "NvbloxMesh")
        or path == RAYCAST_MERGED_MESH_PATH
    )


def _is_collision_enabled(prim) -> bool:
    collision_api = UsdPhysics.CollisionAPI(prim)
    if not collision_api:
        return False
    collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
    if not collision_enabled_attr:
        return True
    value = collision_enabled_attr.Get()
    return True if value is None else bool(value)


def _has_collision_on_self_or_ancestor(prim) -> bool:
    current = prim
    while current and current.IsValid():
        if _is_collision_enabled(current):
            return True
        current = current.GetParent()
    return False


def _collect_raycast_geom_prims():
    stage = sim_utils.get_current_stage()
    geom_prims = []
    collider_geom_prims = []

    for prim in stage.Traverse():
        if not prim.IsValid() or not prim.IsActive():
            continue
        if prim.GetTypeName() not in RAYCAST_GEOM_TYPES:
            continue
        path = prim.GetPath().pathString
        if _should_skip_raycast_geom(path):
            continue
        geom_prims.append(prim)
        if _has_collision_on_self_or_ancestor(prim):
            collider_geom_prims.append(prim)

    return collider_geom_prims if collider_geom_prims else geom_prims


def _build_merged_raycast_mesh() -> tuple[str | None, int]:
    stage = sim_utils.get_current_stage()
    geom_prims = _collect_raycast_geom_prims()
    if not geom_prims:
        print("[WARN] No scene geometry found for merged raycast mesh.")
        return None, 0

    merged_vertices = []
    merged_faces = []
    vertex_offset = 0

    for prim in geom_prims:
        if prim.GetTypeName() == "Mesh":
            mesh = create_trimesh_from_geom_mesh(prim)
        else:
            mesh = create_trimesh_from_geom_shape(prim)

        scale = sim_utils.resolve_prim_scale(prim)
        mesh.apply_scale(scale)

        translation, quat_wxyz = sim_utils.resolve_prim_pose(prim)
        rotation = math_utils.matrix_from_quat(torch.tensor(quat_wxyz, dtype=torch.float32)).cpu().numpy()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(translation, dtype=np.float64)
        mesh.apply_transform(transform)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            continue

        merged_vertices.append(vertices)
        merged_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)

    if not merged_vertices or not merged_faces:
        print("[WARN] Raycast merge produced no triangles.")
        return None, 0

    points = np.concatenate(merged_vertices, axis=0)
    faces = np.concatenate(merged_faces, axis=0)
    counts = np.full((faces.shape[0],), 3, dtype=np.int32)

    if stage.GetPrimAtPath(RAYCAST_MERGED_MESH_PATH):
        stage.RemovePrim(RAYCAST_MERGED_MESH_PATH)

    merged_mesh = UsdGeom.Mesh.Define(stage, RAYCAST_MERGED_MESH_PATH)
    merged_mesh.GetPointsAttr().Set(points.tolist())
    merged_mesh.GetFaceVertexCountsAttr().Set(counts.tolist())
    merged_mesh.GetFaceVertexIndicesAttr().Set(faces.reshape(-1).tolist())
    merged_mesh.GetSubdivisionSchemeAttr().Set("none")
    extent = [
        tuple(points.min(axis=0).tolist()),
        tuple(points.max(axis=0).tolist()),
    ]
    merged_mesh.GetExtentAttr().Set(extent)
    UsdGeom.Imageable(merged_mesh.GetPrim()).MakeInvisible()

    print(
        f"[INFO] Built merged raycast mesh at {RAYCAST_MERGED_MESH_PATH} "
        f"from {len(geom_prims)} source prims ({len(points)} vertices, {len(faces)} triangles)."
    )
    return RAYCAST_MERGED_MESH_PATH, len(geom_prims)


def configure_raycast_camera_with_merged_scene(env) -> Any:
    scene = env.unwrapped.scene
    raycast_camera = scene.sensors.get("raycast_camera")
    if raycast_camera is None:
        print("[WARN] No raycast_camera sensor found; skipping merged raycast setup.")
        return None

    merged_mesh_path, num_sources = _build_merged_raycast_mesh()
    if merged_mesh_path is None:
        return raycast_camera

    new_cfg = raycast_camera.cfg.copy()
    new_cfg.mesh_prim_paths = [merged_mesh_path]

    old_camera = scene.sensors["raycast_camera"]
    old_camera._clear_callbacks()
    new_camera = new_cfg.class_type(new_cfg)
    new_camera._initialize_callback(None)
    new_camera.reset()
    scene.sensors["raycast_camera"] = new_camera
    del old_camera

    print(
        f"[INFO] Retargeted raycast_camera to merged scene mesh at {merged_mesh_path} "
        f"(excluded NvbloxMesh, merged {num_sources} source prims)."
    )
    return new_camera


def _rebind_raycaster_sensor_to_mesh(env, sensor_name: str, mesh_prim_path: str) -> Any:
    scene = env.unwrapped.scene
    sensor = scene.sensors.get(sensor_name)
    if sensor is None:
        return None

    stage = sim_utils.get_current_stage()
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    if not mesh_prim.IsValid():
        raise RuntimeError(f"Merged raycast mesh prim not found at {mesh_prim_path}")

    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    if points.size == 0 or indices.size == 0:
        raise RuntimeError(f"Merged raycast mesh at {mesh_prim_path} has no triangles.")

    RayCaster.meshes[mesh_prim_path] = convert_to_warp_mesh(points, indices, device=sensor.device)
    sensor.cfg.mesh_prim_paths = [mesh_prim_path]
    sensor.reset()
    return sensor


def _estimate_robot_footprint_radius(env) -> tuple[float, tuple[float, float]]:
    robot_prim = sim_utils.find_first_matching_prim("/World/envs/env_.*/Robot")
    if robot_prim is None or not robot_prim.IsValid():
        fallback_radius = 0.35
        fallback_dims = (2.0 * fallback_radius, 2.0 * fallback_radius)
        return fallback_radius, fallback_dims

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    bbox = bbox_cache.ComputeWorldBound(robot_prim).GetRange()
    size_x = float(bbox.GetMax()[0] - bbox.GetMin()[0])
    size_y = float(bbox.GetMax()[1] - bbox.GetMin()[1])
    radius = 0.5 * math.hypot(size_x, size_y)
    return radius, (size_x, size_y)


def _derive_usd_padding_cells(env, horizontal_scale: float) -> tuple[int, int, float, tuple[float, float]]:
    robot_radius_m, footprint_dims = _estimate_robot_footprint_radius(env)

    if args_cli.goal_padding_cells is not None:
        goal_padding_cells = int(args_cli.goal_padding_cells)
    else:
        goal_padding_cells = max(
            PADDING.GOAL_PADDING,
            int(math.ceil((robot_radius_m + 0.05) / max(horizontal_scale, 1.0e-6))),
        )

    if args_cli.spawn_padding_cells is not None:
        spawn_padding_cells = int(args_cli.spawn_padding_cells)
    else:
        spawn_padding_cells = max(
            goal_padding_cells,
            PADDING.SPAWN_PADDING,
            int(math.ceil((robot_radius_m + 0.15) / max(horizontal_scale, 1.0e-6))),
        )

    return goal_padding_cells, spawn_padding_cells, robot_radius_m, footprint_dims


def configure_usd_navigation_scene(env):
    """Build merged USD geometry, derive navigation masks, and retarget ray-cast sensors."""
    terrain = env.unwrapped.scene.terrain
    if not terrain.terrain_prim_paths:
        raise RuntimeError("USD terrain importer did not expose a terrain prim path.")

    terrain_root_path = terrain.terrain_prim_paths[0]
    goal_padding_cells, spawn_padding_cells, robot_radius_m, footprint_dims = _derive_usd_padding_cells(
        env, HORIZONTAL_SCALE
    )
    usd_terrain = build_navigation_terrain_from_usd_stage(
        scene_root_path=terrain_root_path,
        merged_mesh_path=RAYCAST_MERGED_MESH_PATH,
        device=env.unwrapped.device,
        goal_padding_cells=goal_padding_cells,
        spawn_padding_cells=spawn_padding_cells,
        coarse_cell_size=args_cli.cell_size if args_cli.cell_size is not None else 2.0,
    )

    device = env.unwrapped.device
    terrain._height_field_visual = usd_terrain.height_field_visual.to(device=device)
    terrain._height_field_valid_mask = usd_terrain.height_field_valid_mask.to(device=device)
    terrain._height_field_platform_mask = usd_terrain.height_field_platform_mask.to(device=device)
    terrain._height_field_spawn_mask = usd_terrain.height_field_spawn_mask.to(device=device)

    terrain_origin = torch.tensor(
        [[[usd_terrain.terrain_origin_xy[0], usd_terrain.terrain_origin_xy[1], 0.0]]],
        device=device,
        dtype=torch.float32,
    )
    terrain.terrain_origins = terrain_origin
    terrain.terrain_levels = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.long)
    terrain.terrain_types = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.long)
    terrain.max_terrain_level = 1
    terrain.env_origins = terrain_origin[0, 0].repeat(env.unwrapped.num_envs, 1)

    coarse_grid_dim = max(1, int(math.ceil(usd_terrain.terrain_size / max(usd_terrain.coarse_cell_size, 1.0e-6))))
    terrain.cfg.terrain_generator = SimpleNamespace(
        size=(usd_terrain.terrain_size, usd_terrain.terrain_size),
        horizontal_scale=usd_terrain.horizontal_scale,
        num_rows=1,
        num_cols=1,
        curriculum=False,
        sub_terrains={
            "usd_scene": SimpleNamespace(
                cell_size=usd_terrain.coarse_cell_size,
                grid_size=(coarse_grid_dim, coarse_grid_dim),
            )
        },
    )

    robot_goal_term = env.unwrapped.command_manager._terms.get("robot_goal")
    if robot_goal_term is not None:
        robot_goal_term.terrain_size = usd_terrain.terrain_size
        robot_goal_term.num_terrain_rows = 1
        robot_goal_term.num_terrain_cols = 1
        robot_goal_term._sampling_initialized = False
        robot_goal_term._position_sampler = None
        usd_name = Path(args_cli.usd).expanduser().name if args_cli.usd is not None else ""
        if usd_name == "Astera_upd.usd":
            robot_goal_term.fixed_goal_positions_world = torch.tensor(
                ASTERA_UPD_GOAL_LOCATIONS,
                device=device,
                dtype=torch.float32,
            )
            print(
                f"[INFO] Using {len(ASTERA_UPD_GOAL_LOCATIONS)} fixed Astera goal locations "
                f"for {usd_name}"
            )
        elif hasattr(robot_goal_term, "fixed_goal_positions_world"):
            robot_goal_term.fixed_goal_positions_world = None

    raycast_camera = _rebind_raycaster_sensor_to_mesh(env, "raycast_camera", usd_terrain.merged_mesh_path)
    _rebind_raycaster_sensor_to_mesh(env, "height_scanner_critic", usd_terrain.merged_mesh_path)

    num_goal_cells = int(terrain._height_field_valid_mask.sum().item())
    num_spawn_cells = int(terrain._height_field_spawn_mask.sum().item())
    print(
        f"[INFO] USD navigation terrain ready from {terrain_root_path}: "
        f"merged {usd_terrain.num_source_prims} source prims into {usd_terrain.merged_mesh_path}; "
        f"terrain_size={usd_terrain.terrain_size:.2f}m; floor_z={usd_terrain.floor_height:.3f}; "
        f"goal_cells={num_goal_cells}; spawn_cells={num_spawn_cells}"
    )
    print(
        f"[INFO] USD footprint padding: robot_xy=({footprint_dims[0]:.3f}m, {footprint_dims[1]:.3f}m); "
        f"radius={robot_radius_m:.3f}m; goal_pad={goal_padding_cells} cells; spawn_pad={spawn_padding_cells} cells"
    )
    print(
        f"[INFO] Retargeted ray-cast sensors to {usd_terrain.merged_mesh_path} "
        f"(excluded NvbloxMesh from USD merge)."
    )
    return raycast_camera, usd_terrain


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


def _get_base_contact_monitor(env) -> tuple[float | None, bool]:
    term_cfg = getattr(env.unwrapped.cfg.terminations, "base_contact", None)
    if term_cfg is None or not hasattr(term_cfg, "params"):
        return None, False

    threshold = term_cfg.params.get("threshold")
    horizontal_only = bool(term_cfg.params.get("horizontal_only", False))
    if threshold is None:
        return None, horizontal_only
    return float(threshold), horizontal_only


def _resolve_robot_body_name(robot, body_id: int) -> str:
    if body_id < 0 or body_id >= len(robot.body_names):
        return f"body_id={body_id}"
    return robot.body_names[body_id]


def _resample_goal_after_success(env, robot_goal_term: StandaloneRobotGoalCommand | None) -> tuple[bool, torch.Tensor]:
    if robot_goal_term is None or not hasattr(robot_goal_term, "resample_goal_only"):
        return False, torch.empty(0, dtype=torch.long, device=env.unwrapped.device)

    success_cfg = getattr(env.unwrapped.cfg.terminations, "in_goal", None)
    success_params = dict(getattr(success_cfg, "params", {})) if success_cfg is not None else {}
    success_mask = nav_mdp.in_goal(env.unwrapped, **success_params)
    success_env_ids = success_mask.nonzero(as_tuple=False).squeeze(-1)
    if success_env_ids.numel() == 0:
        return False, success_env_ids

    robot_goal_term.resample_goal_only(success_env_ids)
    return True, success_env_ids


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
    usd_terrain_info = None
    if args_cli.usd:
        raycast_camera, usd_terrain_info = configure_usd_navigation_scene(env)
    else:
        raycast_camera = configure_raycast_camera_with_merged_scene(env)
    env = SruRslRlVecEnvWrapper(env)
    robot = env.unwrapped.scene["robot"]

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
    obs, _ = env.reset()

    raycast_camera = env.unwrapped.scene.sensors.get("raycast_camera", raycast_camera)
    robot_goal_term = env.unwrapped.command_manager._terms.get("robot_goal")
    terrain_generator = env.unwrapped.scene.terrain.cfg.terrain_generator
    collision_threshold, collision_horizontal_only = _get_base_contact_monitor(env)
    setup_depth_window()

    physics_hz = 1.0 / env_cfg.sim.dt
    policy_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    camera_period = None if raycast_camera is None else raycast_camera.cfg.update_period
    camera_hz = None if camera_period in (None, 0.0) else 1.0 / camera_period
    print("[INFO] Using the real task env/action/observation pipeline from play.py")
    if args_cli.usd:
        print("[INFO] Terrain/spawn/termination: USD scene with generated navigation masks")
    else:
        print("[INFO] Terrain/spawn/termination: local standalone overrides")
    if terrain_generator is not None:
        print(
            f"[INFO] Terrain grid: {terrain_generator.num_rows}x{terrain_generator.num_cols}; "
            f"sub-terrains: {', '.join(terrain_generator.sub_terrains.keys())}"
        )
    if usd_terrain_info is not None:
        print(
            f"[INFO] USD terrain origin: ({usd_terrain_info.terrain_origin_xy[0]:.3f}, "
            f"{usd_terrain_info.terrain_origin_xy[1]:.3f}); floor_z={usd_terrain_info.floor_height:.3f}"
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
    if collision_threshold is not None:
        axis_label = "force_xy" if collision_horizontal_only else "force"
        print(f"[INFO] Collision monitor: env0 {axis_label} > {collision_threshold:.2f}")

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        goal_resampled, goal_resampled_env_ids = _resample_goal_after_success(env, robot_goal_term)
        if goal_resampled:
            obs, _ = env.get_observations()
            if (goal_resampled_env_ids == 0).any():
                goal_pos = robot_goal_term.goal_position_world[0]
                print(
                    f"[STEP {step_count:06d}] goal_reached=True new_goal_w="
                    f"({goal_pos[0].item():.3f}, {goal_pos[1].item():.3f}, {goal_pos[2].item():.3f})",
                    flush=True,
                )
        robot_pos = robot.data.root_pos_w[0]
        print(
            f"[STEP {step_count:06d}] robot_pos_w="
            f"({robot_pos[0].item():.3f}, {robot_pos[1].item():.3f}, {robot_pos[2].item():.3f})",
            flush=True,
        )
        if robot_goal_term is not None and collision_threshold is not None:
            contact_force = (
                robot_goal_term.last_base_contact_force_xy[0]
                if collision_horizontal_only
                else robot_goal_term.last_base_contact_force[0]
            )
            if contact_force.item() > collision_threshold:
                body_id = int(robot_goal_term.last_contact_body_id[0].item())
                body_name = _resolve_robot_body_name(robot, body_id)
                contact_force_w = robot_goal_term.last_base_contact_force_w[0]
                print(
                    f"[STEP {step_count:06d}] collision_detected=True "
                    f"body={body_name} "
                    f"force_xy={robot_goal_term.last_base_contact_force_xy[0].item():.3f} "
                    f"force_w=({contact_force_w[0].item():.3f}, {contact_force_w[1].item():.3f}, "
                    f"{contact_force_w[2].item():.3f}) "
                    f"threshold={collision_threshold:.3f}",
                    flush=True,
                )

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
