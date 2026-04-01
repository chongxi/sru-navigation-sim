#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Play a trained navigation policy with the real task env and a standalone txt2maze terrain override.

This script keeps the same env -> wrapper -> runner -> policy path as `scripts/play.py`, but replaces
the local terrain generator with a text-layout maze builder where `#` is a wall cell and `.` is a
navigable floor cell. Observations, actions, resets, and the inference policy still go through the real
task code path.

Usage:
    python scripts/play_with_txt2maze_night.py \
      --checkpoint logs/rsl_rl/diff_drive_navigation_mdpo/2026-03-30_22-01-17/model_900.pt \
      --maze path/to/maze.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
import traceback
import textwrap
from collections.abc import Sequence
from pathlib import Path
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
parser.add_argument(
    "--terrain_rows",
    type=int,
    default=None,
    help="Number of terrain rows to instantiate. Defaults to 1.",
)
parser.add_argument(
    "--terrain_cols",
    type=int,
    default=None,
    help="Number of terrain columns to instantiate. Defaults to 1.",
)
parser.add_argument(
    "--maze",
    "--maze_layout_file",
    dest="maze",
    type=str,
    default=None,
    help="Path to a txt or YAML occupancy-grid layout file.",
)
parser.add_argument(
    "--maze_layout",
    type=str,
    default=None,
    help="Inline occupancy-grid layout. Use '/' between rows, e.g. '####/#..#/####'.",
)
parser.add_argument("--maze_wall_token", type=str, default="#", help="Single-character wall token in the txt maze.")
parser.add_argument("--maze_open_token", type=str, default=".", help="Single-character floor token in the txt maze.")
parser.add_argument("--cell_size", type=float, default=None, help="Maze cell size in meters. Defaults to 2.0.")
parser.add_argument(
    "--terrain_size",
    type=float,
    default=None,
    help="Optional square terrain tile size in meters. Defaults to fitting the txt maze exactly.",
)
parser.add_argument("--terrain_border_width", type=float, default=20.0, help="Outer border width around the terrain grid.")
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

parser.add_argument(
    "--debug_lifecycle",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Print lifecycle and first-step debug logs to help diagnose unexpected exits.",
)
parser.add_argument(
    "--debug_print_every",
    type=int,
    default=100,
    help="Print per-step summaries every N policy steps when debug_lifecycle is enabled.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Optional limit on policy steps. Use 0 to run until the app closes.",
)

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
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401
import isaaclab_nav_task.navigation.mdp as nav_mdp

from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import PositionSampler, RobotNavigationGoalCommand
from isaaclab_nav_task.navigation.mdp.navigation.goal_commands_cfg import RobotNavigationGoalCommandCfg
from isaaclab_nav_task.terrains.hf_terrains_maze import TerrainData
from isaaclab_nav_task.terrains.terrain_constants import PADDING
from isaaclab_nav_task.vecenv_wrapper import SruRslRlVecEnvWrapper


try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yaml
except ImportError:
    yaml = None


def debug_log(message: str):
    """Emit a flushed lifecycle log line for debugging unexpected exits."""
    if not getattr(args_cli, "debug_lifecycle", False):
        return
    timestamp = time.strftime("%H:%M:%S")
    print(f"[DEBUG][{timestamp}] {message}", flush=True)


def _tensor_summary(name: str, value: Any) -> str:
    """Compact tensor summary for lifecycle debug output."""
    if not torch.is_tensor(value):
        return f"{name}=<non-tensor {type(value).__name__}>"
    if value.numel() == 0:
        return f"{name}: shape={tuple(value.shape)} empty"
    detached = value.detach()
    if detached.is_cuda:
        detached = detached.cpu()
    detached = detached.float()
    return (
        f"{name}: shape={tuple(value.shape)} "
        f"min={detached.min().item():.4f} "
        f"max={detached.max().item():.4f} "
        f"mean={detached.mean().item():.4f}"
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


DEFAULT_TXT_MAZE_LAYOUT = """
########
#......#
#......#
#...####
#......#
#.#....#
#.#....#
########
"""

DEFAULT_TXT_MAZE_PATH = Path("maze/maze.yaml")


def _extract_first_token(items: Any) -> str | None:
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        token = item.get("token")
        if isinstance(token, str) and len(token) == 1:
            return token
    return None


def _extract_yaml_grid_text(raw_text: str) -> str | None:
    match = re.search(r"grid:\s*\|\s*\n(?P<body>(?:[ \t]+.*\n?)*)", raw_text)
    if match is None:
        return None
    return textwrap.dedent(match.group("body")).strip("\n")


def _normalize_txt_maze_rows(grid_text: str, wall_token: str, open_token: str) -> tuple[str, ...]:
    rows: list[str] = []
    allowed_tokens = {wall_token, open_token}

    for raw_line in textwrap.dedent(grid_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            line = line.split("|", 1)[1].strip()
        line = re.sub(r"\s+", "", line)
        if not line:
            continue
        if set(line).issubset(allowed_tokens):
            rows.append(line)

    if not rows:
        raise ValueError(
            "txt2maze layout did not contain any valid rows. Use only the configured wall/open tokens."
        )

    row_lengths = {len(row) for row in rows}
    if len(row_lengths) != 1:
        raise ValueError(f"txt2maze layout rows must all have the same width. Got widths: {sorted(row_lengths)}")

    if not any(open_token in row for row in rows):
        raise ValueError("txt2maze layout must contain at least one open cell.")

    return tuple(rows)


def txt2maze(grid_text: str, wall_token: str = "#", open_token: str = ".") -> np.ndarray:
    """Convert a txt occupancy layout into a 0/1 maze array where 1=wall and 0=open floor."""
    rows = _normalize_txt_maze_rows(grid_text, wall_token=wall_token, open_token=open_token)
    maze = np.zeros((len(rows), len(rows[0])), dtype=np.uint8)
    for row_idx, row in enumerate(rows):
        for col_idx, token in enumerate(row):
            maze[row_idx, col_idx] = 1 if token == wall_token else 0
    return maze


def _resolve_txt_maze_spec() -> tuple[tuple[str, ...], str, str, str]:
    wall_token = args_cli.maze_wall_token
    open_token = args_cli.maze_open_token

    if len(wall_token) != 1 or len(open_token) != 1:
        raise ValueError("maze_wall_token and maze_open_token must each be a single character.")
    if wall_token == open_token:
        raise ValueError("maze_wall_token and maze_open_token must be different.")

    default_maze_path = DEFAULT_TXT_MAZE_PATH.resolve()

    if args_cli.maze is not None:
        maze_path = Path(args_cli.maze).expanduser().resolve()
        raw_text = maze_path.read_text(encoding="utf-8")
        source_label = str(maze_path)
    elif args_cli.maze_layout is not None:
        raw_text = args_cli.maze_layout.replace("/", "\n")
        source_label = "inline --maze_layout"
    elif default_maze_path.exists():
        raw_text = default_maze_path.read_text(encoding="utf-8")
        source_label = str(default_maze_path)
    else:
        raw_text = DEFAULT_TXT_MAZE_LAYOUT
        source_label = "built-in demo layout"

    grid_text: str | None = None
    if yaml is not None:
        try:
            parsed = yaml.safe_load(raw_text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            layout_cfg = parsed.get("layout")
            if isinstance(layout_cfg, dict):
                grid_value = layout_cfg.get("grid")
                if isinstance(grid_value, str):
                    grid_text = grid_value
            if wall_token == parser.get_default("maze_wall_token"):
                token = _extract_first_token((parsed.get("config") or {}).get("wall_elements"))
                if token is not None:
                    wall_token = token
            if open_token == parser.get_default("maze_open_token"):
                token = _extract_first_token((parsed.get("config") or {}).get("cell_elements"))
                if token is not None:
                    open_token = token

    if grid_text is None:
        grid_text = _extract_yaml_grid_text(raw_text) or raw_text

    rows = _normalize_txt_maze_rows(grid_text, wall_token=wall_token, open_token=open_token)
    return rows, wall_token, open_token, source_label


@height_field_to_mesh
def txt_maze_terrain(_difficulty: float, cfg: "TxtMazeTerrainCfg") -> np.ndarray:
    """Generate a height-field terrain directly from a txt occupancy layout."""
    cell_pixels = int(cfg.cell_size / cfg.horizontal_scale)
    wall_height = int(cfg.wall_height / cfg.vertical_scale)
    terrain_w = int(cfg.size[0] / cfg.horizontal_scale)
    terrain_h = int(cfg.size[1] / cfg.horizontal_scale)

    maze = txt2maze("\n".join(cfg.maze_rows), wall_token=cfg.wall_token, open_token=cfg.open_token)
    grid_rows, grid_cols = maze.shape
    maze_w = grid_rows * cell_pixels
    maze_h = grid_cols * cell_pixels
    if maze_w > terrain_w or maze_h > terrain_h:
        raise ValueError(
            "txt2maze layout is larger than the configured terrain tile. "
            f"Required pixels=({maze_w}, {maze_h}), available=({terrain_w}, {terrain_h})."
        )

    terrain = TerrainData.create(terrain_w, terrain_h)
    terrain.heights[:, :] = 0
    terrain.valid_mask[:, :] = False
    terrain.platform_mask[:, :] = False

    x_offset = (terrain_w - maze_w) // 2
    y_offset = (terrain_h - maze_h) // 2

    for row_idx in range(grid_rows):
        for col_idx in range(grid_cols):
            xs = x_offset + row_idx * cell_pixels
            xe = xs + cell_pixels
            ys = y_offset + col_idx * cell_pixels
            ye = ys + cell_pixels
            if maze[row_idx, col_idx] == 0:
                terrain.set_ground(xs, xe, ys, ye)
            else:
                terrain.set_obstacle(xs, xe, ys, ye, wall_height)

    goal_padding_cells = int(cfg.goal_padding_cells) if cfg.goal_padding_cells is not None else PADDING.GOAL_PADDING
    spawn_padding_cells = (
        int(cfg.spawn_padding_cells) if cfg.spawn_padding_cells is not None else PADDING.SPAWN_PADDING
    )

    terrain.apply_padding(goal_padding_cells)
    terrain.exclude_borders(PADDING.BORDER_CELLS)

    spawn_mask = terrain.create_spawn_mask(spawn_padding_cells)
    spawn_mask[:PADDING.BORDER_CELLS, :] = False
    spawn_mask[-PADDING.BORDER_CELLS:, :] = False
    spawn_mask[:, :PADDING.BORDER_CELLS] = False
    spawn_mask[:, -PADDING.BORDER_CELLS:] = False

    if cfg.add_goal:
        cfg.height_field_visual = torch.from_numpy(terrain.heights.copy()).unsqueeze(0)
        cfg.height_field_valid_mask = torch.from_numpy(terrain.valid_mask.copy()).unsqueeze(0)
        cfg.height_field_platform_mask = torch.from_numpy(terrain.platform_mask.copy()).unsqueeze(0)
        cfg.height_field_spawn_mask = torch.from_numpy(spawn_mask.copy()).unsqueeze(0)

    return terrain.heights


@configclass
class TxtMazeTerrainCfg(HfTerrainBaseCfg):
    """Standalone height-field terrain config backed by a txt occupancy-grid layout."""

    function = txt_maze_terrain

    height_field_visual: torch.Tensor = None
    height_field_valid_mask: torch.Tensor = None
    height_field_platform_mask: torch.Tensor = None
    height_field_spawn_mask: torch.Tensor = None

    maze_rows: tuple[str, ...] = ()
    wall_token: str = "#"
    open_token: str = "."
    cell_size: float = 2.0
    wall_height: float = 1.5
    goal_padding_cells: int | None = None
    spawn_padding_cells: int | None = None
    add_goal: Any = True


def _build_txt_maze_subterrain() -> tuple[str, TxtMazeTerrainCfg, int, int]:
    rows, wall_token, open_token, source_label = _resolve_txt_maze_spec()
    grid_rows = len(rows)
    grid_cols = len(rows[0])
    cell_size = args_cli.cell_size if args_cli.cell_size is not None else 2.0
    cfg = TxtMazeTerrainCfg(
        proportion=1.0,
        maze_rows=rows,
        wall_token=wall_token,
        open_token=open_token,
        cell_size=cell_size,
        wall_height=args_cli.wall_height,
        goal_padding_cells=args_cli.goal_padding_cells,
        spawn_padding_cells=args_cli.spawn_padding_cells,
        add_goal=True,
    )
    return source_label, cfg, grid_rows, grid_cols


def build_standalone_terrain_cfg(env_cfg: ManagerBasedRLEnvCfg) -> TerrainImporterCfg:
    """Build the terrain locally in this script while keeping the rest of the task env unchanged."""
    base_terrain_cfg = env_cfg.scene.terrain
    base_tg = base_terrain_cfg.terrain_generator
    if base_tg is None:
        raise ValueError("Standalone terrain override requires a generator-based terrain config.")

    source_label, subterrain_cfg, grid_rows, grid_cols = _build_txt_maze_subterrain()
    cell_size = subterrain_cfg.cell_size
    maze_extent = max(grid_rows, grid_cols) * cell_size
    base_tile_size = float(base_tg.size[0])
    if args_cli.terrain_size is not None:
        terrain_size = float(args_cli.terrain_size)
    else:
        terrain_size = max(maze_extent, base_tile_size)
    if terrain_size < maze_extent:
        raise ValueError(
            f"terrain_size={terrain_size:.2f}m is smaller than the txt maze footprint {maze_extent:.2f}m. "
            "Increase --terrain_size or reduce the maze dimensions."
        )
    if terrain_size < base_tile_size:
        raise ValueError(
            f"terrain_size={terrain_size:.2f}m is smaller than the base task terrain tile size {base_tile_size:.2f}m. "
            "This task path expects at least the base tile size. Omit --terrain_size or set it >= "
            f"{base_tile_size:.2f}."
        )
    terrain_size_xy = (terrain_size, terrain_size)
    num_rows = args_cli.terrain_rows if args_cli.terrain_rows is not None else 1
    num_cols = args_cli.terrain_cols if args_cli.terrain_cols is not None else 1
    sub_terrains = {"txt2maze": subterrain_cfg}

    max_init_terrain_level = base_terrain_cfg.max_init_terrain_level
    if max_init_terrain_level is not None:
        max_init_terrain_level = min(int(max_init_terrain_level), max(0, num_rows - 1))

    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="multiply",
        restitution=args_cli.ground_restitution,
        static_friction=args_cli.ground_static_friction,
        dynamic_friction=args_cli.ground_dynamic_friction,
        compliant_contact_stiffness=5e5,
        compliant_contact_damping=300.0,
    )

    print(
        f"[INFO] Local txt2maze terrain: layout={grid_rows}x{grid_cols} cells from {source_label}; "
        f"cell_size={cell_size:.2f}m; maze_extent={maze_extent:.1f}m; "
        f"tile_grid={num_rows}x{num_cols}; tile_size={terrain_size_xy[0]:.1f}m"
    )

    return TerrainImporterCfg(
        prim_path=base_terrain_cfg.prim_path,
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=terrain_size_xy,
            border_width=args_cli.terrain_border_width,
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_scale=0.5,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            difficulty_range=(1.0, 1.0),
            sub_terrains=sub_terrains,
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
    debug_log(
        f"main() start | task={args_cli.task} checkpoint={args_cli.checkpoint} "
        f"maze={args_cli.maze} headless={getattr(args_cli, 'headless', False)} "
        f"show_depth={args_cli.show_depth}"
    )

    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")
    debug_log(
        f"Resolved gym spec | env_cfg={getattr(env_cfg_class, '__name__', env_cfg_class)} "
        f"agent_cfg={getattr(agent_cfg_class, '__name__', agent_cfg_class)}"
    )

    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    agent_cfg = agent_cfg_class()
    debug_log("Instantiated env_cfg and agent_cfg")

    configure_night_scene(env_cfg)
    apply_cli_overrides(env_cfg)
    debug_log("Applied scene and CLI overrides")

    debug_log("Calling gym.make(...)")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    debug_log(f"gym.make returned {type(env).__name__}")
    env = SruRslRlVecEnvWrapper(env)
    debug_log(
        f"Wrapped env | num_envs={env.num_envs} num_actions={env.num_actions} "
        f"max_episode_length={env.max_episode_length}"
    )

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = find_latest_checkpoint(log_root_path, checkpoint_pattern="model_.*.pt")
    debug_log(f"Using checkpoint: {resume_path}")

    debug_log("Constructing OnPolicyRunner")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    debug_log("Loading checkpoint into runner")
    load_checkpoint_with_fallback(runner, resume_path)

    if args_cli.export_jit:
        export_policy_jit(runner, resume_path)
    if args_cli.export_onnx:
        export_policy_onnx(runner, resume_path)

    debug_log("Creating inference policy")
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    debug_log("Fetching initial observations")
    obs, _ = env.get_observations()
    debug_log(_tensor_summary("initial_obs", obs))

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
    debug_log(f"Before loop | simulation_app.is_running()={simulation_app.is_running()}")
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
            if step_count < 3 or (
                args_cli.debug_print_every > 0 and step_count % args_cli.debug_print_every == 0
            ):
                debug_log(_tensor_summary(f"actions@step{step_count}", actions))

            obs, rew, dones, infos = env.step(actions)

            if step_count < 3 or (
                args_cli.debug_print_every > 0 and step_count % args_cli.debug_print_every == 0
            ):
                debug_log(
                    f"step={step_count} "
                    f"{_tensor_summary('obs', obs)} "
                    f"{_tensor_summary('rew', rew)} "
                    f"dones_sum={int(dones.sum().item())} "
                    f"time_outs_sum={int(infos.get('time_outs', torch.zeros_like(dones)).sum().item())}"
                )

            if raycast_camera is not None and step_count % max(args_cli.depth_vis_every, 1) == 0:
                update_depth_window(raycast_camera, args_cli.depth_env_idx)

            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                debug_log(f"Reached max_steps={args_cli.max_steps}; leaving loop")
                break
    finally:
        debug_log(f"Leaving loop | step_count={step_count} simulation_app.is_running()={simulation_app.is_running()}")
        env.close()
        debug_log("env.close() finished")
        if cv2 is not None and args_cli.show_depth:
            cv2.destroyAllWindows()
            debug_log("cv2.destroyAllWindows() finished")


if __name__ == "__main__":
    debug_log("Script entry")
    try:
        main()
    except BaseException as exc:
        debug_log(f"Unhandled exception: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        debug_log("Calling simulation_app.close()")
        simulation_app.close()
        debug_log("simulation_app.close() finished")
