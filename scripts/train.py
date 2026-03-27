#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Train a navigation policy using RSL-RL (PPO/MDPO algorithms).

Usage:
    python scripts/train.py --task <task_name> --num_envs <num> [options]

Arguments:
    --task               Task name (required)
    --num_envs           Number of parallel environments
    --seed               Random seed
    --max_iterations     Training iterations
    --run_name           Custom run name for logging
    --video              Enable video recording
    --video_length       Video length in steps (default: 200)
    --video_interval     Recording interval in steps (default: 2000)

Examples:
    python scripts/train.py --task Isaac-Navigation-B2W-v0 --num_envs 2048
    python scripts/train.py --task Isaac-Navigation-B2W-v0 --video --seed 42

Logs saved to: logs/rsl_rl/<experiment_name>/<timestamp>/
"""

from __future__ import annotations

import argparse
import sys

# Add the parent directory to the path so we can import from the extension
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train a navigation policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--run_name", type=str, default=None, help="Name of the wandb run (appended to log directory).")

# Terrain arguments
parser.add_argument("--terrain_rows", type=int, default=None, help="Terrain grid rows (difficulty levels).")
parser.add_argument("--terrain_cols", type=int, default=None, help="Terrain grid columns (variations).")
parser.add_argument("--terrain_type", type=str, default=None,
                    choices=["maze", "non_maze", "both", "flat"],
                    help="Terrain type: maze, non_maze, both (default from config), or flat (no walls).")
parser.add_argument("--difficulty", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="Terrain difficulty range, e.g. --difficulty 0.3 0.8")
parser.add_argument("--cell_size", type=float, default=None, help="Maze cell size in meters.")
parser.add_argument("--wall_ratio", type=float, default=None, help="Random wall ratio (0=no walls, 1=max walls).")
parser.add_argument("--terrain_size", type=float, default=None, help="Terrain tile size in meters (default 30).")
parser.add_argument("--grid_size", type=int, default=None, help="Maze grid dimension, e.g. 15 for 15x15 (default 15).")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching simulation
import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

# Import Isaac Lab extensions
import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_nav_task.vecenv_wrapper import SruRslRlVecEnvWrapper

# Set torch backends for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train navigation policy with RSL-RL."""
    # Load the configurations from the registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # Override config from command line
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name

    # Override terrain config from command line
    tg = env_cfg.scene.terrain.terrain_generator
    if tg is not None:
        if args_cli.terrain_rows is not None:
            tg.num_rows = args_cli.terrain_rows
        if args_cli.terrain_cols is not None:
            tg.num_cols = args_cli.terrain_cols
        if args_cli.difficulty is not None:
            tg.difficulty_range = list(args_cli.difficulty)
        if args_cli.terrain_size is not None:
            tg.size = (args_cli.terrain_size, args_cli.terrain_size)
        gs = (args_cli.grid_size, args_cli.grid_size) if args_cli.grid_size is not None else None
        if args_cli.terrain_type is not None:
            from isaaclab_nav_task.terrains.hf_terrains_maze_cfg import HfMazeTerrainCfg
            base = next(iter(tg.sub_terrains.values()))
            cs = args_cli.cell_size or getattr(base, "cell_size", 2.0)
            wr = args_cli.wall_ratio if args_cli.wall_ratio is not None else getattr(base, "random_wall_ratio", 0.5)
            _gs = gs or getattr(base, "grid_size", (15, 15))
            if args_cli.terrain_type == "maze":
                tg.sub_terrains = {
                    "maze": HfMazeTerrainCfg(
                        proportion=1.0, open_probability=0.9, grid_size=_gs,
                        cell_size=cs, add_noise_to_flat=False, add_goal=True,
                        randomize_wall=True, random_wall_ratio=wr,
                        add_stairs_to_maze=False,
                    ),
                }
            elif args_cli.terrain_type == "non_maze":
                tg.sub_terrains = {
                    "non_maze": HfMazeTerrainCfg(
                        proportion=1.0, open_probability=0.9, grid_size=_gs,
                        cell_size=cs, add_noise_to_flat=False, add_goal=True,
                        randomize_wall=True, random_wall_ratio=wr,
                        non_maze_terrain=True, dynamic_obstacles=False,
                    ),
                }
            elif args_cli.terrain_type == "flat":
                tg.sub_terrains = {
                    "flat": HfMazeTerrainCfg(
                        proportion=1.0, open_probability=1.0, grid_size=_gs,
                        cell_size=cs, add_noise_to_flat=False, add_goal=True,
                        randomize_wall=False, random_wall_ratio=0.0,
                        non_maze_terrain=True, dynamic_obstacles=False,
                    ),
                }
        else:
            # Apply per-sub-terrain overrides
            for st in tg.sub_terrains.values():
                if args_cli.cell_size is not None:
                    st.cell_size = args_cli.cell_size
                if args_cli.wall_ratio is not None:
                    st.random_wall_ratio = args_cli.wall_ratio
                if gs is not None:
                    st.grid_size = gs

    # Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Wrap the environment
    env = SruRslRlVecEnvWrapper(env)

    # Specify log directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # Specify run directory based on timestamp
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # Write git state to log
    runner.add_git_repo_to_log(__file__)
    # Save configuration
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation
    simulation_app.close()
