#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Continue training from an existing checkpoint for a second curriculum stage.

This script is intended for stage-wise training where you want to:
- load a previous checkpoint,
- optionally change terrain difficulty/composition,
- optionally change reward weights,
- continue training with the same or a different algorithm config.

By default it loads model weights only and resets the optimizer state, which is
usually the safer option when reward scales or terrain distributions change.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Continue navigation training from a checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to load for second-stage training.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="Additional training iterations to run after loading the checkpoint.",
)
parser.add_argument("--run_name", type=str, default=None, help="Name appended to the log directory.")
parser.add_argument(
    "--staggered_reset_buckets",
    type=int,
    default=0,
    help=(
        "Number of rollout-phase buckets for startup staggered resets. "
        "Set to 0 or 1 to disable and keep the old random episode-length initialization."
    ),
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=None,
    help=(
        "Override the algorithm learning rate. When set, the resumed optimizer LR is replaced after checkpoint load "
        "and the second-stage learning-rate schedule restarts from iteration 0."
    ),
)
parser.add_argument(
    "--load_optimizer",
    action="store_true",
    default=False,
    help="Load optimizer state from the checkpoint. Default is model-only warm start.",
)
parser.add_argument(
    "--reset_learning_iteration",
    action="store_true",
    default=False,
    help="Reset the runner iteration counter to zero after loading the checkpoint.",
)

# Terrain arguments
parser.add_argument("--terrain_rows", type=int, default=None, help="Terrain grid rows (difficulty levels).")
parser.add_argument("--terrain_cols", type=int, default=None, help="Terrain grid columns (variations).")
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    choices=["maze", "non_maze", "both", "flat"],
    help="Terrain type override: maze, non_maze, both (keep mixed set), or flat.",
)
parser.add_argument(
    "--difficulty", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
    help="Terrain difficulty range, e.g. --difficulty 0.5 1.0",
)
parser.add_argument("--cell_size", type=float, default=None, help="Maze cell size in meters.")
parser.add_argument("--wall_ratio", type=float, default=None, help="Random wall ratio (0=no walls, 1=max walls).")
parser.add_argument("--terrain_size", type=float, default=None, help="Terrain tile size in meters.")
parser.add_argument("--grid_size", type=int, default=None, help="Maze grid dimension, e.g. 10 for 10x10.")

# Reward / termination overrides for second stage
parser.add_argument(
    "--disable_goal_progress",
    action="store_true",
    default=False,
    help="Disable the goal_progress reward term for second-stage training.",
)
parser.add_argument("--goal_progress_weight", type=float, default=None, help="Override goal_progress weight.")
parser.add_argument(
    "--pose_goal_proximity_weight", type=float, default=None, help="Override pose_goal_proximity weight."
)
parser.add_argument(
    "--pose_goal_hold_bonus_weight", type=float, default=None, help="Override pose_goal_hold_bonus weight."
)
parser.add_argument(
    "--reach_goal_xy_soft_weight", type=float, default=None, help="Override reach_goal_xy_soft weight."
)
parser.add_argument(
    "--reach_goal_xy_tight_weight", type=float, default=None, help="Override reach_goal_xy_tight weight."
)
parser.add_argument(
    "--episode_termination_weight",
    type=float,
    default=None,
    help="Override the episode_termination reward weight (collision/failure penalty).",
)
parser.add_argument(
    "--terrain_fall_penalty_weight",
    type=float,
    default=None,
    help="Override the terrain_fall_penalty reward weight.",
)
parser.add_argument(
    "--base_contact_penalty_weight",
    type=float,
    default=None,
    help="Override the base_contact_penalty reward weight.",
)
parser.add_argument(
    "--base_contact_threshold",
    type=float,
    default=None,
    help="Override the base contact termination threshold.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_nav_task.vecenv_wrapper import SruRslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _apply_terrain_overrides(env_cfg) -> None:
    tg = env_cfg.scene.terrain.terrain_generator
    if tg is None:
        return

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

        if args_cli.terrain_type == "both":
            for st in tg.sub_terrains.values():
                if args_cli.cell_size is not None:
                    st.cell_size = args_cli.cell_size
                if args_cli.wall_ratio is not None:
                    st.random_wall_ratio = args_cli.wall_ratio
                if gs is not None:
                    st.grid_size = gs
            return

        base = next(iter(tg.sub_terrains.values()))
        cs = args_cli.cell_size or getattr(base, "cell_size", 2.0)
        wr = args_cli.wall_ratio if args_cli.wall_ratio is not None else getattr(base, "random_wall_ratio", 0.5)
        resolved_gs = gs or getattr(base, "grid_size", (15, 15))

        if args_cli.terrain_type == "maze":
            tg.sub_terrains = {
                "maze": HfMazeTerrainCfg(
                    proportion=1.0,
                    open_probability=0.9,
                    grid_size=resolved_gs,
                    cell_size=cs,
                    add_noise_to_flat=False,
                    add_goal=True,
                    randomize_wall=True,
                    random_wall_ratio=wr,
                    add_stairs_to_maze=False,
                ),
            }
        elif args_cli.terrain_type == "non_maze":
            tg.sub_terrains = {
                "non_maze": HfMazeTerrainCfg(
                    proportion=1.0,
                    open_probability=0.9,
                    grid_size=resolved_gs,
                    cell_size=cs,
                    add_noise_to_flat=False,
                    add_goal=True,
                    randomize_wall=True,
                    random_wall_ratio=wr,
                    non_maze_terrain=True,
                    dynamic_obstacles=False,
                ),
            }
        elif args_cli.terrain_type == "flat":
            tg.sub_terrains = {
                "flat": HfMazeTerrainCfg(
                    proportion=1.0,
                    open_probability=1.0,
                    grid_size=resolved_gs,
                    cell_size=cs,
                    add_noise_to_flat=False,
                    add_goal=True,
                    randomize_wall=False,
                    random_wall_ratio=0.0,
                    non_maze_terrain=True,
                    dynamic_obstacles=False,
                ),
            }
    else:
        for st in tg.sub_terrains.values():
            if args_cli.cell_size is not None:
                st.cell_size = args_cli.cell_size
            if args_cli.wall_ratio is not None:
                st.random_wall_ratio = args_cli.wall_ratio
            if gs is not None:
                st.grid_size = gs


def _apply_second_stage_overrides(env_cfg) -> None:
    if args_cli.disable_goal_progress and hasattr(env_cfg.rewards, "goal_progress"):
        env_cfg.rewards.goal_progress.weight = 0.0
    if args_cli.goal_progress_weight is not None and hasattr(env_cfg.rewards, "goal_progress"):
        env_cfg.rewards.goal_progress.weight = args_cli.goal_progress_weight
    if args_cli.pose_goal_proximity_weight is not None and hasattr(env_cfg.rewards, "pose_goal_proximity"):
        env_cfg.rewards.pose_goal_proximity.weight = args_cli.pose_goal_proximity_weight
    if args_cli.pose_goal_hold_bonus_weight is not None and hasattr(env_cfg.rewards, "pose_goal_hold_bonus"):
        env_cfg.rewards.pose_goal_hold_bonus.weight = args_cli.pose_goal_hold_bonus_weight
    if args_cli.reach_goal_xy_soft_weight is not None and hasattr(env_cfg.rewards, "reach_goal_xy_soft"):
        env_cfg.rewards.reach_goal_xy_soft.weight = args_cli.reach_goal_xy_soft_weight
    if args_cli.reach_goal_xy_tight_weight is not None and hasattr(env_cfg.rewards, "reach_goal_xy_tight"):
        env_cfg.rewards.reach_goal_xy_tight.weight = args_cli.reach_goal_xy_tight_weight
    if args_cli.episode_termination_weight is not None and hasattr(env_cfg.rewards, "episode_termination"):
        env_cfg.rewards.episode_termination.weight = args_cli.episode_termination_weight
    if args_cli.terrain_fall_penalty_weight is not None and hasattr(env_cfg.rewards, "terrain_fall_penalty"):
        env_cfg.rewards.terrain_fall_penalty.weight = args_cli.terrain_fall_penalty_weight
    if args_cli.base_contact_penalty_weight is not None and hasattr(env_cfg.rewards, "base_contact_penalty"):
        env_cfg.rewards.base_contact_penalty.weight = args_cli.base_contact_penalty_weight
    if args_cli.base_contact_threshold is not None and hasattr(env_cfg.terminations, "base_contact"):
        env_cfg.terminations.base_contact.params["threshold"] = args_cli.base_contact_threshold


def _apply_agent_overrides(agent_cfg) -> None:
    if args_cli.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate


def _override_loaded_learning_rate(runner, learning_rate: float) -> None:
    alg = runner.alg

    if hasattr(alg, "learning_rate"):
        alg.learning_rate = learning_rate
    if hasattr(alg, "max_learning_rate"):
        alg.max_learning_rate = learning_rate
    if hasattr(alg, "original_learning_rate"):
        alg.original_learning_rate = learning_rate

    for optimizer_name in ("optimizer", "optimizer_1", "optimizer_2"):
        optimizer = getattr(alg, optimizer_name, None)
        if optimizer is None:
            continue
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name

    _apply_terrain_overrides(env_cfg)
    _apply_second_stage_overrides(env_cfg)
    _apply_agent_overrides(agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = SruRslRlVecEnvWrapper(env)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    print(f"[INFO] Loading optimizer state: {args_cli.load_optimizer}")
    runner.load(checkpoint_path, load_optimizer=args_cli.load_optimizer)
    if args_cli.learning_rate is not None:
        _override_loaded_learning_rate(runner, args_cli.learning_rate)
        print(f"[INFO] Overriding learning rate after checkpoint load: {args_cli.learning_rate:.6g}")
        runner.current_learning_iteration = 0
        print("[INFO] Reset learning iteration counter to 0 to restart the second-stage learning-rate schedule.")
    elif args_cli.reset_learning_iteration:
        runner.current_learning_iteration = 0
        print("[INFO] Reset learning iteration counter to 0 for second-stage logging.")

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
        staggered_reset_buckets=args_cli.staggered_reset_buckets,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
