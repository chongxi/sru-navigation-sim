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
from dataclasses import make_dataclass

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
parser.add_argument("--episode_length_s", type=float, default=None, help="Override the episode length in seconds.")
parser.add_argument("--torch_compile_policy", action="store_true", default=False, help="Use torch.compile on supported policy hot paths.")
parser.add_argument("--torch_compile_mode", type=str, default=None, help="torch.compile mode to use, e.g. default or reduce-overhead.")
parser.add_argument("--run_name", type=str, default=None, help="Name of the wandb run (appended to log directory).")
parser.add_argument(
    "--staggered_reset_buckets",
    type=int,
    default=0,
    help=(
        "Number of rollout-phase buckets for startup staggered resets. "
        "Set to 0 or 1 to disable and keep the old random episode-length initialization."
    ),
)

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
    "--goal_progress_weight",
    type=float,
    default=None,
    help="Override the goal_progress reward weight.",
)
parser.add_argument(
    "--reach_goal_xy_soft_weight",
    type=float,
    default=None,
    help="Override the reach_goal_xy_soft reward weight.",
)
parser.add_argument(
    "--reach_goal_xy_tight_weight",
    type=float,
    default=None,
    help="Override the reach_goal_xy_tight reward weight.",
)
parser.add_argument(
    "--pose_goal_hold_bonus_weight",
    type=float,
    default=None,
    help="Override the pose_goal_hold_bonus reward weight.",
)
parser.add_argument(
    "--pose_goal_proximity_weight",
    type=float,
    default=None,
    help="Override the pose_goal_proximity reward weight.",
)
parser.add_argument(
    "--in_goal_bonus_weight",
    type=float,
    default=None,
    help="Override the in_goal terminal bonus reward weight.",
)
parser.add_argument(
    "--trapped_penalty_weight",
    type=float,
    default=None,
    help="Override the trapped termination penalty weight.",
)
parser.add_argument(
    "--large_pitch_angle_penalty_weight",
    type=float,
    default=None,
    help="Override the large_pitch_angle termination penalty weight.",
)
parser.add_argument(
    "--goal_progress_weight_end",
    type=float,
    default=None,
    help="Linearly decay goal_progress to this final weight over training.",
)
parser.add_argument(
    "--reach_goal_xy_soft_weight_end",
    type=float,
    default=None,
    help="Linearly change reach_goal_xy_soft to this final weight over training.",
)
parser.add_argument(
    "--reach_goal_xy_tight_weight_end",
    type=float,
    default=None,
    help="Linearly change reach_goal_xy_tight to this final weight over training.",
)
parser.add_argument(
    "--pose_goal_hold_bonus_weight_end",
    type=float,
    default=None,
    help="Linearly change pose_goal_hold_bonus to this final weight over training.",
)
parser.add_argument(
    "--pose_goal_proximity_weight_end",
    type=float,
    default=None,
    help="Linearly change pose_goal_proximity to this final weight over training.",
)
parser.add_argument(
    "--terrain_fall_penalty_weight_end",
    type=float,
    default=None,
    help="Linearly increase terrain_fall_penalty to this final weight over training.",
)
parser.add_argument(
    "--base_contact_penalty_weight_end",
    type=float,
    default=None,
    help="Linearly increase base_contact_penalty to this final weight over training.",
)
parser.add_argument(
    "--trapped_penalty_weight_end",
    type=float,
    default=None,
    help="Linearly increase trapped_penalty to this final weight over training.",
)
parser.add_argument(
    "--large_pitch_angle_penalty_weight_end",
    type=float,
    default=None,
    help="Linearly increase large_pitch_angle_penalty to this final weight over training.",
)
parser.add_argument(
    "--reward_schedule_start_frac",
    type=float,
    default=None,
    help="Fraction of total training env-steps where reward scheduling starts.",
)
parser.add_argument(
    "--reward_schedule_end_frac",
    type=float,
    default=None,
    help="Fraction of total training env-steps where reward scheduling ends.",
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _apply_task_default_overrides(args: argparse.Namespace):
    """Apply task-specific defaults when the user did not pass explicit overrides."""
    if args.task != "Isaac-Nav-MDPO-DiffDrive-v0":
        return

    default_overrides = {
        "num_envs": 4096,
        "max_iterations": 3000,
        "episode_length_s": 100.0,
        "difficulty": [0.5, 1.0],
        "goal_progress_weight": 15.0,
        "goal_progress_weight_end": 0.0,
        "reach_goal_xy_soft_weight_end": 0.0,
        "reach_goal_xy_tight_weight_end": 0.75,
        "in_goal_bonus_weight": 1800.0,
        "terrain_fall_penalty_weight": -250.0,
        "terrain_fall_penalty_weight_end": -1500.0,
        "base_contact_penalty_weight": 0.0,
        "base_contact_penalty_weight_end": -1200.0,
        "trapped_penalty_weight": 0.0,
        "trapped_penalty_weight_end": -3000.0,
        "large_pitch_angle_penalty_weight": -50.0,
        "large_pitch_angle_penalty_weight_end": -300.0,
        "reward_schedule_start_frac": 0.0,
        "reward_schedule_end_frac": 0.2,
    }

    for name, value in default_overrides.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


_apply_task_default_overrides(args_cli)

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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_nav_task.navigation.mdp as nav_mdp
from isaaclab_nav_task.vecenv_wrapper import SruRslRlVecEnvWrapper

# Set torch backends for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _attach_linear_reward_schedule(
    env_cfg: ManagerBasedRLEnvCfg,
    *,
    attr_name: str,
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
):
    """Attach a linear reward-weight curriculum term to the env config."""
    if env_cfg.curriculum is None:
        RuntimeCurriculumCfg = make_dataclass("RuntimeCurriculumCfg", [])
        env_cfg.curriculum = RuntimeCurriculumCfg()

    setattr(
        env_cfg.curriculum,
        attr_name,
        CurrTerm(
            func=nav_mdp.linearly_interpolate_reward_weight,
            params={
                "term_name": term_name,
                "start_weight": float(start_weight),
                "end_weight": float(end_weight),
                "start_step": int(start_step),
                "end_step": int(end_step),
            },
        ),
    )


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
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.torch_compile_policy:
        agent_cfg.torch_compile_policy = True
    if args_cli.torch_compile_mode is not None:
        agent_cfg.torch_compile_mode = args_cli.torch_compile_mode
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name

    # Override reward config from command line
    if args_cli.terrain_fall_penalty_weight is not None and hasattr(env_cfg.rewards, "terrain_fall_penalty"):
        env_cfg.rewards.terrain_fall_penalty.weight = args_cli.terrain_fall_penalty_weight
    if args_cli.base_contact_penalty_weight is not None and hasattr(env_cfg.rewards, "base_contact_penalty"):
        env_cfg.rewards.base_contact_penalty.weight = args_cli.base_contact_penalty_weight
    if args_cli.goal_progress_weight is not None and hasattr(env_cfg.rewards, "goal_progress"):
        env_cfg.rewards.goal_progress.weight = args_cli.goal_progress_weight
    if args_cli.reach_goal_xy_soft_weight is not None and hasattr(env_cfg.rewards, "reach_goal_xy_soft"):
        env_cfg.rewards.reach_goal_xy_soft.weight = args_cli.reach_goal_xy_soft_weight
    if args_cli.reach_goal_xy_tight_weight is not None and hasattr(env_cfg.rewards, "reach_goal_xy_tight"):
        env_cfg.rewards.reach_goal_xy_tight.weight = args_cli.reach_goal_xy_tight_weight
    if args_cli.pose_goal_hold_bonus_weight is not None and hasattr(env_cfg.rewards, "pose_goal_hold_bonus"):
        env_cfg.rewards.pose_goal_hold_bonus.weight = args_cli.pose_goal_hold_bonus_weight
    if args_cli.pose_goal_proximity_weight is not None and hasattr(env_cfg.rewards, "pose_goal_proximity"):
        env_cfg.rewards.pose_goal_proximity.weight = args_cli.pose_goal_proximity_weight
    if args_cli.in_goal_bonus_weight is not None and getattr(env_cfg.rewards, "in_goal_bonus", None) is not None:
        env_cfg.rewards.in_goal_bonus.weight = args_cli.in_goal_bonus_weight
    if args_cli.trapped_penalty_weight is not None and getattr(env_cfg.rewards, "trapped_penalty", None) is not None:
        env_cfg.rewards.trapped_penalty.weight = args_cli.trapped_penalty_weight
    if (
        args_cli.large_pitch_angle_penalty_weight is not None
        and getattr(env_cfg.rewards, "large_pitch_angle_penalty", None) is not None
    ):
        env_cfg.rewards.large_pitch_angle_penalty.weight = args_cli.large_pitch_angle_penalty_weight

    reward_schedule_start_frac = 0.0 if args_cli.reward_schedule_start_frac is None else args_cli.reward_schedule_start_frac
    reward_schedule_end_frac = 1.0 if args_cli.reward_schedule_end_frac is None else args_cli.reward_schedule_end_frac

    max_iterations_for_schedule = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    total_train_steps = int(agent_cfg.num_steps_per_env * max_iterations_for_schedule)
    schedule_start_step = int(total_train_steps * reward_schedule_start_frac)
    schedule_end_step = int(total_train_steps * reward_schedule_end_frac)

    if args_cli.goal_progress_weight_end is not None and hasattr(env_cfg.rewards, "goal_progress"):
        _attach_linear_reward_schedule(
            env_cfg,
            attr_name="goal_progress_weight_schedule",
            term_name="goal_progress",
            start_weight=env_cfg.rewards.goal_progress.weight,
            end_weight=args_cli.goal_progress_weight_end,
            start_step=schedule_start_step,
            end_step=schedule_end_step,
        )
    if args_cli.reach_goal_xy_soft_weight_end is not None and hasattr(env_cfg.rewards, "reach_goal_xy_soft"):
        _attach_linear_reward_schedule(
            env_cfg,
            attr_name="reach_goal_xy_soft_weight_schedule",
            term_name="reach_goal_xy_soft",
            start_weight=env_cfg.rewards.reach_goal_xy_soft.weight,
            end_weight=args_cli.reach_goal_xy_soft_weight_end,
            start_step=schedule_start_step,
            end_step=schedule_end_step,
        )
    if args_cli.reach_goal_xy_tight_weight_end is not None and hasattr(env_cfg.rewards, "reach_goal_xy_tight"):
        _attach_linear_reward_schedule(
            env_cfg,
            attr_name="reach_goal_xy_tight_weight_schedule",
            term_name="reach_goal_xy_tight",
            start_weight=env_cfg.rewards.reach_goal_xy_tight.weight,
            end_weight=args_cli.reach_goal_xy_tight_weight_end,
            start_step=schedule_start_step,
            end_step=schedule_end_step,
        )
    if args_cli.pose_goal_hold_bonus_weight_end is not None and hasattr(env_cfg.rewards, "pose_goal_hold_bonus"):
        _attach_linear_reward_schedule(
            env_cfg,
            attr_name="pose_goal_hold_bonus_weight_schedule",
            term_name="pose_goal_hold_bonus",
            start_weight=env_cfg.rewards.pose_goal_hold_bonus.weight,
            end_weight=args_cli.pose_goal_hold_bonus_weight_end,
            start_step=schedule_start_step,
            end_step=schedule_end_step,
        )
    if args_cli.pose_goal_proximity_weight_end is not None and hasattr(env_cfg.rewards, "pose_goal_proximity"):
        _attach_linear_reward_schedule(
            env_cfg,
            attr_name="pose_goal_proximity_weight_schedule",
            term_name="pose_goal_proximity",
            start_weight=env_cfg.rewards.pose_goal_proximity.weight,
            end_weight=args_cli.pose_goal_proximity_weight_end,
            start_step=schedule_start_step,
            end_step=schedule_end_step,
        )
    if args_cli.terrain_fall_penalty_weight_end is not None and hasattr(env_cfg.rewards, "terrain_fall_penalty"):
        _attach_linear_reward_schedule(
            env_cfg,
            attr_name="terrain_fall_penalty_weight_schedule",
            term_name="terrain_fall_penalty",
            start_weight=env_cfg.rewards.terrain_fall_penalty.weight,
            end_weight=args_cli.terrain_fall_penalty_weight_end,
            start_step=schedule_start_step,
            end_step=schedule_end_step,
        )
    if args_cli.base_contact_penalty_weight_end is not None and hasattr(env_cfg.rewards, "base_contact_penalty"):
        _attach_linear_reward_schedule(
            env_cfg,
            attr_name="base_contact_penalty_weight_schedule",
            term_name="base_contact_penalty",
            start_weight=env_cfg.rewards.base_contact_penalty.weight,
            end_weight=args_cli.base_contact_penalty_weight_end,
            start_step=schedule_start_step,
            end_step=schedule_end_step,
        )
    if args_cli.trapped_penalty_weight_end is not None and hasattr(env_cfg.rewards, "trapped_penalty"):
        _attach_linear_reward_schedule(
            env_cfg,
            attr_name="trapped_penalty_weight_schedule",
            term_name="trapped_penalty",
            start_weight=env_cfg.rewards.trapped_penalty.weight,
            end_weight=args_cli.trapped_penalty_weight_end,
            start_step=schedule_start_step,
            end_step=schedule_end_step,
        )
    if (
        args_cli.large_pitch_angle_penalty_weight_end is not None
        and hasattr(env_cfg.rewards, "large_pitch_angle_penalty")
    ):
        _attach_linear_reward_schedule(
            env_cfg,
            attr_name="large_pitch_angle_penalty_weight_schedule",
            term_name="large_pitch_angle_penalty",
            start_weight=env_cfg.rewards.large_pitch_angle_penalty.weight,
            end_weight=args_cli.large_pitch_angle_penalty_weight_end,
            start_step=schedule_start_step,
            end_step=schedule_end_step,
        )

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
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
        staggered_reset_buckets=args_cli.staggered_reset_buckets,
    )

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation
    simulation_app.close()
