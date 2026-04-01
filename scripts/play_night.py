#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Play a trained navigation policy (PPO/MDPO) with standalone-scene night lighting.

Same as play.py but replaces the HDR sky light and terrain material setup to
match the scene used in drive_terrain_pid_depth.py.

Usage:
    python scripts/play_night.py --task <task_name> [options]

Arguments:
    --task                   Task name (required, typically *-Play-v0 variant)
    --checkpoint             Path to model checkpoint (.pt file)
    --use_last_checkpoint    Use latest checkpoint from logs (default behavior)
    --num_envs              Number of parallel environments
    --video                 Enable video recording
    --video_length          Video length in steps (default: 200)

Examples:
    python scripts/play_night.py --task Isaac-Navigation-B2W-Play-v0
    python scripts/play_night.py --task Isaac-Navigation-B2W-Play-v0 --checkpoint path/to/model.pt

    python3 scripts/play_night.py \
      --task Isaac-Nav-MDPO-DiffDrive-Play-v0 \
      --checkpoint logs/rsl_rl/diff_drive_navigation_mdpo/2026-03-30_09-39-36/model_850.pt \
      --num_envs 1

Note: Automatically finds latest checkpoint if --checkpoint not specified.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained navigation policy with RSL-RL (night mode).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
parser.add_argument("--episode_length", type=float, default=None, help="Episode length in seconds (overrides config).")
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
parser.add_argument("--debug_raycast", action="store_true", default=True, help="Enable raycast camera debug visualization.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras
args_cli.enable_cameras = True

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching simulation
import gymnasium as gym
import os
import re
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from rsl_rl.runners import OnPolicyRunner

# Import Isaac Lab extensions
import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_nav_task.vecenv_wrapper import SruRslRlVecEnvWrapper


def find_latest_checkpoint(log_path: str, checkpoint_pattern: str = "model_.*.pt") -> str:
    """Find the latest checkpoint file in the log directory.

    Args:
        log_path: Base log directory path
        checkpoint_pattern: Regex pattern for checkpoint files

    Returns:
        Path to the latest checkpoint file
    """
    # Find all run directories
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

    run_dirs = []
    for entry in os.scandir(log_path):
        if entry.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", entry.name):
            run_dirs.append(entry.name)

    if not run_dirs:
        raise ValueError(f"No run directories found in: {log_path}")

    # Sort to get latest run
    run_dirs.sort()
    latest_run = run_dirs[-1]
    run_path = os.path.join(log_path, latest_run)

    # Find checkpoint files
    checkpoint_files = []
    for f in os.listdir(run_path):
        if re.match(checkpoint_pattern, f):
            checkpoint_files.append(f)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files matching '{checkpoint_pattern}' found in: {run_path}")

    # Sort to get latest checkpoint
    checkpoint_files.sort(key=lambda m: f"{m:0>15}")
    latest_checkpoint = checkpoint_files[-1]

    return os.path.join(run_path, latest_checkpoint)


def load_checkpoint_with_fallback(runner: OnPolicyRunner, checkpoint_path: str, load_optimizer: bool = True):
    """Load checkpoint with fallback for PyTorch compatibility issues.

    Args:
        runner: RSL-RL runner instance
        checkpoint_path: Path to checkpoint file
        load_optimizer: Whether to load optimizer state
    """
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint to CPU first for compatibility
    loaded_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load model state - handle both standard algorithms (PPO) and MDPO
    if runner.is_mdpo:
        # MDPO uses two actor-critics, load same state into both
        runner.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    else:
        # Standard algorithms use one actor-critic
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)

    # Load normalizers if using empirical normalization
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

    # Load optimizer if requested
    if load_optimizer:
        if runner.is_mdpo:
            runner.alg.optimizer_1.load_state_dict(loaded_dict["optimizer_state_dict"])
        else:
            runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

    runner.current_learning_iteration = loaded_dict["iter"]
    print(f"[INFO] Loaded checkpoint from iteration {loaded_dict['iter']}")


def export_policy_jit(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as JIT module to an 'export' folder next to the checkpoint.

    Args:
        runner: RSL-RL runner instance with loaded policy
        checkpoint_path: Path to the checkpoint file (used to determine export location)
    """
    # Determine export directory (create 'export' folder in the same directory as checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    # Get the actor-critic module
    if runner.is_mdpo:
        actor_critic = runner.alg.actor_critic_1
    else:
        actor_critic = runner.alg.actor_critic

    # Get normalizer if using empirical normalization
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    # Export using the module's export_jit method
    print(f"[INFO] Exporting JIT policy to: {export_dir}")
    actor_critic.export_jit(path=export_dir, filename="policy.pt", normalizer=normalizer)
    print(f"[INFO] JIT export complete!")


def export_policy_onnx(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as ONNX model to an 'export' folder next to the checkpoint.

    Args:
        runner: RSL-RL runner instance with loaded policy
        checkpoint_path: Path to the checkpoint file (used to determine export location)
    """
    # Determine export directory (create 'export' folder in the same directory as checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    # Get the actor-critic module
    if runner.is_mdpo:
        actor_critic = runner.alg.actor_critic_1
    else:
        actor_critic = runner.alg.actor_critic

    # Get normalizer if using empirical normalization
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    # Check if the module has export_onnx method
    if not hasattr(actor_critic, "export_onnx"):
        raise NotImplementedError(
            f"ONNX export not implemented for {type(actor_critic).__name__}. "
            "Please add an export_onnx method to this module."
        )

    # Export using the module's export_onnx method
    print(f"[INFO] Exporting ONNX policy to: {export_dir}")
    actor_critic.export_onnx(path=export_dir, filename="policy.onnx", normalizer=normalizer)
    print(f"[INFO] ONNX export complete!")


def main():
    """Play navigation policy with RSL-RL (night mode with standalone-scene lighting)."""
    # Parse command-line arguments
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")

    # Instantiate the configs
    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    agent_cfg = agent_cfg_class()

    # --- Night mode: match the standalone terrain driving scene ---
    # Drop the HDR sky, use the same dome light as drive_terrain_pid_depth.py,
    # and replace the base env's marble terrain material with the standalone
    # terrain settings.
    env_cfg.scene.sky_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
    env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        restitution=0.0,
        static_friction=1.0,
        dynamic_friction=0.9,
        compliant_contact_stiffness=5e5,
        compliant_contact_damping=300.0,
    )
    env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.01, 0.01, 0.01),
    )
    env_cfg.sim.physics_material = env_cfg.scene.terrain.physics_material

    # Override config from command line
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

            default_base = next(iter(tg.sub_terrains.values()))

            def _subterrain_or_default(name: str):
                return tg.sub_terrains.get(name, default_base)

            if args_cli.terrain_type == "maze":
                base = _subterrain_or_default("maze")
                cs = args_cli.cell_size or getattr(base, "cell_size", 2.0)
                wr = (
                    args_cli.wall_ratio
                    if args_cli.wall_ratio is not None
                    else getattr(base, "random_wall_ratio", 0.5)
                )
                _gs = gs or getattr(base, "grid_size", (15, 15))
                tg.sub_terrains = {
                    "maze": HfMazeTerrainCfg(
                        proportion=1.0,
                        open_probability=getattr(base, "open_probability", 0.9),
                        grid_size=_gs,
                        cell_size=cs,
                        add_noise_to_flat=False,
                        add_goal=True,
                        randomize_wall=True,
                        random_wall_ratio=wr,
                        add_stairs_to_maze=getattr(base, "add_stairs_to_maze", False),
                    ),
                }
            elif args_cli.terrain_type == "non_maze":
                base = _subterrain_or_default("non_maze")
                cs = args_cli.cell_size or getattr(base, "cell_size", 2.0)
                wr = (
                    args_cli.wall_ratio
                    if args_cli.wall_ratio is not None
                    else getattr(base, "random_wall_ratio", 0.5)
                )
                _gs = gs or getattr(base, "grid_size", (15, 15))
                tg.sub_terrains = {
                    "non_maze": HfMazeTerrainCfg(
                        proportion=1.0,
                        open_probability=getattr(base, "open_probability", 0.9),
                        grid_size=_gs,
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
                base = _subterrain_or_default("non_maze")
                cs = args_cli.cell_size or getattr(base, "cell_size", 2.0)
                _gs = gs or getattr(base, "grid_size", (15, 15))
                tg.sub_terrains = {
                    "flat": HfMazeTerrainCfg(
                        proportion=1.0,
                        open_probability=1.0,
                        grid_size=_gs,
                        cell_size=cs,
                        add_noise_to_flat=False,
                        add_goal=True,
                        randomize_wall=False,
                        random_wall_ratio=0.0,
                        non_maze_terrain=True,
                        dynamic_obstacles=False,
                    ),
                }
            elif args_cli.terrain_type == "both":
                pass
        else:
            for st in tg.sub_terrains.values():
                if args_cli.cell_size is not None:
                    st.cell_size = args_cli.cell_size
                if args_cli.wall_ratio is not None:
                    st.random_wall_ratio = args_cli.wall_ratio
                if gs is not None:
                    st.grid_size = gs
        if args_cli.terrain_type == "both":
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

    # Get checkpoint path
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        # Get last checkpoint from log directory
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = find_latest_checkpoint(log_root_path, checkpoint_pattern="model_.*.pt")

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Load checkpoint with compatibility handling
    load_checkpoint_with_fallback(runner, resume_path)

    # Export JIT if requested
    if args_cli.export_jit:
        export_policy_jit(runner, resume_path)

    # Export ONNX if requested
    if args_cli.export_onnx:
        export_policy_onnx(runner, resume_path)

    # Obtain policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Reset environment
    obs, _ = env.get_observations()

    # Simulate environment
    while simulation_app.is_running():
        # Run policy
        with torch.inference_mode():
            actions = policy(obs)
        # Step environment
        obs, _, _, _ = env.step(actions)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation
    simulation_app.close()
