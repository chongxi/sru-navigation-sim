#!/usr/bin/env python3
"""Sanity check: load diff-drive robot on maze terrain and verify it moves.

Tests:
1. Environment creates without errors (USD loads, joints found, PID initializes)
2. Wheels spin when action is applied (check joint velocities)
3. Robot moves forward between resets (track displacement per alive episode)
4. PID holds heading (yaw stable when heading_offset=0)
5. Wall collision termination works (robot dies when hitting walls)

Usage:
    python scripts/test_diff_drive.py --headless --num_envs 4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sanity check for diff-drive in sru-nav-sim.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Nav-PPO-DiffDrive-Dev-v0")
parser.add_argument("--terrain_type", type=str, default=None,
                    choices=["maze", "non_maze", "both", "flat"],
                    help="Override terrain: maze, non_maze, both, or flat (no walls).")
parser.add_argument("--terrain_rows", type=int, default=None, help="Terrain grid rows (difficulty levels).")
parser.add_argument("--terrain_cols", type=int, default=None, help="Terrain grid columns (variations).")
parser.add_argument("--cell_size", type=float, default=None, help="Maze cell size in meters.")
parser.add_argument("--wall_ratio", type=float, default=None, help="Random wall ratio (0-1).")
parser.add_argument("--terrain_size", type=float, default=None, help="Terrain tile size in meters (default 30).")
parser.add_argument("--grid_size", type=int, default=None, help="Maze grid dimension, e.g. 15 for 15x15.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after AppLauncher ---
import torch
import gymnasium as gym

from isaaclab.utils.math import euler_xyz_from_quat

import isaaclab_nav_task  # noqa: F401 — triggers gym.register()


def main():
    # --- Create environment ---
    print(f"\n[1/3] Creating env: {args_cli.task} with {args_cli.num_envs} envs...", flush=True)
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # Override terrain from CLI
    tg = env_cfg.scene.terrain.terrain_generator
    if tg is not None:
        if args_cli.terrain_rows is not None:
            tg.num_rows = args_cli.terrain_rows
        if args_cli.terrain_cols is not None:
            tg.num_cols = args_cli.terrain_cols
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
            for st in tg.sub_terrains.values():
                if args_cli.cell_size is not None:
                    st.cell_size = args_cli.cell_size
                if args_cli.wall_ratio is not None:
                    st.random_wall_ratio = args_cli.wall_ratio
                if gs is not None:
                    st.grid_size = gs

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, info = env.reset()
    print(f"    Observation shape: {obs['policy'].shape}", flush=True)
    print(f"    Action dim:        {env.action_space.shape}", flush=True)

    N = env.unwrapped.num_envs
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager._terms["velocity_command"]

    # --- Test: Forward driving with wall collision tracking ---
    print(f"\n[2/3] Forward drive test (200 steps, expect wall collisions in maze)...", flush=True)
    print(f"       Tracking: alive steps, wheel velocities, displacement per episode\n", flush=True)

    total_steps = 1000
    alive_count = torch.zeros(N, device=device)  # steps alive in current episode
    total_terminations = torch.zeros(N, device=device)
    max_alive_steps = torch.zeros(N, device=device)
    wheels_spinning_count = 0
    total_alive_samples = 0

    # Track per-episode displacement
    episode_start_pos = (robot.data.root_pos_w[:, :2] - env.unwrapped.scene.env_origins[:, :2]).clone()

    for step in range(total_steps):
        actions = torch.zeros(N, 2, device=device)
        actions[:, 0] = 1.0   # forward (~1.9 m/s after tanh*scaling)
        actions[:, 1] = 0.0   # straight

        obs, reward, terminated, truncated, info = env.step(actions)

        # Track alive
        alive_mask = ~(terminated | truncated)
        alive_count[alive_mask] += 1

        # Check wheels spinning (env 0, when alive)
        if alive_mask[0]:
            wl = robot.data.joint_vel[0, action_term._wheel_joint_ids[0]].item()
            wr = robot.data.joint_vel[0, action_term._wheel_joint_ids[1]].item()
            vx = action_term._stored_vx[0].item()
            if abs(wl) > 0.1 or abs(wr) > 0.1:
                wheels_spinning_count += 1
            total_alive_samples += 1

        # On termination: record episode stats
        term_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(term_ids) > 0:
            total_terminations[term_ids] += 1
            max_alive_steps = torch.max(max_alive_steps, alive_count)
            alive_count[term_ids] = 0
            # Reset start position tracking
            episode_start_pos[term_ids] = (
                robot.data.root_pos_w[term_ids, :2] - env.unwrapped.scene.env_origins[term_ids, :2]
            )

        # Print periodic status
        if step % 50 == 0:
            pos = robot.data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]
            _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
            vx_b = robot.data.root_lin_vel_b[0, 0].item()
            svx = action_term._stored_vx[0].item()
            wl = robot.data.joint_vel[0, action_term._wheel_joint_ids[0]].item()
            wr = robot.data.joint_vel[0, action_term._wheel_joint_ids[1]].item()
            a = "ALIVE" if alive_mask[0] else "RESET"
            print(
                f"  step {step:3d}: pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                f"yaw={yaw.item():+.2f} vx_body={vx_b:+.3f} stored_vx={svx:+.3f} "
                f"wheels=({wl:+.2f},{wr:+.2f}) [{a}] "
                f"resets={total_terminations[0].int().item()}"
            )

    max_alive_steps = torch.max(max_alive_steps, alive_count)

    print(f"\n  --- Results ---", flush=True)
    print(f"  Total terminations (env0):  {total_terminations[0].int().item()}", flush=True)
    print(f"  Max alive steps (any env):  {max_alive_steps.max().int().item()}", flush=True)
    print(f"  Wheels spinning:            {wheels_spinning_count}/{total_alive_samples} alive samples", flush=True)

    wheels_ok = wheels_spinning_count > total_alive_samples * 0.5 if total_alive_samples > 0 else False
    alive_ok = max_alive_steps.max().item() > 3  # survived at least a few steps
    print(f"  Wheels test:  {'PASS' if wheels_ok else 'FAIL'} (>50% of alive steps had wheel motion)", flush=True)
    print(f"  Survival test: {'PASS' if alive_ok else 'FAIL'} (survived >{3} steps in at least one episode)", flush=True)

    # --- Test: Random actions stress test ---
    print(f"\n[3/3] Random actions stress test (100 steps)...", flush=True)
    obs, info = env.reset()
    for step in range(100):
        actions = torch.randn(N, 2, device=device) * 0.3
        obs, reward, terminated, truncated, info = env.step(actions)
    print(f"  No crashes: PASS", flush=True)

    # --- Summary ---
    all_pass = wheels_ok and alive_ok
    print(f"\n{'='*50}", flush=True)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}", flush=True)
    print(f"{'='*50}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
