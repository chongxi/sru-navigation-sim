#!/usr/bin/env python3
"""Minimal diff-drive test: direct wheel commands on various terrain types.

Matches drive_terrain_pid.py setup but automated (no keyboard).
Validates that the xlerobot can drive on each terrain type.

Usage:
    python scripts/test_diff_drive_simple.py --terrain flat
    python scripts/test_diff_drive_simple.py --terrain ground   # heightfield, flat, no obstacles
    python scripts/test_diff_drive_simple.py --terrain maze
    python scripts/test_diff_drive_simple.py --terrain ground --headless --num_envs 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal diff-drive driving test.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--terrain", type=str, default="flat",
    choices=["flat", "ground", "maze"],
    help="flat=plane, ground=heightfield flat (no obstacles), maze=heightfield maze",
)
parser.add_argument("--steps", type=int, default=600, help="Number of sim steps (at 120Hz).")
parser.add_argument("--spawn_height", type=float, default=0.02, help="Spawn height above ground.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after AppLauncher ---
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.math import euler_xyz_from_quat

# Use the same USD as DIFF_DRIVE_CFG
REPO_ROOT = Path(__file__).resolve().parents[1]
USD_PATH = (
    REPO_ROOT / "isaaclab_nav_task" / "navigation" / "assets" / "data"
    / "Robots" / "xlerobot" / "xlerobot_wheel_v14.usd"
)

# Physics material — matches drive_terrain_pid.py
GROUND_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    restitution=0.0,
    static_friction=1.0,
    dynamic_friction=0.9,
    compliant_contact_stiffness=5e5,
    compliant_contact_damping=300.0,
)

# Robot config — matches drive_terrain_pid.py (no rigid_props overrides)
ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.spawn_height)),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            stiffness=0.0,
            damping=50.0,
            effort_limit_sim=5000.0,
            velocity_limit_sim=200.0,
        )
    },
)


def build_terrain_cfg():
    if args_cli.terrain == "flat":
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=GROUND_MATERIAL,
        )

    # Both "ground" and "maze" use the heightfield terrain generator
    from isaaclab_nav_task.terrains.patches import apply_terrain_patches
    apply_terrain_patches()
    from isaaclab_nav_task.terrains import HfMazeTerrainCfg

    if args_cli.terrain == "ground":
        # Flat heightfield terrain — same generator as NavigationEnv but
        # no walls, no stairs, no pits, no obstacles. Tests whether the
        # heightfield terrain system itself causes issues.
        sub_terrains = {
            "flat_ground": HfMazeTerrainCfg(
                proportion=1.0,
                open_probability=1.0,       # all cells open
                grid_size=(15, 15),
                cell_size=2.0,
                add_noise_to_flat=False,
                add_goal=False,
                randomize_wall=False,        # no random walls
                random_wall_ratio=0.0,       # zero walls
                add_stairs_to_maze=False,    # no stairs
                non_maze_terrain=True,       # non-maze (open area)
                dynamic_obstacles=False,     # no pits/obstacles
            ),
        }
    else:  # maze
        sub_terrains = {
            "maze": HfMazeTerrainCfg(
                proportion=1.0,
                open_probability=0.9,
                grid_size=(15, 15),
                cell_size=2.0,
                add_noise_to_flat=False,
                add_goal=True,
                randomize_wall=True,
                random_wall_ratio=0.5,
                add_stairs_to_maze=False,
            ),
        }

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(30.0, 30.0),
            border_width=20.0,
            num_rows=1,
            num_cols=2,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            difficulty_range=(0.5, 1.0),
            sub_terrains=sub_terrains,
        ),
        collision_group=-1,
        physics_material=GROUND_MATERIAL,
    )


class SimpleSceneCfg(InteractiveSceneCfg):
    terrain = build_terrain_cfg()
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = ROBOT_CFG


def main():
    # 120Hz physics, same as drive_terrain_pid.py
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1 / 120.0,
        render_interval=4,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, -2.5, 3.0], [0.0, 0.0, 0.0])

    scene_cfg = SimpleSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    wheel_ids, wheel_names = robot.find_joints(
        ["wheel_left_joint", "wheel_right_joint"], preserve_order=True
    )
    print(f"Wheel joints: {wheel_names}, ids: {wheel_ids}")
    print(f"Terrain type: {args_cli.terrain}")

    N = scene.num_envs
    device = args_cli.device
    sim_dt = sim.get_physics_dt()

    # --- Settle phase (0.5s) ---
    settle_steps = int(0.5 / sim_dt)
    print(f"\n[1/3] Settling for {settle_steps} steps ({0.5}s)...")
    for _ in range(settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # Record settled position
    settled_pos = robot.data.root_pos_w.clone()
    _, _, settled_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    for i in range(min(N, 4)):
        print(
            f"  env {i}: settled z={settled_pos[i, 2].item():.4f}  "
            f"yaw={settled_yaw[i].item():+.3f} rad"
        )

    # --- Drive forward ---
    drive_steps = args_cli.steps
    vx_target = 1.0  # m/s
    wheel_radius = 0.08
    target_wheel_vel = vx_target / wheel_radius  # both wheels equal = straight

    print(f"\n[2/3] Driving forward for {drive_steps} steps ({drive_steps * sim_dt:.1f}s)...")
    print(f"  Target: vx={vx_target} m/s, wheel_vel={target_wheel_vel:.1f} rad/s")
    print()

    for step in range(drive_steps):
        # Equal wheel velocities = straight line, no PID needed
        wheel_targets = torch.full((N, 2), target_wheel_vel, device=device)
        robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if step % 100 == 0:
            pos = robot.data.root_pos_w[0]
            vx_body = robot.data.root_lin_vel_b[0, 0].item()
            _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
            wl = robot.data.joint_vel[0, wheel_ids[0]].item()
            wr = robot.data.joint_vel[0, wheel_ids[1]].item()
            dx = pos[0].item() - settled_pos[0, 0].item()
            dy = pos[1].item() - settled_pos[0, 1].item()
            dist = (dx**2 + dy**2) ** 0.5
            print(
                f"  step {step:4d}: z={pos[2].item():+.4f} "
                f"vx_body={vx_body:+.3f} "
                f"wheels=({wl:+.2f},{wr:+.2f}) "
                f"yaw={yaw.item():+.3f} "
                f"dist={dist:.2f}m"
            )

    # --- Results ---
    print(f"\n[3/3] Results:")
    final_pos = robot.data.root_pos_w.clone()
    n_pass = 0
    for i in range(N):
        dx = final_pos[i, 0].item() - settled_pos[i, 0].item()
        dy = final_pos[i, 1].item() - settled_pos[i, 1].item()
        dist = (dx**2 + dy**2) ** 0.5
        expected = vx_target * drive_steps * sim_dt
        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[i:i+1])
        ok = dist > expected * 0.5
        if ok:
            n_pass += 1
        if N <= 16 or not ok:
            print(
                f"  env {i}: dist={dist:.2f}m (expected ~{expected:.1f}m) "
                f"z={final_pos[i, 2].item():.4f} "
                f"yaw_drift={yaw.item() - settled_yaw[i].item():+.3f} rad  "
                f"{'PASS' if ok else 'FAIL'}"
            )
    print(f"\n  {n_pass}/{N} passed")

    scene.reset()
    sim.clear_all_callbacks()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
