#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Drive the xlerobot on generated maze terrain using keyboard input with PID yaw hold.

Combines the terrain generation from view_terrain.py with the differential-drive
keyboard control from xlerobot_keyboard_drive_yaw_pid.py.

Usage:
    python scripts/drive_terrain_pid.py
    python scripts/drive_terrain_pid.py --terrain_type maze --num_rows 1 --num_cols 2
    python scripts/drive_terrain_pid.py --terrain_type stairs --linear_speed 1.5
    python scripts/drive_terrain_pid.py --usd path/to/robot.usd --debug_drive
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

# ── Paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = (
    REPO_ROOT / "isaaclab_nav_task" / "navigation" / "assets" / "data"
    / "Robots" / "xlerobot" / "xlerobot_wheel_v14.usd"
)

# ── CLI ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Drive xlerobot on maze terrain with keyboard + PID yaw hold."
)

# Terrain arguments
parser.add_argument("--num_rows", type=int, default=2, help="Terrain grid rows (difficulty levels).")
parser.add_argument("--num_cols", type=int, default=4, help="Terrain grid columns (variations).")
parser.add_argument(
    "--terrain_type", type=str, default=None,
    choices=["maze", "non_maze", "stairs", "pits"],
    help="Show only one terrain type (default: all).",
)

# Robot arguments
parser.add_argument("--usd", type=str, default=str(DEFAULT_USD), help="Path to the robot USD.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--spawn_height", type=float, default=0.20, help="Initial height above terrain surface.")
parser.add_argument("--wheel_radius", type=float, default=0.08, help="Wheel radius in meters.")
parser.add_argument("--wheel_track", type=float, default=0.56, help="Distance between left and right wheel centres.")

# Drive arguments
parser.add_argument("--linear_speed", type=float, default=2.5, help="Forward/backward speed (m/s).")
parser.add_argument("--yaw_speed", type=float, default=10.0, help="Yaw rate (rad/s).")
parser.add_argument("--settle_time", type=float, default=0.5, help="Settle time after reset (s).")
parser.add_argument("--command_ramp_up_time", type=float, default=0.8, help="Ramp-up time constant (s).")
parser.add_argument("--command_ramp_down_time", type=float, default=0.15, help="Ramp-down time constant (s).")

# PID arguments
parser.add_argument("--yaw_pid_kp", type=float, default=40.0, help="Yaw-hold P gain.")
parser.add_argument("--yaw_pid_ki", type=float, default=0.0, help="Yaw-hold I gain.")
parser.add_argument("--yaw_pid_kd", type=float, default=4.0, help="Yaw-hold D gain.")
parser.add_argument("--yaw_pid_integral_limit", type=float, default=1.5, help="Integral clamp.")
parser.add_argument("--yaw_hold_max_correction", type=float, default=5.0, help="Max yaw correction (rad/s).")
parser.add_argument("--yaw_hold_engage_speed", type=float, default=0.10, help="Min |v_x| to engage yaw hold.")
parser.add_argument("--yaw_rate_filter_time", type=float, default=0.3, help="Low-pass filter τ for yaw rate (s).")

# Debug
parser.add_argument("--debug_drive", action="store_true", help="Print drive diagnostics.")
parser.add_argument("--debug_every", type=int, default=10, help="Diagnostic print interval (sim steps).")

# Friction
parser.add_argument("--ground_static_friction", type=float, default=1.0)
parser.add_argument("--ground_dynamic_friction", type=float, default=0.9)
parser.add_argument("--ground_restitution", type=float, default=0.0)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_known_args()[0]
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after sim launch ───────────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.math import euler_xyz_from_quat

# Apply terrain patches (mesh optimisation + height-field mask storage)
from isaaclab_nav_task.terrains.patches import apply_terrain_patches
apply_terrain_patches()

from isaaclab_nav_task.terrains import HfMazeTerrainCfg


# ── Terrain builder ───────────────────────────────────────────────────

def build_terrain_cfg(args) -> TerrainImporterCfg:
    """Build a TerrainImporterCfg from CLI arguments."""
    all_sub_terrains = {
        "maze": HfMazeTerrainCfg(
            proportion=0.3, open_probability=0.9,
            grid_size=(15, 15), cell_size=2.0,
            add_noise_to_flat=False, add_goal=True,
            randomize_wall=True, random_wall_ratio=0.5,
            add_stairs_to_maze=True,
        ),
        "non_maze": HfMazeTerrainCfg(
            proportion=0.2, open_probability=0.9,
            grid_size=(15, 15), cell_size=2.0,
            add_noise_to_flat=False, add_goal=True,
            randomize_wall=True, random_wall_ratio=1.0,
            non_maze_terrain=True,
        ),
        "stairs": HfMazeTerrainCfg(
            proportion=0.3, open_probability=0.9,
            grid_size=(15, 15), cell_size=2.0,
            add_noise_to_flat=False, add_goal=True,
            randomize_wall=False, random_wall_ratio=1.0,
            non_maze_terrain=False, stairs=True,
        ),
        "pits": HfMazeTerrainCfg(
            proportion=0.2, open_probability=0.9,
            grid_size=(15, 15), cell_size=2.0,
            add_noise_to_flat=False, add_goal=True,
            randomize_wall=True, random_wall_ratio=1.0,
            non_maze_terrain=True, dynamic_obstacles=True,
        ),
    }

    if args.terrain_type is not None:
        sub_terrains = {args.terrain_type: all_sub_terrains[args.terrain_type]}
        sub_terrains[args.terrain_type].proportion = 1.0
    else:
        sub_terrains = all_sub_terrains

    terrain_gen = TerrainGeneratorCfg(
        size=(30.0, 30.0),
        border_width=20.0,
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        difficulty_range=(0.5, 1.0),
        sub_terrains=sub_terrains,
    )

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            restitution=args.ground_restitution,
            static_friction=args.ground_static_friction,
            dynamic_friction=args.ground_dynamic_friction,
            compliant_contact_stiffness=5e5,
            compliant_contact_damping=300.0,
        ),
        debug_vis=False,
    )


# ── Robot config ──────────────────────────────────────────────────────

ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=str(Path(args_cli.usd).expanduser().resolve())),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.spawn_height)),
    actuators={
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*_joint"],
            stiffness=0.0,
            damping=50.0,
            effort_limit_sim=5000.0,
            velocity_limit_sim=200.0,
        )
    },
)


# ── Scene: terrain + robot + light ────────────────────────────────────

class TerrainDriveSceneCfg(InteractiveSceneCfg):
    terrain = build_terrain_cfg(args_cli)

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = ROBOT_CFG


# ── Keyboard ──────────────────────────────────────────────────────────

class DifferentialDriveKeyboard(Se2Keyboard):
    """Keyboard mapping for differential-drive base."""

    def __str__(self) -> str:
        msg = f"Keyboard Controller for SE(2): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset all commands: L\n"
        msg += "\tMove forward:  Arrow Up\n"
        msg += "\tMove backward: Arrow Down\n"
        msg += "\tYaw left:      Arrow Left\n"
        msg += "\tYaw right:     Arrow Right\n"
        msg += "\tReset robot:   R"
        return msg

    def _create_key_bindings(self):
        self._INPUT_KEY_MAPPING = {
            "NUMPAD_8": (self.v_x_sensitivity, 0.0, 0.0),
            "UP":       (self.v_x_sensitivity, 0.0, 0.0),
            "NUMPAD_2": (-self.v_x_sensitivity, 0.0, 0.0),
            "DOWN":     (-self.v_x_sensitivity, 0.0, 0.0),
            "NUMPAD_4": (0.0, 0.0, self.omega_z_sensitivity),
            "LEFT":     (0.0, 0.0, self.omega_z_sensitivity),
            "NUMPAD_6": (0.0, 0.0, -self.omega_z_sensitivity),
            "RIGHT":    (0.0, 0.0, -self.omega_z_sensitivity),
        }


# ── Helpers ───────────────────────────────────────────────────────────

def quat_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return euler_xyz_from_quat(quat_wxyz)[2]


def ramp_command(current: float, target: float, dt: float, ramp_up: float, ramp_down: float) -> float:
    ramp_time = ramp_up if abs(target) > abs(current) else ramp_down
    if ramp_time <= 0.0:
        return target
    alpha = min(1.0, dt / ramp_time)
    return current + alpha * (target - current)


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def low_pass_filter(current: float, measurement: float, dt: float, tau: float) -> float:
    if tau <= 0.0:
        return measurement
    alpha = min(1.0, dt / tau)
    return current + alpha * (measurement - current)


def reset_robot(scene: InteractiveScene, keyboard: Se2Keyboard) -> None:
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()
    keyboard.reset()
    print("[INFO] Robot reset.")


# ── Main loop ─────────────────────────────────────────────────────────

def main() -> None:
    # Simulation context — 200 Hz physics for smooth wheel-on-mesh contact
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1 / 120.0,
        render_interval=4,  # render at 50 Hz (every 4th physics step)
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, -2.5, 3.0], [0.0, 0.0, 0.0])

    # Build scene (terrain + robot)
    scene_cfg = TerrainDriveSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # Find wheel joints
    robot = scene["robot"]
    wheel_joint_ids, wheel_joint_names = robot.find_joints(
        ["wheel_left_joint", "wheel_right_joint"], preserve_order=True
    )
    if len(wheel_joint_ids) != 2:
        raise RuntimeError(f"Expected 2 wheel joints, found {wheel_joint_names}.")

    # Keyboard
    keyboard = DifferentialDriveKeyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.linear_speed,
            v_y_sensitivity=0.0,
            omega_z_sensitivity=args_cli.yaw_speed,
            sim_device=args_cli.device,
        )
    )

    # PID state
    filtered_v_x = 0.0
    filtered_omega_z = 0.0
    settle_timer = max(0.0, args_cli.settle_time)
    yaw_target: float | None = None
    yaw_error_integral = 0.0
    filtered_yaw_rate = 0.0

    def handle_reset() -> None:
        nonlocal filtered_v_x, filtered_omega_z, settle_timer
        nonlocal yaw_target, yaw_error_integral, filtered_yaw_rate
        reset_robot(scene, keyboard)
        filtered_v_x = 0.0
        filtered_omega_z = 0.0
        settle_timer = max(0.0, args_cli.settle_time)
        yaw_target = None
        yaw_error_integral = 0.0
        filtered_yaw_rate = 0.0

    keyboard.add_callback("R", handle_reset)

    # Print info
    print("\n" + "=" * 64)
    print("  xlerobot on maze terrain — keyboard drive + PID yaw hold")
    print("=" * 64)
    print(f"  Terrain:  {args_cli.num_rows}×{args_cli.num_cols} grid, 30m×30m tiles")
    if args_cli.terrain_type:
        print(f"  Type:     {args_cli.terrain_type}")
    else:
        print(f"  Types:    maze, non_maze, stairs, pits")
    print(f"  Robot:    {Path(args_cli.usd).name}")
    print(f"  Wheels:   {wheel_joint_names}")
    print(f"  Controls: Arrow keys (fwd/back/yaw), R=reset, L=clear")
    print(f"  PID:      kp={args_cli.yaw_pid_kp}, ki={args_cli.yaw_pid_ki}, kd={args_cli.yaw_pid_kd}")
    print("=" * 64 + "\n")

    handle_reset()

    root_pos = robot.data.root_pos_w[0]
    print(
        f"[INFO] Spawn position: x={root_pos[0].item():.3f}  "
        f"y={root_pos[1].item():.3f}  z={root_pos[2].item():.3f}"
    )

    # Sim loop
    sim_dt = sim.get_physics_dt()
    debug_every = max(1, args_cli.debug_every)
    debug_step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            command = keyboard.advance()
            raw_v_x = float(command[0].item())
            raw_omega_z = float(command[2].item())
            current_yaw = float(quat_yaw(robot.data.root_quat_w[0:1])[0].item())
            current_yaw_rate = float(robot.data.root_ang_vel_b[0, 2].item())
            filtered_yaw_rate = low_pass_filter(
                filtered_yaw_rate, current_yaw_rate, sim_dt, args_cli.yaw_rate_filter_time
            )

            if settle_timer > 0.0:
                settle_timer = max(0.0, settle_timer - sim_dt)
                filtered_v_x = 0.0
                filtered_omega_z = 0.0
                v_x = 0.0
                commanded_omega_z = 0.0
                yaw_target = current_yaw
                yaw_error_integral = 0.0
                filtered_yaw_rate = 0.0
            else:
                filtered_v_x = ramp_command(
                    filtered_v_x, raw_v_x, sim_dt,
                    args_cli.command_ramp_up_time, args_cli.command_ramp_down_time,
                )
                filtered_omega_z = ramp_command(
                    filtered_omega_z, raw_omega_z, sim_dt,
                    args_cli.command_ramp_up_time, args_cli.command_ramp_down_time,
                )
                v_x = filtered_v_x

                manual_yaw_active = abs(filtered_omega_z) > 1e-4
                moving_command_active = abs(filtered_v_x) > args_cli.yaw_hold_engage_speed

                if yaw_target is None:
                    yaw_target = current_yaw

                if manual_yaw_active:
                    yaw_target = wrap_to_pi(yaw_target + filtered_omega_z * sim_dt)
                    yaw_error_integral = 0.0
                elif not moving_command_active:
                    yaw_error_integral = 0.0

                yaw_error = wrap_to_pi(yaw_target - current_yaw)
                yaw_error_integral += yaw_error * sim_dt
                yaw_error_integral = max(
                    -args_cli.yaw_pid_integral_limit,
                    min(args_cli.yaw_pid_integral_limit, yaw_error_integral),
                )
                pid_correction = (
                    args_cli.yaw_pid_kp * yaw_error
                    + args_cli.yaw_pid_ki * yaw_error_integral
                    - args_cli.yaw_pid_kd * filtered_yaw_rate
                )
                pid_correction = max(
                    -args_cli.yaw_hold_max_correction,
                    min(args_cli.yaw_hold_max_correction, pid_correction),
                )

                if moving_command_active or manual_yaw_active:
                    commanded_omega_z = pid_correction
                else:
                    commanded_omega_z = 0.0

            # Differential-drive kinematics
            left_w = (v_x - 0.5 * args_cli.wheel_track * commanded_omega_z) / args_cli.wheel_radius
            right_w = (v_x + 0.5 * args_cli.wheel_track * commanded_omega_z) / args_cli.wheel_radius

            wheel_targets = torch.tensor(
                [[left_w, right_w]], dtype=torch.float32, device=args_cli.device
            ).repeat(scene.num_envs, 1)

            robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            # Debug output
            if args_cli.debug_drive:
                debug_step += 1
                command_active = abs(raw_v_x) > 1e-4 or abs(raw_omega_z) > 1e-4 or settle_timer > 0.0
                if command_active and debug_step % debug_every == 0:
                    measured_joint_vel = robot.data.joint_vel[0, wheel_joint_ids]
                    root_lin_vel = robot.data.root_lin_vel_b[0]
                    root_ang_vel = robot.data.root_ang_vel_b[0]
                    yaw_after = float(quat_yaw(robot.data.root_quat_w[0:1])[0].item())
                    yaw_t = current_yaw if yaw_target is None else yaw_target
                    yaw_err = wrap_to_pi(yaw_t - yaw_after)
                    print(
                        "[DEBUG] "
                        f"raw(vx={raw_v_x:+.3f}, wz={raw_omega_z:+.3f}) "
                        f"filt(vx={v_x:+.3f}, wz={filtered_omega_z:+.3f}) "
                        f"yaw_tgt={yaw_t:+.3f} err={yaw_err:+.3f} corr={commanded_omega_z:+.3f} "
                        f"w_tgt=({left_w:+.1f},{right_w:+.1f}) "
                        f"w_meas=({measured_joint_vel[0].item():+.1f},{measured_joint_vel[1].item():+.1f}) "
                        f"vel=({root_lin_vel[0].item():+.2f},{root_lin_vel[1].item():+.2f}) "
                        f"yaw={yaw_after:+.3f}"
                    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
