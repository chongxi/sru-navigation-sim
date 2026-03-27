#!/usr/bin/env python3
"""Drive xlerobot with keyboard + PID yaw hold + live depth camera visualization.

Based on drive_terrain_pid.py with additions:
- Raycast depth camera on base_link (same config as NavigationEnv)
- OpenCV window showing live colorized depth image
- Arrow + sphere markers showing camera position and direction in viewport

Usage:
    python scripts/drive_terrain_pid_depth.py
    python scripts/drive_terrain_pid_depth.py --terrain_type maze --num_rows 1 --num_cols 2
    python scripts/drive_terrain_pid_depth.py --cam_debug_vis
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
    description="Drive xlerobot with keyboard + PID yaw hold + depth camera."
)

# Terrain arguments
parser.add_argument("--num_rows", type=int, default=1)
parser.add_argument("--num_cols", type=int, default=2)
parser.add_argument("--terrain_type", type=str, default="maze",
                    choices=["maze", "non_maze", "stairs", "pits"])

# Robot arguments
parser.add_argument("--usd", type=str, default=str(DEFAULT_USD))
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--spawn_height", type=float, default=0.02)
parser.add_argument("--wheel_radius", type=float, default=0.08)
parser.add_argument("--wheel_track", type=float, default=0.56)

# Drive arguments
parser.add_argument("--linear_speed", type=float, default=2.5)
parser.add_argument("--yaw_speed", type=float, default=10.0)
parser.add_argument("--settle_time", type=float, default=0.5)
parser.add_argument("--command_ramp_up_time", type=float, default=0.8)
parser.add_argument("--command_ramp_down_time", type=float, default=0.15)

# PID arguments
parser.add_argument("--yaw_pid_kp", type=float, default=40.0)
parser.add_argument("--yaw_pid_ki", type=float, default=0.0)
parser.add_argument("--yaw_pid_kd", type=float, default=3.0)
parser.add_argument("--yaw_pid_integral_limit", type=float, default=1.5)
parser.add_argument("--yaw_hold_max_correction", type=float, default=5.0)
parser.add_argument("--yaw_hold_engage_speed", type=float, default=0.10)
parser.add_argument("--yaw_rate_filter_time", type=float, default=0.3)

# Camera arguments (defaults match diff_drive/navigation_env_cfg.py)
parser.add_argument("--cam_x", type=float, default=0.22, help="Camera X offset from base_link (m).")
parser.add_argument("--cam_z", type=float, default=0.50, help="Camera Z offset from base_link (m).")
parser.add_argument("--cam_pitch", type=float, default=15.0, help="Camera downward pitch (degrees).")
parser.add_argument("--cam_max_dist", type=float, default=11.0, help="Max depth range (m).")
parser.add_argument("--cam_debug_vis", action="store_true", help="Show raycaster debug rays in viewport.")

# Debug / Friction
parser.add_argument("--debug_drive", action="store_true")
parser.add_argument("--debug_every", type=int, default=10)
parser.add_argument("--ground_static_friction", type=float, default=1.0)
parser.add_argument("--ground_dynamic_friction", type=float, default=0.9)
parser.add_argument("--ground_restitution", type=float, default=0.0)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_known_args()[0]
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after sim launch ───────────────────────────────────────────
import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCameraCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_nav_task.terrains.patches import apply_terrain_patches
apply_terrain_patches()
from isaaclab_nav_task.terrains import HfMazeTerrainCfg

# ── Create OpenCV window BEFORE anything else ─────────────────────────
cv2.namedWindow("Depth Camera (env 0)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth Camera (env 0)", 64 * 8, 40 * 8)
cv2.moveWindow("Depth Camera (env 0)", 50, 50)
print("[INFO] OpenCV depth window created.")


# ── Terrain builder ───────────────────────────────────────────────────

def build_terrain_cfg(args) -> TerrainImporterCfg:
    all_sub_terrains = {
        "maze": HfMazeTerrainCfg(
            proportion=0.3, open_probability=0.9, grid_size=(15, 15), cell_size=2.0,
            add_noise_to_flat=False, add_goal=True, randomize_wall=True,
            random_wall_ratio=0.5, add_stairs_to_maze=False,
        ),
        "non_maze": HfMazeTerrainCfg(
            proportion=0.2, open_probability=0.9, grid_size=(15, 15), cell_size=2.0,
            add_noise_to_flat=False, add_goal=True, randomize_wall=True,
            random_wall_ratio=1.0, non_maze_terrain=True,
        ),
        "stairs": HfMazeTerrainCfg(
            proportion=0.3, open_probability=0.9, grid_size=(15, 15), cell_size=2.0,
            add_noise_to_flat=False, add_goal=True, randomize_wall=False,
            random_wall_ratio=1.0, non_maze_terrain=False, stairs=True,
        ),
        "pits": HfMazeTerrainCfg(
            proportion=0.2, open_probability=0.9, grid_size=(15, 15), cell_size=2.0,
            add_noise_to_flat=False, add_goal=True, randomize_wall=True,
            random_wall_ratio=1.0, non_maze_terrain=True, dynamic_obstacles=True,
        ),
    }
    if args.terrain_type is not None:
        sub_terrains = {args.terrain_type: all_sub_terrains[args.terrain_type]}
        sub_terrains[args.terrain_type].proportion = 1.0
    else:
        sub_terrains = all_sub_terrains

    return TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(30.0, 30.0), border_width=20.0,
            num_rows=args.num_rows, num_cols=args.num_cols,
            horizontal_scale=0.1, vertical_scale=0.005,
            slope_threshold=0.75, use_cache=False,
            curriculum=False, difficulty_range=(0.5, 1.0),
            sub_terrains=sub_terrains,
        ),
        max_init_terrain_level=0, collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            restitution=args.ground_restitution,
            static_friction=args.ground_static_friction,
            dynamic_friction=args.ground_dynamic_friction,
            compliant_contact_stiffness=5e5, compliant_contact_damping=300.0,
        ),
        debug_vis=False,
    )


# ── Camera config ────────────────────────────────────────────────────

def build_camera_cfg() -> RayCasterCameraCfg:
    pitch_rad = math.radians(args_cli.cam_pitch)
    half = pitch_rad / 2.0
    qw, qy = math.cos(half), math.sin(half)
    return RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        mesh_prim_paths=["/World/ground"],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(args_cli.cam_x, 0.0, args_cli.cam_z),
            rot=(qw, 0.0, qy, 0.0), convention="world",
        ),
        data_types=["distance_to_image_plane"],
        debug_vis=args_cli.cam_debug_vis,
        max_distance=args_cli.cam_max_dist,
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_ros_camera_info(
            fx=72.7025, fy=72.7025, cx=94.4457, cy=62.5424,
            width=192, height=120, downsample_factor=3,
        ),
    )


# ── Robot + Scene ────────────────────────────────────────────────────

ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=str(Path(args_cli.usd).expanduser().resolve())),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.spawn_height)),
    actuators={
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*_joint"],
            stiffness=0.0, damping=50.0,
            effort_limit_sim=5000.0, velocity_limit_sim=200.0,
        )
    },
)


class DepthDriveSceneCfg(InteractiveSceneCfg):
    terrain = build_terrain_cfg(args_cli)
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = ROBOT_CFG
    depth_camera = build_camera_cfg()


# ── Helpers ──────────────────────────────────────────────────────────

class DifferentialDriveKeyboard(Se2Keyboard):
    def __str__(self):
        return ("Arrow keys: fwd/back/yaw, R=reset, L=clear")

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


def quat_yaw(quat_wxyz):
    return euler_xyz_from_quat(quat_wxyz)[2]

def ramp_command(current, target, dt, ramp_up, ramp_down):
    ramp_time = ramp_up if abs(target) > abs(current) else ramp_down
    if ramp_time <= 0.0: return target
    alpha = min(1.0, dt / ramp_time)
    return current + alpha * (target - current)

def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def low_pass_filter(current, measurement, dt, tau):
    if tau <= 0.0: return measurement
    alpha = min(1.0, dt / tau)
    return current + alpha * (measurement - current)

def reset_robot(scene, keyboard):
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                   robot.data.default_joint_vel.clone())
    scene.reset()
    keyboard.reset()
    print("[INFO] Robot reset.")

def depth_to_colormap(depth, max_dist):
    clean = np.nan_to_num(depth, nan=max_dist, posinf=max_dist, neginf=0.0)
    depth_norm = np.clip((clean - 0.3) / (8.0 - 0.3), 0.0, 1.0)
    depth_u8 = (255 * (1.0 - depth_norm)).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


# ── Main loop ────────────────────────────────────────────────────────

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1/120.0, render_interval=4)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, -2.5, 3.0], [0.0, 0.0, 0.0])

    scene_cfg = DepthDriveSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    depth_camera = scene["depth_camera"]

    wheel_joint_ids, wheel_joint_names = robot.find_joints(
        ["wheel_left_joint", "wheel_right_joint"], preserve_order=True
    )

    # ── Camera visualization markers ──
    # Thin red arrow showing camera forward direction
    cam_arrow_cfg = RED_ARROW_X_MARKER_CFG.copy()
    cam_arrow_cfg.prim_path = "/Visuals/CameraArrow"
    cam_arrow_cfg.markers["arrow"].scale = (0.02, 0.02, 0.15)
    cam_arrow = VisualizationMarkers(cam_arrow_cfg)

    # Small yellow sphere at camera origin
    cam_sphere_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CameraOrigin",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        },
    )
    cam_sphere = VisualizationMarkers(cam_sphere_cfg)

    keyboard = DifferentialDriveKeyboard(Se2KeyboardCfg(
        v_x_sensitivity=args_cli.linear_speed, v_y_sensitivity=0.0,
        omega_z_sensitivity=args_cli.yaw_speed, sim_device=args_cli.device,
    ))

    # PID state
    filtered_v_x = 0.0
    filtered_omega_z = 0.0
    settle_timer = max(0.0, args_cli.settle_time)
    yaw_target = None
    yaw_error_integral = 0.0
    filtered_yaw_rate = 0.0

    def handle_reset():
        nonlocal filtered_v_x, filtered_omega_z, settle_timer
        nonlocal yaw_target, yaw_error_integral, filtered_yaw_rate
        reset_robot(scene, keyboard)
        filtered_v_x = filtered_omega_z = 0.0
        settle_timer = max(0.0, args_cli.settle_time)
        yaw_target = None
        yaw_error_integral = filtered_yaw_rate = 0.0

    keyboard.add_callback("R", handle_reset)

    print("\n" + "=" * 64)
    print("  xlerobot — keyboard drive + PID yaw hold + DEPTH CAMERA")
    print("=" * 64)
    print(f"  Terrain:  {args_cli.num_rows}x{args_cli.num_cols}, type={args_cli.terrain_type}")
    print(f"  Camera:   offset=({args_cli.cam_x}, 0, {args_cli.cam_z}), pitch={args_cli.cam_pitch}deg")
    print(f"  Controls: Arrow keys (fwd/back/yaw), R=reset, L=clear")
    print("=" * 64 + "\n")

    handle_reset()
    sim_dt = sim.get_physics_dt()
    frame_count = 0
    depth_first = True

    while simulation_app.is_running():
        with torch.inference_mode():
            # ── Keyboard + PID (identical to drive_terrain_pid.py) ──
            command = keyboard.advance()
            raw_v_x = float(command[0].item())
            raw_omega_z = float(command[2].item())
            current_yaw = float(quat_yaw(robot.data.root_quat_w[0:1])[0].item())
            current_yaw_rate = float(robot.data.root_ang_vel_b[0, 2].item())
            filtered_yaw_rate = low_pass_filter(
                filtered_yaw_rate, current_yaw_rate, sim_dt, args_cli.yaw_rate_filter_time)

            if settle_timer > 0.0:
                settle_timer = max(0.0, settle_timer - sim_dt)
                filtered_v_x = filtered_omega_z = v_x = 0.0
                commanded_omega_z = 0.0
                yaw_target = current_yaw
                yaw_error_integral = filtered_yaw_rate = 0.0
            else:
                filtered_v_x = ramp_command(filtered_v_x, raw_v_x, sim_dt,
                    args_cli.command_ramp_up_time, args_cli.command_ramp_down_time)
                filtered_omega_z = ramp_command(filtered_omega_z, raw_omega_z, sim_dt,
                    args_cli.command_ramp_up_time, args_cli.command_ramp_down_time)
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
                yaw_error_integral = max(-args_cli.yaw_pid_integral_limit,
                    min(args_cli.yaw_pid_integral_limit, yaw_error_integral))
                pid_correction = (args_cli.yaw_pid_kp * yaw_error
                    + args_cli.yaw_pid_ki * yaw_error_integral
                    - args_cli.yaw_pid_kd * filtered_yaw_rate)
                pid_correction = max(-args_cli.yaw_hold_max_correction,
                    min(args_cli.yaw_hold_max_correction, pid_correction))
                commanded_omega_z = pid_correction if (moving_command_active or manual_yaw_active) else 0.0

            # ── Differential drive kinematics ──
            left_w = (v_x - 0.5 * args_cli.wheel_track * commanded_omega_z) / args_cli.wheel_radius
            right_w = (v_x + 0.5 * args_cli.wheel_track * commanded_omega_z) / args_cli.wheel_radius
            wheel_targets = torch.tensor([[left_w, right_w]], dtype=torch.float32,
                                          device=args_cli.device).repeat(scene.num_envs, 1)
            robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            # ── Camera arrow + sphere in viewport ──
            cam_pos = depth_camera.data.pos_w[0:1]       # (1, 3)
            cam_quat = depth_camera.data.quat_w_world[0:1]  # (1, 4) wxyz
            # In "world" camera convention, forward = +X.
            # RED_ARROW_X_MARKER also points along +X. Use quat directly.
            cam_arrow.visualize(cam_pos, cam_quat)
            cam_sphere.visualize(cam_pos)

            # ── Depth image display ──
            frame_count += 1
            if frame_count % 4 == 0:
                depth_data = depth_camera.data.output["distance_to_image_plane"]
                if depth_data is not None and depth_data.numel() > 0:
                    depth_img = depth_data[0].cpu().numpy()
                    if depth_img.ndim == 3:
                        depth_img = depth_img.squeeze(-1)
                    if depth_first:
                        print(f"[INFO] Depth: shape={depth_img.shape}, "
                              f"range=[{depth_img.min():.1f}, {depth_img.max():.1f}]m")
                        depth_first = False
                    color = depth_to_colormap(depth_img, args_cli.cam_max_dist)
                    h, w_ = color.shape[:2]
                    display = cv2.resize(color, (w_*8, h*8), interpolation=cv2.INTER_NEAREST)
                    valid = depth_img[~np.isnan(depth_img) & (depth_img < args_cli.cam_max_dist)]
                    if valid.size > 0:
                        txt = f"min={valid.min():.1f}m max={valid.max():.1f}m mean={valid.mean():.1f}m"
                    else:
                        txt = "no returns"
                    cv2.putText(display, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    cv2.imshow("Depth Camera (env 0)", display)
                    cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
