#!/usr/bin/env python3
"""Load Astera.usd, spawn xlerobot at the origin, and drive it with keyboard + PID yaw hold.

This mirrors the control loop from drive_terrain_pid_depth.py, but replaces the
generated terrain with a referenced USD scene. The Astera mesh is given runtime
triangle-mesh collision so the robot can rest and drive on it.

Usage:
    python scripts/drive_astera_pid_depth.py
    python scripts/drive_astera_pid_depth.py --scene_usd /path/to/Astera.usd
    python scripts/drive_astera_pid_depth.py --spawn_height 0.35
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_USD = REPO_ROOT / "Astera_upd.usd"
DEFAULT_ROBOT_USD = (
    REPO_ROOT / "isaaclab_nav_task" / "navigation" / "assets" / "data"
    / "Robots" / "xlerobot" / "xlerobot_wheel_v14.usd"
)
def resolve_prim_path(parent_prim_path: str, child_prim_path: str) -> str:
    if child_prim_path.startswith("/"):
        return child_prim_path
    return f"{parent_prim_path.rstrip('/')}/{child_prim_path.lstrip('/')}"


parser = argparse.ArgumentParser(
    description="Drive xlerobot inside Astera.usd with keyboard + PID yaw hold + depth camera."
)

# Scene arguments
parser.add_argument("--scene_usd", type=str, default=str(DEFAULT_SCENE_USD), help="Path to the scene USD.")
parser.add_argument("--scene_prim_path", type=str, default="/World/Astera", help="Prim path for the referenced scene.")
parser.add_argument(
    "--scene_mesh_prim",
    type=str,
    default="Cube",
    help="Depth raycast mesh prim path. Relative values are resolved under --scene_prim_path.",
)
parser.add_argument("--env_spacing", type=float, default=5.0, help="Spacing between environments if num_envs > 1.")
parser.add_argument(
    "--visual_mesh_prim",
    type=str,
    default="NvbloxMesh",
    help="Visual-only mesh prim path to keep non-physical. Relative values are resolved under --scene_prim_path.",
)

# Robot arguments
parser.add_argument("--usd", type=str, default=str(DEFAULT_ROBOT_USD), help="Path to the xlerobot USD.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robot environments to spawn.")
parser.add_argument("--spawn_height", type=float, default=0.25, help="Initial robot root height at x=y=0.")
parser.add_argument("--wheel_radius", type=float, default=0.08)
parser.add_argument("--wheel_track", type=float, default=0.56)
parser.add_argument(
    "--wheel_effort_limit",
    type=float,
    default=5000.0,
    help="Wheel actuator effort limit in simulation.",
)

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

# Camera arguments
parser.add_argument("--cam_x", type=float, default=0.22, help="Camera X offset from base_link (m).")
parser.add_argument("--cam_z", type=float, default=0.50, help="Camera Z offset from base_link (m).")
parser.add_argument("--cam_pitch", type=float, default=15.0, help="Camera downward pitch (degrees).")
parser.add_argument("--cam_max_dist", type=float, default=11.0, help="Max depth range (m).")
parser.add_argument("--cam_debug_vis", action="store_true", help="Show raycaster debug rays in viewport.")

# Scene contact material
parser.add_argument("--scene_static_friction", type=float, default=1.0)
parser.add_argument("--scene_dynamic_friction", type=float, default=0.9)
parser.add_argument("--scene_restitution", type=float, default=0.0)
parser.add_argument(
    "--ground_z",
    type=float,
    default=-0.05,
    help="Deprecated and ignored. Ground now comes from the scene floor mesh.",
)
parser.add_argument(
    "--ground_size",
    type=float,
    default=80.0,
    help="Deprecated and ignored. Ground now comes from the scene floor mesh.",
)

# Debug / automation
parser.add_argument("--debug_drive", action="store_true")
parser.add_argument("--debug_every", type=int, default=10)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Exit after this many sim steps. Use 0 to run until the app window closes.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_known_args()[0]

SCENE_USD_PATH = Path(args_cli.scene_usd).expanduser().resolve()
ROBOT_USD_PATH = Path(args_cli.usd).expanduser().resolve()
SCENE_MESH_PRIM_PATH = resolve_prim_path(args_cli.scene_prim_path, args_cli.scene_mesh_prim)
VISUAL_MESH_PRIM_PATH = resolve_prim_path(args_cli.scene_prim_path, args_cli.visual_mesh_prim)

if not SCENE_USD_PATH.is_file():
    raise FileNotFoundError(f"Scene USD not found: {SCENE_USD_PATH}")
if not ROBOT_USD_PATH.is_file():
    raise FileNotFoundError(f"Robot USD not found: {ROBOT_USD_PATH}")

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2
import numpy as np
import torch
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCameraCfg, patterns
from isaaclab.utils.math import euler_xyz_from_quat


DEPTH_WINDOW_NAME = "Astera Depth Camera (env 0)"

if not args_cli.headless:
    cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEPTH_WINDOW_NAME, 64 * 8, 40 * 8)
    cv2.moveWindow(DEPTH_WINDOW_NAME, 50, 50)
    print("[INFO] OpenCV depth window created.")


def build_camera_cfg() -> RayCasterCameraCfg:
    pitch_rad = math.radians(args_cli.cam_pitch)
    half = pitch_rad / 2.0
    qw, qy = math.cos(half), math.sin(half)
    intrinsic_matrix = [
        72.7025, 0.0, 94.4457,
        0.0, 72.7025, 62.5424,
        0.0, 0.0, 1.0,
    ]
    return RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        mesh_prim_paths=[SCENE_MESH_PRIM_PATH],
        update_period=0,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(args_cli.cam_x, 0.0, args_cli.cam_z),
            rot=(qw, 0.0, qy, 0.0),
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        debug_vis=args_cli.cam_debug_vis,
        max_distance=args_cli.cam_max_dist,
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            width=192,
            height=120,
        ),
    )


SCENE_CFG = AssetBaseCfg(
    prim_path=args_cli.scene_prim_path,
    spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_USD_PATH)),
    collision_group=-1,
)

ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USD_PATH)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.spawn_height)),
    actuators={
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*_joint"],
            stiffness=0.0,
            damping=50.0,
            effort_limit_sim=args_cli.wheel_effort_limit,
            velocity_limit_sim=200.0,
        )
    },
)


class AsteraDriveSceneCfg(InteractiveSceneCfg):
    astera = SCENE_CFG
    robot = ROBOT_CFG
    depth_camera = build_camera_cfg()


class DifferentialDriveKeyboard(Se2Keyboard):
    def __str__(self):
        return "Arrow keys: fwd/back/yaw, R=reset, L=clear"

    def _create_key_bindings(self):
        self._INPUT_KEY_MAPPING = {
            "NUMPAD_8": (self.v_x_sensitivity, 0.0, 0.0),
            "UP": (self.v_x_sensitivity, 0.0, 0.0),
            "NUMPAD_2": (-self.v_x_sensitivity, 0.0, 0.0),
            "DOWN": (-self.v_x_sensitivity, 0.0, 0.0),
            "NUMPAD_4": (0.0, 0.0, self.omega_z_sensitivity),
            "LEFT": (0.0, 0.0, self.omega_z_sensitivity),
            "NUMPAD_6": (0.0, 0.0, -self.omega_z_sensitivity),
            "RIGHT": (0.0, 0.0, -self.omega_z_sensitivity),
        }


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


def depth_to_colormap(depth: np.ndarray, max_dist: float) -> np.ndarray:
    clean = np.nan_to_num(depth, nan=max_dist, posinf=max_dist, neginf=0.0)
    depth_norm = np.clip((clean - 0.3) / (8.0 - 0.3), 0.0, 1.0)
    depth_u8 = (255 * (1.0 - depth_norm)).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def disable_mesh_physics(mesh_prim_path: str):
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(mesh_prim_path)
    if not prim.IsValid():
        return

    collision_api = UsdPhysics.CollisionAPI(prim)
    if collision_api:
        collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
        if collision_enabled_attr:
            collision_enabled_attr.Set(False)

    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    if rigid_body_api:
        rigid_body_enabled_attr = rigid_body_api.GetRigidBodyEnabledAttr()
        if rigid_body_enabled_attr:
            rigid_body_enabled_attr.Set(False)

    print(f"[INFO] Disabled physics on {mesh_prim_path}.")


def reset_robot(scene: InteractiveScene, keyboard: DifferentialDriveKeyboard | None):
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    # Keep the robot at a fixed world-space spawn regardless of env layout.
    root_state[:, 0] = 0.0
    root_state[:, 1] = 0.0
    root_state[:, 2] = args_cli.spawn_height
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    scene.reset()
    if keyboard is not None:
        keyboard.reset()
    print("[INFO] Robot reset.")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1 / 120.0, render_interval=4)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([4.0, -4.0, 3.0], [0.0, 0.0, 0.5])

    scene_cfg = AsteraDriveSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)
    disable_mesh_physics(VISUAL_MESH_PRIM_PATH)
    scene.filter_collisions()
    sim.reset()

    robot = scene["robot"]
    depth_camera = scene["depth_camera"]

    wheel_joint_ids, wheel_joint_names = robot.find_joints(
        ["wheel_left_joint", "wheel_right_joint"],
        preserve_order=True,
    )

    cam_arrow_cfg = RED_ARROW_X_MARKER_CFG.copy()
    cam_arrow_cfg.prim_path = "/Visuals/CameraArrow"
    cam_arrow_cfg.markers["arrow"].scale = (0.02, 0.02, 0.15)
    cam_arrow = VisualizationMarkers(cam_arrow_cfg)

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

    keyboard = None
    if not args_cli.headless:
        keyboard = DifferentialDriveKeyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=args_cli.linear_speed,
                v_y_sensitivity=0.0,
                omega_z_sensitivity=args_cli.yaw_speed,
                sim_device=args_cli.device,
            )
        )

    zero_command = torch.zeros(3, dtype=torch.float32, device=args_cli.device)

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
        filtered_v_x = 0.0
        filtered_omega_z = 0.0
        settle_timer = max(0.0, args_cli.settle_time)
        yaw_target = None
        yaw_error_integral = 0.0
        filtered_yaw_rate = 0.0

    if keyboard is not None:
        keyboard.add_callback("R", handle_reset)

    print("\n" + "=" * 64)
    print("  xlerobot in Astera.usd — keyboard drive + PID yaw hold + DEPTH CAMERA")
    print("=" * 64)
    print(f"  Scene USD:   {SCENE_USD_PATH}")
    print(f"  Scene prim:  {args_cli.scene_prim_path}")
    print(f"  Raycast on:  {SCENE_MESH_PRIM_PATH}")
    print(f"  Visual only: {VISUAL_MESH_PRIM_PATH}")
    print("  Ground:      disabled, using the floor mesh from the scene USD")
    print(f"  Robot USD:   {ROBOT_USD_PATH}")
    print(f"  Wheel joints: {wheel_joint_names}")
    print(f"  Spawn pose:  world x=0.0, y=0.0, z={args_cli.spawn_height}")
    print(f"  Camera:      offset=({args_cli.cam_x}, 0, {args_cli.cam_z}), pitch={args_cli.cam_pitch}deg")
    if keyboard is None:
        print("  Controls:    headless mode, keyboard disabled")
    else:
        print("  Controls:    Arrow keys (fwd/back/yaw), R=reset, L=clear")
    print("=" * 64 + "\n")

    handle_reset()
    sim_dt = sim.get_physics_dt()
    frame_count = 0
    depth_first = True

    while simulation_app.is_running():
        with torch.inference_mode():
            command = keyboard.advance() if keyboard is not None else zero_command
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
                    filtered_v_x,
                    raw_v_x,
                    sim_dt,
                    args_cli.command_ramp_up_time,
                    args_cli.command_ramp_down_time,
                )
                filtered_omega_z = ramp_command(
                    filtered_omega_z,
                    raw_omega_z,
                    sim_dt,
                    args_cli.command_ramp_up_time,
                    args_cli.command_ramp_down_time,
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
                commanded_omega_z = pid_correction if (moving_command_active or manual_yaw_active) else 0.0

            left_w = (v_x - 0.5 * args_cli.wheel_track * commanded_omega_z) / args_cli.wheel_radius
            right_w = (v_x + 0.5 * args_cli.wheel_track * commanded_omega_z) / args_cli.wheel_radius
            wheel_targets = torch.tensor(
                [[left_w, right_w]],
                dtype=torch.float32,
                device=args_cli.device,
            ).repeat(scene.num_envs, 1)
            robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            robot_pos = robot.data.root_pos_w[0]
            print(
                f"[STEP {frame_count:06d}] robot_pos_w="
                f"({robot_pos[0].item():.3f}, {robot_pos[1].item():.3f}, {robot_pos[2].item():.3f})",
                flush=True,
            )

            cam_pos = depth_camera.data.pos_w[0:1]
            cam_quat = depth_camera.data.quat_w_world[0:1]
            cam_arrow.visualize(cam_pos, cam_quat)
            cam_sphere.visualize(cam_pos)

            frame_count += 1
            if not args_cli.headless and frame_count % 4 == 0:
                depth_data = depth_camera.data.output["distance_to_image_plane"]
                if depth_data is not None and depth_data.numel() > 0:
                    depth_img = depth_data[0].cpu().numpy()
                    if depth_img.ndim == 3:
                        depth_img = depth_img.squeeze(-1)
                    if depth_first:
                        print(
                            f"[INFO] Depth: shape={depth_img.shape}, "
                            f"range=[{depth_img.min():.1f}, {depth_img.max():.1f}]m"
                        )
                        depth_first = False
                    color = depth_to_colormap(depth_img, args_cli.cam_max_dist)
                    height, width = color.shape[:2]
                    display = cv2.resize(color, (width * 8, height * 8), interpolation=cv2.INTER_NEAREST)
                    valid = depth_img[~np.isnan(depth_img) & (depth_img < args_cli.cam_max_dist)]
                    if valid.size > 0:
                        txt = f"min={valid.min():.1f}m max={valid.max():.1f}m mean={valid.mean():.1f}m"
                    else:
                        txt = "no returns"
                    cv2.putText(display, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow(DEPTH_WINDOW_NAME, display)
                    cv2.waitKey(1)

            if args_cli.max_steps > 0 and frame_count >= args_cli.max_steps:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
