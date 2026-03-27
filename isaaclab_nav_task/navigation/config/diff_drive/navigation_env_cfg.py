# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Differential-drive specific configuration for navigation environment."""

import math

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from isaaclab_nav_task.navigation.navigation_env_cfg import NavigationEnvCfg
import isaaclab_nav_task.navigation.mdp as mdp
from isaaclab_nav_task.terrains.hf_terrains_maze_cfg import HfMazeTerrainCfg

from isaaclab_nav_task.navigation.assets import DIFF_DRIVE_CFG  # isort: skip


@configclass
class DiffDriveNavigationEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator

        initialize_depth_noise_generator(robot_name="diff_drive", use_jit_precompiled=False)

        # --- Robot ---
        self.scene.robot = DIFF_DRIVE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # --- Sensors ---
        # Raycast camera on base_link (front-center of chassis)
        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.raycast_camera.offset.pos = (0.22, 0.0, 0.50)
        # 15° downward pitch (w,x,y,z) = (cos(7.5°), 0, sin(7.5°), 0)
        self.scene.raycast_camera.offset.rot = (0.9914449, 0.0, 0.1305262, 0.0)
        # Height scanner for critic
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # --- Actions ---
        # Replace hierarchical (low-level policy) with direct diff-drive + PID yaw hold
        self.actions.velocity_command = mdp.DiffDriveNavigationSE2ActionCfg(
            asset_name="robot",
            wheel_velocity_action=mdp.JointVelocityActionCfg(
                asset_name="robot",
                joint_names=["wheel_left_joint", "wheel_right_joint"],
                scale=1.0,
                use_default_offset=False,
            ),
            policy_scaling=[2.5, 1.0, 3.14159],  # [vx m/s, vy(ignored), yaw_target rad]
            use_raw_actions=True,
            policy_distr_type="gaussian",
        )

        # --- Physics & Control frequency ---
        # 120Hz physics (matches drive_terrain_pid.py)
        self.sim.dt = 1.0 / 120.0
        # 10Hz navigation policy (PID handles yaw at 120Hz between decisions)
        self.decimation = 12  # 120Hz / 12 = 10Hz

        # Enable contact processing for diff-drive (base class disables it for
        # legged robots as an optimization, but wheels need proper friction solving)
        self.sim.disable_contact_processing = False

        # Match test_diff_drive_simple.py / drive_terrain_pid.py friction
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            restitution=0.0,
            static_friction=1.0,
            dynamic_friction=0.9,
            compliant_contact_stiffness=5e5,
            compliant_contact_damping=300.0,
        )
        self.sim.physics_material = self.scene.terrain.physics_material

        # --- Observations ---
        # Remove low-level policy obs group (no low-level policy for diff-drive)
        self.observations.low_level_policy = None

        # --- Rewards ---
        # Use wheel joints for acceleration penalty
        self.rewards.joint_acc_l2_joint.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["wheel_left_joint", "wheel_right_joint"]
            )
        }
        # Reduce action_rate penalty to encourage exploration
        self.rewards.action_rate_l1.weight = -0.01
        # Dense reward: distance decrease toward goal (potential-based shaping)
        from isaaclab.managers import RewardTermCfg as RewTerm
        self.rewards.goal_progress = RewTerm(
            func=mdp.goal_progress,
            weight=10.0,
            params={"command_name": "robot_goal"},
        )

        # --- Terminations ---
        # Re-enable base_contact for obstacle avoidance learning.
        # Chassis idles at ~294N ground contact; wall collisions at speed >> 800N.
        # Threshold 800N catches real collisions, ignores ground contact.
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base_link"]
            ),
            "threshold": 800.0,
        }

        # --- Events ---
        # Disable push_robot for diff-drive: a 2-wheel robot is much less
        # stable than a 4-legged robot, velocity impulses cause tipping/spinning.
        self.events.push_robot = None

        # Narrow robot body friction randomization (base uses 0.7-1.0 dynamic
        # which causes traction loss for 2-wheel robots)
        self.events.physics_material.params["static_friction_range"] = (0.9, 1.1)
        self.events.physics_material.params["dynamic_friction_range"] = (0.85, 1.0)

        # Low-pass filter alpha ranges for diff-drive
        self.events.randomize_low_pass_filter_alpha.params = {
            "alpha_range": (0.4, 0.8),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.4, 0.8),
            "alpha_range_vy": (0.4, 0.8),
            "alpha_range_omega": (0.4, 0.8),
        }

        # --- Terrain ---
        # Diff-drive: wider corridors (cell_size=3.0 vs B2W's 2.0) because
        # xlerobot is wider (~0.7m) than B2W (~0.4m). More open walls and
        # fewer random obstacles for easier initial learning.
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.2, 0.7]
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.sub_terrains = {
            "maze": HfMazeTerrainCfg(
                proportion=0.4,
                open_probability=0.95,
                grid_size=(10, 10),
                cell_size=3.0,
                add_noise_to_flat=False,
                add_goal=True,
                randomize_wall=False,  # no random obstacles, DFS maze only
                random_wall_ratio=0.0,
                add_stairs_to_maze=False,
            ),
            "non_maze": HfMazeTerrainCfg(
                proportion=0.3,
                open_probability=0.95,
                grid_size=(10, 10),
                cell_size=3.0,
                add_noise_to_flat=False,
                add_goal=True,
                randomize_wall=True,
                random_wall_ratio=0.2,  # sparse obstacles, less chance of blocking
                non_maze_terrain=True,
            ),
            "pits": HfMazeTerrainCfg(
                proportion=0.3,
                open_probability=0.95,
                grid_size=(10, 10),
                cell_size=3.0,
                add_noise_to_flat=False,
                add_goal=True,
                randomize_wall=False,  # no random walls in pit terrain
                random_wall_ratio=0.0,
                non_maze_terrain=True,
                dynamic_obstacles=True,
            ),
        }


@configclass
class DiffDriveNavigationEnvCfg_DEV(DiffDriveNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class DiffDriveNavigationEnvCfg_PLAY(DiffDriveNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
