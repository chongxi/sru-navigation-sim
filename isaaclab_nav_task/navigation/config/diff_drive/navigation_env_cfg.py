# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Differential-drive specific configuration for navigation environment."""

import math

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
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

        # Avoid trivial episodes where the robot spawns already too close to the goal.
        self.commands.robot_goal.min_spawn_goal_distance = 2.0
        self.commands.robot_goal.spawn_goal_resample_attempts = 32

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
            policy_scaling=[2.5, 0.0, 2.0],  # [vx m/s, vy(ignored), yaw-target slew rate rad/s]
            use_raw_actions=True,
            policy_distr_type="gaussian",
        )

        # --- Physics & Control frequency ---
        # 120Hz physics (matches drive_terrain_pid.py)
        self.sim.dt = 1.0 / 60.0
        # 10Hz navigation policy (PID handles yaw at 120Hz between decisions)
        self.decimation = 6  # 120Hz / 12 = 10Hz
        self.episode_length_s = 60.0  # longer horizon for navigation (60s = 1min at sim_dt=1/60s)

        # Enable contact processing for diff-drive (base class disables it for
        # legged robots as an optimization, but wheels need proper friction solving)
        self.sim.disable_contact_processing = False

        # Use average friction combine so robot-side friction randomization does
        # not get multiplied down into overly slippery wheel-ground contact.
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
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
        # Pit falls are catastrophic for this platform, so penalize them more
        # heavily than generic episode terminations.
        self.rewards.terrain_fall_penalty = RewTerm(
            func=mdp.is_terminated_term,
            weight=-50.0,
            params={"term_keys": ["terrain_fall"]},
        )
        # Keep a dedicated collision penalty term available for stage-wise
        # tuning without changing the generic termination penalty.
        self.rewards.base_contact_penalty = RewTerm(
            func=mdp.is_terminated_term,
            weight=0.0,
            params={"term_keys": ["base_contact"]},
        )
        # Dense reward: distance decrease toward goal (potential-based shaping)
        self.rewards.goal_progress = RewTerm(
            func=mdp.goal_progress,
            weight=10.0,
            params={"command_name": "robot_goal"},
        )
        self.rewards.pose_goal_proximity = RewTerm(
            func=mdp.pose_goal_proximity,
            weight=4.0,
            params={
                "command_name": "robot_goal",
                "xy_scale": 0.25,
                "yaw_scale": math.radians(25.0),
                "activation_xy_threshold": 0.5,
            },
        )
        self.rewards.pose_goal_hold_bonus = RewTerm(
            func=mdp.pose_goal_hold_bonus,
            weight=10.0,
            params={
                "command_name": "robot_goal",
                "xy_threshold": 0.35,
                "yaw_threshold": math.radians(15.0),
                "lin_speed_threshold": 0.15,
                "yaw_rate_threshold": 0.30,
            },
        )
        # Align success and metrics with the pose objective. The policy should
        # only count as successful once it is both near the goal and settled
        # with the correct heading.
        pose_goal_success_params = {
            "distance_threshold": 0.35,
            "yaw_threshold": math.radians(15.0),
            "lin_speed_threshold": 0.15,
            "yaw_rate_threshold": 0.30,
        }
        self.terminations.time_out.params = dict(pose_goal_success_params)
        self.terminations.early_termination.params = dict(pose_goal_success_params)
        self.observations.metrics.in_goal.params = dict(pose_goal_success_params)

        # --- Terminations ---
        # Detect navigation collisions from horizontal normal forces on the
        # chassis only. Ground support loads are mostly vertical, and wheel
        # contacts are excluded from this check.
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base_link"]
            ),
            "threshold": 50.0,
            "horizontal_only": True,
        }

        # --- Events ---
        # Disable push_robot for diff-drive: a 2-wheel robot is much less
        # stable than a 4-legged robot, velocity impulses cause tipping/spinning.
        self.events.push_robot = None

        # Bypass material / friction randomization for diff-drive for now. It
        # perturbs wheel/base friction and restitution enough to dominate
        # controller debugging on this 2-wheel platform.
        # To restore randomization, delete or comment out the next line
        # (`self.events.physics_material = None`) and uncomment the parameter
        # overrides below.
        # self.events.physics_material.params["static_friction_range"] = (0.9, 1.1)
        # self.events.physics_material.params["dynamic_friction_range"] = (0.85, 1.0)
        # self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.physics_material = None

        # The diff-drive controller uses ramp-limited commands instead of the
        # base task's low-pass-filter randomization.
        self.events.randomize_low_pass_filter_alpha = None

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
                goal_padding_cells=8,
                spawn_padding_cells=8,
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
                goal_padding_cells=8,
                spawn_padding_cells=8,
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
                goal_padding_cells=8,
                spawn_padding_cells=8,
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
