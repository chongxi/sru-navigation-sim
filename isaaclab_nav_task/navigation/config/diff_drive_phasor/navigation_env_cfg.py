# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Phasor-specific diff-drive navigation environment configuration."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

import isaaclab_nav_task.navigation.mdp as mdp
from isaaclab_nav_task.navigation.config.diff_drive.navigation_env_cfg import DiffDriveNavigationEnvCfg
from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand


def forward_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward velocity in the robot body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0:1]


def yaw_rate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw rate in the robot body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 2:3]


def robot_heading_trig(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot heading encoded as [cos(yaw), sin(yaw)] in the world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)


def goal_pos_error_world(
    env: ManagerBasedRLEnv,
    command_name: str = "robot_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Unnormalized goal position error in the world frame."""
    goal_cmd: RobotNavigationGoalCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    return goal_cmd.goal_position_world[:, :2] - asset.data.root_pos_w[:, :2]


def goal_heading_trig_world(
    env: ManagerBasedRLEnv,
    command_name: str = "robot_goal",
) -> torch.Tensor:
    """Goal heading encoded as [cos(goal_yaw), sin(goal_yaw)] in the world frame."""
    goal_cmd: RobotNavigationGoalCommand = env.command_manager.get_term(command_name)
    goal_yaw = goal_cmd.goal_heading_world
    return torch.stack((torch.cos(goal_yaw), torch.sin(goal_yaw)), dim=-1)


@configclass
class PhasorPolicyCfg(ObsGroup):
    """Observation group for phasor policies."""

    forward_vel = ObsTerm(func=forward_vel)
    yaw_rate = ObsTerm(func=yaw_rate)
    robot_heading = ObsTerm(func=robot_heading_trig)
    goal_pos_error = ObsTerm(func=goal_pos_error_world)
    goal_heading = ObsTerm(func=goal_heading_trig_world, params={"command_name": "robot_goal"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    last_action = ObsTerm(func=mdp.last_action)
    depth_image = ObsTerm(
        func=mdp.depth_image_noisy_delayed,
        params={"sensor_cfg": SceneEntityCfg("raycast_camera")},
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class PhasorCriticCfg(ObsGroup):
    """Critic observations for phasor policies."""

    forward_vel = ObsTerm(func=forward_vel)
    yaw_rate = ObsTerm(func=yaw_rate)
    robot_heading = ObsTerm(func=robot_heading_trig)
    goal_pos_error = ObsTerm(func=goal_pos_error_world)
    goal_heading = ObsTerm(func=goal_heading_trig_world, params={"command_name": "robot_goal"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    last_action = ObsTerm(func=mdp.last_action)
    time_normalized = ObsTerm(func=mdp.time_normalized, params={"command_name": "robot_goal"})
    height_scan_critic = ObsTerm(
        func=mdp.height_scan_feat,
        params={"sensor_cfg": SceneEntityCfg("height_scanner_critic")},
    )
    depth_image = ObsTerm(
        func=mdp.depth_image_prefect,
        params={"sensor_cfg": SceneEntityCfg("raycast_camera")},
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class DiffDriveNavigationPhasorEnvCfg(DiffDriveNavigationEnvCfg):
    """Diff-drive environment with phasor observations and baseline depth encoding."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = PhasorPolicyCfg()
        self.observations.critic = PhasorCriticCfg()


@configclass
class DiffDriveNavigationPhasorEnvCfg_DEV(DiffDriveNavigationPhasorEnvCfg):
    """Development variant with the same terrain settings as diff-drive DEV."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class DiffDriveNavigationPhasorEnvCfg_PLAY(DiffDriveNavigationPhasorEnvCfg):
    """Play variant with the same scene settings as diff-drive PLAY."""

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
