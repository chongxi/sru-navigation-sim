# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import gymnasium as gym

from . import agents, navigation_env_cfg

##
# Register Gym environments.
##

##############################################################################################################
# MDPO

gym.register(
    id="Isaac-Nav-MDPO-DiffDrive-v0",
    entry_point="isaaclab_nav_task.navigation.diff_drive_env:DiffDriveNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DiffDriveNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DiffDriveNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-DiffDrive-Play-v0",
    entry_point="isaaclab_nav_task.navigation.diff_drive_env:DiffDriveNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DiffDriveNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DiffDriveNavMDPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-DiffDrive-Dev-v0",
    entry_point="isaaclab_nav_task.navigation.diff_drive_env:DiffDriveNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DiffDriveNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DiffDriveNavMDPORunnerDevCfg,
    },
)

######################################################################################
# PPO

gym.register(
    id="Isaac-Nav-PPO-DiffDrive-v0",
    entry_point="isaaclab_nav_task.navigation.diff_drive_env:DiffDriveNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DiffDriveNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DiffDriveNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-DiffDrive-Play-v0",
    entry_point="isaaclab_nav_task.navigation.diff_drive_env:DiffDriveNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DiffDriveNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DiffDriveNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-DiffDrive-Dev-v0",
    entry_point="isaaclab_nav_task.navigation.diff_drive_env:DiffDriveNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DiffDriveNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DiffDriveNavPPORunnerDevCfg,
    },
)
