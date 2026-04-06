# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Phasor-specific diff-drive navigation task registrations."""

import gymnasium as gym

from . import agents, navigation_env_cfg


gym.register(
    id="Isaac-Nav-MDPO-DiffDrive-Phasor-v0",
    entry_point="isaaclab_nav_task.navigation.diff_drive_env:DiffDriveNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DiffDriveNavigationPhasorEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DiffDriveNavMDPOPhasorRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-DiffDrive-Phasor-Play-v0",
    entry_point="isaaclab_nav_task.navigation.diff_drive_env:DiffDriveNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DiffDriveNavigationPhasorEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DiffDriveNavMDPOPhasorRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-DiffDrive-Phasor-Dev-v0",
    entry_point="isaaclab_nav_task.navigation.diff_drive_env:DiffDriveNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.DiffDriveNavigationPhasorEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DiffDriveNavMDPOPhasorRunnerDevCfg,
    },
)
