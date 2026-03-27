# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Diff-drive navigation environment."""

from __future__ import annotations

# Re-export the base NavigationEnv as DiffDriveNavigationEnv
from isaaclab_nav_task.navigation.navigation_env import NavigationEnv as DiffDriveNavigationEnv

__all__ = ["DiffDriveNavigationEnv"]
