# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for the xlerobot differential-drive robot.

The following configuration parameters are available:

* :obj:`DIFF_DRIVE_CFG`: The xlerobot 2-wheel differential-drive base (v13).

Reference:
    xlerobot-diff-drive: 2-wheel diff-drive base with placeholder arm.
    Wheel radius 0.08m, track width 0.56m, chassis mass 30kg.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_ASSETS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

__all__ = ["DIFF_DRIVE_CFG"]


DIFF_DRIVE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSETS_DATA_DIR, "Robots", "xlerobot", "xlerobot_wheel_v14.usd"),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, -0.04),
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            effort_limit_sim=5000.0,
            velocity_limit_sim=200.0,
            stiffness={".*": 0.0},   # velocity control mode
            damping={".*": 50.0},
        ),
    },
)
"""Configuration of xlerobot diff-drive robot (v13) using ImplicitActuatorCfg."""
