# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Curriculum functions for navigation tasks.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def disable_backward_penalty_after_steps(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    term_name: str = "backward_movement_penalty", 
    num_steps: int = 1000
) -> torch.Tensor:
    """Curriculum that disables the backward movement penalty after a certain number of steps.
    
    This helps with early training by preventing backward movement, but removes the constraint
    later to allow more natural movement patterns.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the backward movement penalty term.
        num_steps: The number of steps after which the penalty should be disabled.
        
    Returns:
        Current step counter as float for logging purposes.
    """
    if env.common_step_counter > num_steps:
        # Check if the term exists and has a non-zero weight
        if hasattr(env.reward_manager, 'get_term_cfg'):
            try:
                term_cfg = env.reward_manager.get_term_cfg(term_name)
                if term_cfg.weight != 0.0:
                    # Disable the penalty by setting weight to 0
                    term_cfg.weight = 0.0
                    env.reward_manager.set_term_cfg(term_name, term_cfg)
                    print(f"Disabled backward movement penalty at step {env.common_step_counter}")
            except KeyError:
                # Term doesn't exist, which is fine
                pass
    
    return torch.tensor(float(env.common_step_counter))


def linearly_interpolate_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
) -> torch.Tensor:
    """Linearly interpolate a reward weight over training steps.

    The curriculum state returned is the currently applied weight, which Isaac Lab
    logs through the curriculum manager when environments reset.

    Args:
        env: The learning environment.
        env_ids: Not used directly; the weight change applies globally.
        term_name: Reward term name in the reward manager.
        start_weight: Weight before the schedule starts.
        end_weight: Weight after the schedule finishes.
        start_step: Global environment step where interpolation begins.
        end_step: Global environment step where interpolation ends.

    Returns:
        Current applied reward weight for logging.
    """
    del env_ids

    if end_step <= start_step:
        current_weight = float(end_weight)
    else:
        step = int(env.common_step_counter)
        if step <= start_step:
            alpha = 0.0
        elif step >= end_step:
            alpha = 1.0
        else:
            alpha = (step - start_step) / float(end_step - start_step)
        current_weight = float(start_weight + alpha * (end_weight - start_weight))

    if hasattr(env.reward_manager, "get_term_cfg"):
        try:
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            if term_cfg.weight != current_weight:
                term_cfg.weight = current_weight
                env.reward_manager.set_term_cfg(term_name, term_cfg)
        except ValueError:
            pass

    return torch.tensor(current_weight, device=env.device)
