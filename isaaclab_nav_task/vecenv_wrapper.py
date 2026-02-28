# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""VecEnv compatibility wrapper for custom rsl_rl v2.3.4.

Bridges the IsaacLab v2.3.2 gymnasium environment interface to the
tuple-based VecEnv interface expected by the custom rsl_rl OnPolicyRunner.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class SruRslRlVecEnvWrapper:
    """Wraps an IsaacLab gymnasium environment for the custom rsl_rl v2.3.4 library.

    The custom rsl_rl OnPolicyRunner expects:
    - ``get_observations()`` -> ``(obs_tensor, extras_dict)``
    - ``step(actions)`` -> ``(obs_tensor, rewards, dones, infos)``
    - ``infos["observations"]["critic"]`` for asymmetric critic observations
    - ``infos["time_outs"]`` for bootstrapping on truncation
    - ``infos["episode"]`` or ``infos["log"]`` for logging

    This wrapper must be the last wrapper in the chain.
    """

    def __init__(self, env: gym.Env, clip_actions: float | None = None):
        if not isinstance(env.unwrapped, (ManagerBasedRLEnv, DirectRLEnv)):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv."
                f" Environment type: {type(env)}"
            )

        self.env = env
        self.clip_actions = clip_actions

        # Properties required by OnPolicyRunner
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        if self.clip_actions is not None:
            self._modify_action_space()

        # Reset at the start since the custom rsl_rl runner does not call reset
        self.env.reset()

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    def _obs_dict_to_tuple(self, obs_dict: dict) -> tuple[torch.Tensor, dict]:
        """Convert IsaacLab obs_dict to (obs_tensor, extras_dict) tuple."""
        obs = obs_dict["policy"]
        extras = {"observations": {}}
        if "critic" in obs_dict:
            extras["observations"]["critic"] = obs_dict["critic"]
        return obs, extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Return current observations as (obs_tensor, extras_dict)."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return self._obs_dict_to_tuple(obs_dict)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment and return (obs, rewards, dones, infos)."""
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # Extract policy obs and build infos
        obs = obs_dict["policy"]
        infos = dict(extras)
        infos["observations"] = {}
        if "critic" in obs_dict:
            infos["observations"]["critic"] = obs_dict["critic"]

        # Compute dones
        dones = (terminated | truncated).to(dtype=torch.long)

        # Add time_outs for bootstrapping (infinite horizon tasks only)
        if not self.unwrapped.cfg.is_finite_horizon:
            infos["time_outs"] = truncated

        return obs, rew, dones, infos

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.reset()
        return self._obs_dict_to_tuple(obs_dict)

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def close(self):
        return self.env.close()

    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)
