# Copyright (c) 2025, Chongxi Lai.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""RSL-RL agent configurations for phasor diff-drive navigation tasks."""

from isaaclab.utils import configclass

from isaaclab_nav_task.navigation.config.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DiffDriveNavMDPOPhasorRunnerCfg(RslRlOnPolicyRunnerCfg):
    """MDPO runner configuration for phasor diff-drive navigation."""

    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 50
    logger = "wandb"
    seed = 42
    wandb_project = "isaaclab_nav_diff_drive"
    experiment_name = "diff_drive_navigation_phasor_mdpo"
    empirical_normalization = True
    torch_compile_policy = True
    torch_compile_mode = "default"
    reward_shifting_value = 0.05
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticPhasor",
        init_noise_std=[0.8, 0.8],  # [vx, yaw_target_rate]
        actor_hidden_dims=[512, 256],
        critic_hidden_dims=[512, 256],
        activation="gelu",
        rnn_hidden_size=512,
        rnn_type="lstm",
        rnn_num_layers=2,
        dropout=0.1,
        num_cameras=1,
        image_input_dims=(64, 5, 8),
        height_input_dims=(64, 7, 7),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="MDPO",
        value_loss_coef=0.0002,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        entropy_coef=1.2e-3,
        num_learning_epochs=4,
        num_mini_batches=6,
        learning_rate=1.0e-3,
        min_learning_rate=3.0e-5,
        schedule="linear",
        gamma=0.999,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DiffDriveNavMDPOPhasorRunnerDevCfg(DiffDriveNavMDPOPhasorRunnerCfg):
    """Development MDPO configuration for phasor diff-drive navigation."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "diff_drive_navigation_phasor_mdpo_dev"
        self.logger = "wandb"
