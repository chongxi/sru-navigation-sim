#!/usr/bin/env python3
"""Quick test: construct ActorCriticSRU + load checkpoint, print architecture & param count.

No Isaac Sim needed — pure PyTorch.

Usage:
    python scripts/test_policy_load.py \
        --checkpoint logs/rsl_rl/diff_drive_navigation_mdpo/2026-03-30_09-39-36/model_850.pt
"""

import argparse
import sys
import torch

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

device = torch.device(args.device)

# ── Depth encoder (skip - requires Isaac Sim) ──
print("=" * 60)
print("1) Depth encoder: SKIPPED (needs Isaac Sim / carb)")
print("=" * 60)
print()

# ── Policy ──
print("=" * 60)
print("2) Constructing ActorCriticSRU...")
print("=" * 60, flush=True)

from rsl_rl.modules.actor_critic_sru import ActorCriticSRU

NUM_ACTOR_OBS = 17 + 64 * 5 * 8  # 2577
NUM_ACTIONS = 2
# Critic obs = proprio(17) + height(64*7*7=3136) + image(2560) + time(1) = 5714
NUM_CRITIC_OBS = 17 + 64 * 7 * 7 + 64 * 5 * 8 + 1

actor_critic = ActorCriticSRU(
    num_actor_obs=NUM_ACTOR_OBS,
    num_critic_obs=NUM_CRITIC_OBS,
    num_actions=NUM_ACTIONS,
    actor_hidden_dims=[256, 256],
    critic_hidden_dims=[256, 256],
    activation="lrelu",
    init_noise_std=[1.0, 1.0],
    rnn_hidden_size=512,
    rnn_type="lstm_sru",
    rnn_num_layers=1,
    dropout=0.2,
    num_cameras=1,
    image_input_dims=(64, 5, 8),
    height_input_dims=(64, 7, 7),
)
print("[OK] Model constructed on CPU.", flush=True)

# Print architecture
print()
print("=" * 60)
print("3) Architecture")
print("=" * 60)
print(actor_critic)
print()

# Parameter counts
total = sum(p.numel() for p in actor_critic.parameters())
actor_params = (
    sum(p.numel() for p in actor_critic.actor.parameters())
    + sum(p.numel() for p in actor_critic.memory_a.parameters())
    + sum(p.numel() for p in actor_critic.attn_image_net.parameters())
    + sum(p.numel() for p in actor_critic.linear_dropout_actor.parameters())
    + actor_critic.log_std.numel()
)
print(f"   Total params:  {total:,}")
print(f"   Actor params:  {actor_params:,}")
print(f"   Critic params: {total - actor_params:,}")
print()

# ── Load checkpoint ──
print("=" * 60)
print(f"4) Loading checkpoint: {args.checkpoint}")
print("=" * 60, flush=True)

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)
actor_critic.eval()
print(f"   Loaded from iteration {ckpt['iter']}")
print(f"   Checkpoint keys: {list(ckpt.keys())}")
print()

# ── Move to device ──
print(f"5) Moving to {device}...", flush=True)
actor_critic = actor_critic.to(device)
print(f"   [OK] Model on {device}", flush=True)

# ── Quick inference test ──
print()
print("=" * 60)
print("6) Inference test")
print("=" * 60, flush=True)

with torch.inference_mode():
    dummy_obs = torch.randn(1, NUM_ACTOR_OBS, device=device)
    actions = actor_critic.act_inference(dummy_obs)
    print(f"   Input:  obs shape = {tuple(dummy_obs.shape)}")
    print(f"   Output: actions = {actions.detach().cpu().numpy().flatten()}")
    print(f"   Action shape: {tuple(actions.shape)}")

print()
print("[DONE] All checks passed.")
