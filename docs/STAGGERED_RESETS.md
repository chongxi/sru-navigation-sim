# Staggered Resets

## Summary

We added a full staggered-reset training path for massively parallel on-policy RL.

The goal is to avoid collecting rollout batches where most environments are at the same part of the episode horizon. When all environments are synchronized, each `N x K` batch is temporally narrow. That makes value estimation and policy updates noisier and more cyclic, especially when:

- the task horizon `H` is long,
- the rollout length `K` is short,
- and the number of environments `N` is large.

For the current DiffDrive setup:

- `H ~= 600` env steps
- `K = 24` env steps
- `H / K = 25`

This is exactly the regime where staggered resets help.

## What Was Changed

### 1. Runner-side startup staggering

We changed the shared `rsl_rl` runner:

- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/runners/on_policy_runner.py`

The runner now:

- partitions environments into `N_B` stagger buckets,
- assigns each bucket a target phase over the full horizon,
- pre-rolls the environments in rollout-length chunks,
- applies reset gates during the pre-roll,
- starts training from a mixed distribution of task phases rather than all envs at `t = 0`.

This replaced the old approximation where we only randomized `episode_length_buf`.

### 2. Env-side gated reset handling

We changed Isaac Lab's manager-based RL env:

- `/home/chongxi/Work/Astera/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`

The env now supports staggered reset gating:

- newly terminated envs are flagged as pending instead of being reset immediately,
- pending envs are frozen until the next scheduled reset gate,
- scheduled gate envs and pending envs are reset together in a batched reset call,
- pending envs do not continue accumulating episode age while waiting.

This is the key change that turns the feature from "startup-only staggering" into "staggered resets during training."

### 3. Masking invalid waiting steps out of learning

We changed rollout storage and on-policy losses:

- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/storage/rollout_storage.py`
- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/algorithms/ppo.py`
- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/algorithms/mdpo.py`
- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/algorithms/spo.py`

When an env has terminated early and is waiting for the next gate:

- its transition is marked invalid,
- reward accumulation for that waiting step is suppressed,
- PPO/MDPO/SPO losses ignore those steps with masked means and masked advantage normalization.

Without this mask, the learner would optimize on frozen "dead time" after early termination, which would corrupt the batch.

### 4. CLI support

We added the same control flag to both training entrypoints:

- [scripts/train.py](/home/chongxi/Work/Astera/navigation/sru-navigation-sim/scripts/train.py)
- [scripts/train_second_stage.py](/home/chongxi/Work/Astera/navigation/sru-navigation-sim/scripts/train_second_stage.py)

Flag:

```bash
--staggered_reset_buckets <N_B>
```

Behavior:

- `0` or `1`: disabled
- `>1`: enable staggered resets with `N_B` buckets

## Why This Implements Staggered Resets

The implemented behavior matches the algorithmic idea:

### Initial staggering

Before learning starts:

- each environment is placed into a bucket,
- each bucket is assigned a discrete offset over the horizon,
- environments are advanced so different buckets begin at different effective times.

This creates rollout batches with mixed temporal phases from the start.

### Synchronous rollouts after staggering

Training still uses standard synchronous on-policy collection:

- all envs collect `K` new steps per rollout,
- the batch size stays `N x K`.

The difference is that the envs are no longer synchronized to the same part of the episode.

### Reset gates during training

During training:

- end-of-horizon resets happen at scheduled bucket gates,
- early-terminated envs are flagged and wait,
- at the next scheduled gate, they are reset together with the gate bucket.

That preserves temporal diversity across rollouts instead of losing it after the first few resets.

## Why Staggered Resets Help

### Problem with synchronized resets

If all envs start together:

- rollout 1 mostly contains early-episode data,
- rollout 2 mostly contains slightly later data,
- rollout 3 later still,
- and so on.

So the learner sees a batch distribution that cycles through time segments of the task horizon.

That creates:

- cyclic nonstationarity,
- poorer value targets,
- less representative batches,
- and slower or less stable learning.

### Benefit of staggered resets

With staggered resets:

- each batch contains a mixture of early, middle, and late episode phases,
- value learning sees a more representative state distribution,
- policy updates are less tied to one narrow horizon slice,
- training becomes more stationary from the optimizer's point of view.

In short, staggered resets improve within-batch temporal diversity.

## What It Achieves in Practice

For this navigation task, the observed effect was:

- faster early learning,
- quicker rise in success rate,
- smoother improvement compared with synchronized or partially staggered starts.

This is expected because the task has:

- many parallel environments,
- short rollouts,
- long horizons,
- and frequent early terminations.

Those are exactly the conditions where synchronized collection is weakest.

## Choosing `staggered_reset_buckets`

The paper-style heuristic is:

- `N_B ~= H / K`

For DiffDrive:

- `H / K ~= 25`

But using fewer buckets is often a good engineering compromise.

Current practical recommendation:

- start with `--staggered_reset_buckets 8`
- try larger values only if startup cost is acceptable and you want more temporal coverage

Using fewer than `H / K` buckets still helps. It just coarsens the phase discretization.

## Current Usage

Example:

```bash
/home/chongxi/.venv/isaaclab_51/bin/python scripts/train_second_stage.py \
  --task Isaac-Nav-MDPO-DiffDrive-v0 \
  --checkpoint /home/chongxi/Work/Astera/navigation/sru-navigation-sim/logs/rsl_rl/diff_drive_navigation_mdpo/2026-03-28_12-49-18/model_300.pt \
  --num_envs 4096 \
  --max_iterations 3000 \
  --run_name stage2_rebalanced \
  --load_optimizer \
  --learning_rate 1e-3 \
  --disable_goal_progress \
  --difficulty 0.5 1.0 \
  --staggered_reset_buckets 8 \
  --base_contact_penalty_weight -200 \
  --terrain_fall_penalty_weight -200 \
  --headless
```

## Important Note

This implementation changed shared training infrastructure, not only the DiffDrive task:

- PPO uses it
- MDPO uses it
- SPO uses it

because the changes were made in the shared runner, shared storage, and shared on-policy algorithms.

## Appendix: Codebase Boundaries

This implementation crosses three repositories / layers:

- Isaac Lab:
  runtime environment stepping and reset behavior
- `rsl_rl`:
  rollout collection, storage, and optimization
- `sru-navigation-sim`:
  CLI entrypoints and project documentation

That split matters because staggered resets are not just a task-level feature. They require coordinated changes at the simulator boundary and at the learner boundary.

### What changed in Isaac Lab

File:

- `/home/chongxi/Work/Astera/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`

Exactly what changed:

- added gated-reset state:
  - `_staggered_reset_enabled`
  - `_staggered_reset_pending`
- changed `step(...)` so early-terminated envs are not immediately reset when staggered gating is enabled
- froze pending envs while they wait for the next gate:
  - zero action
  - zero reward
  - no episode-length increment
- exposed per-step flags through `extras`:
  - `staggered_invalid`
  - `staggered_pending`
- added control methods:
  - `configure_staggered_reset_gating(...)`
  - `apply_staggered_reset_gate(...)`
- made reset clear pending state in `_reset_idx(...)`

Relevant code excerpts from Isaac Lab:

1. Added env-side gating state in `__init__`:

```python
# Optional gate-based reset handling used for staggered reset training.
self._staggered_reset_enabled = False
self._staggered_reset_pending = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
```

2. Changed `step(...)` so early-finished envs wait for the next gate instead of resetting immediately:

```python
previous_pending = self._staggered_reset_pending.clone()
if self._staggered_reset_enabled and torch.any(previous_pending):
    action = action.clone()
    action[previous_pending] = 0.0

...

if self._staggered_reset_enabled:
    self.episode_length_buf[~previous_pending] += 1
else:
    self.episode_length_buf += 1

...

if self._staggered_reset_enabled and torch.any(previous_pending):
    self.reward_buf[previous_pending] = 0.0

if self._staggered_reset_enabled:
    newly_terminated = self.reset_buf & ~previous_pending
    self._staggered_reset_pending |= newly_terminated
    self.reset_terminated &= ~previous_pending
    self.reset_time_outs &= ~previous_pending
    self.reset_buf = self.reset_terminated | self.reset_time_outs
    self.extras["staggered_invalid"] = previous_pending.clone()
    self.extras["staggered_pending"] = self._staggered_reset_pending.clone()
```

This is the core deferred-reset behavior:

- already-pending envs are frozen
- they do not keep accumulating reward or episode age
- newly terminated envs become pending
- pending steps are marked invalid for the learner

3. Added explicit gate application instead of immediate reset:

```python
def configure_staggered_reset_gating(self, enabled: bool):
    self._staggered_reset_enabled = enabled
    self._staggered_reset_pending.zero_()
    self.extras["staggered_invalid"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.extras["staggered_pending"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

def apply_staggered_reset_gate(self, env_ids: Sequence[int]) -> torch.Tensor:
    gate_env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    reset_mask = self._staggered_reset_pending.clone()
    if gate_env_ids.numel() > 0:
        reset_mask[gate_env_ids] = True
    reset_env_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
    if reset_env_ids.numel() > 0:
        self.reset(env_ids=reset_env_ids)
        self._staggered_reset_pending[reset_env_ids] = False
    return reset_env_ids
```

4. Reset now also clears pending state:

```python
self.episode_length_buf[env_ids] = 0
self._staggered_reset_pending[env_ids] = False
```

These four changes are the complete Isaac Lab side of the feature.

Why Isaac Lab had to change:

- the paper-style algorithm requires deferred resets during training
- default Isaac Lab resets immediately inside `step(...)`
- immediate reset destroys the phase separation created by staggering

So Isaac Lab is where the runtime semantics changed from:

- terminate -> immediate reset

to:

- terminate -> mark pending -> wait for gate -> batched reset

### What changed in `rsl_rl`

Files:

- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/runners/on_policy_runner.py`
- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/storage/rollout_storage.py`
- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/algorithms/ppo.py`
- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/algorithms/mdpo.py`
- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/algorithms/spo.py`
- `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/utils/wandb_utils.py`

Exactly what changed in the runner:

- added startup staggered warm-up over discrete horizon phases
- distributed bucket phases across the full `H / K` range, not just a local startup offset
- added rollout-boundary reset-gate application
- added bucket bookkeeping:
  - env -> bucket assignment
  - current bucket phase
  - maximum rollout phases
- reset recurrent policy memory after gate resets
- cleared warm-up transitions before real learning starts
- passed `transition_valid` masks from env extras into the algorithms
- added warm-up progress reporting with `tqdm`

Why the runner had to change:

- it owns synchronous `K`-step rollout collection
- reset gates must align with rollout boundaries
- the runner is the only place that can enforce:
  - initial staggered starts
  - synchronous collection after staggering
  - gate application between rollouts

Exactly what changed in rollout storage:

- added `valid_mask` to transition data
- stored it per time step and env
- emitted it from feedforward and recurrent mini-batch generators

#### RNN / padded-trajectory fix

While integrating staggered resets with recurrent training, we hit a shape mismatch in the recurrent update path.

What caused it:

- recurrent observation batches are padded into trajectory form with `split_and_pad_trajectories(...)`
- but PPO / MDPO advantage, return, value, and log-prob tensors are still consumed in raw rollout layout
- I initially paired the raw advantage tensor with a padded validity mask

That was wrong because the two tensors describe different indexing schemes:

- padded trajectory layout:
  - number of packed trajectories after splitting on `done`
- raw rollout layout:
  - `[T, env_batch, ...]`

This is why the recurrent MDPO update failed with a mismatch like:

- raw width: `512`
- padded trajectory width: `529`

The failure appeared when `_masked_normalize(...)` tried to apply the validity mask to recurrent advantages.

The fix was to keep the two masks separate:

- `trajectory_masks`
  - still used only for padded recurrent forward passes
- `valid_mask_batch`
  - changed to stay in raw rollout layout, aligned with:
    - `advantages_batch`
    - `returns_batch`
    - `values_batch`
    - `old_actions_log_prob_batch`

Relevant code change in `rollout_storage.py`:

```python
masks_batch = trajectory_masks[:, first_traj:last_traj]
obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

actions_batch = self.actions[:, start:stop]
old_mu_batch = self.mu[:, start:stop]
old_sigma_batch = self.sigma[:, start:stop]
returns_batch = self.returns[:, start:stop]
advantages_batch = self.advantages[:, start:stop]
values_batch = self.values[:, start:stop]
old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
valid_mask_batch = self.valid_mask[:, start:stop]
```

And the recurrent algorithms now apply the mask to raw tensors, for example in `mdpo.py`:

```python
with torch.no_grad():
    advantages_batch_1 = _masked_normalize(advantages_batch_1, valid_mask_batch_1)
    advantages_batch_2 = _masked_normalize(advantages_batch_2, valid_mask_batch_2)
```

Why this is the correct fix:

- the RNN still needs padded sequence masks for forward propagation over split trajectories
- the optimizer still needs validity masks in the same layout as the loss tensors
- those are not the same object once padding is introduced

So the recurrent solution is:

- padded `trajectory_masks` for sequence execution
- raw `valid_mask_batch` for staggered-reset invalid-step masking

Why storage had to change:

- deferred-reset waiting steps are collected physically, but they are not valid training samples
- the mask must survive from collection to minibatch construction

Exactly what changed in PPO / MDPO / SPO:

- added masked reduction helpers
- normalized advantages with masks
- masked:
  - surrogate loss
  - value loss
  - entropy term
- for PPO: masked KL averaging
- for MDPO: masked mutual-distillation KL as well

Why the algorithms had to change:

- otherwise the learner would optimize on frozen waiting steps
- that would bias both policy and value learning
- in MDPO it would also bias distillation between the two policies

Exactly what changed in `wandb_utils.py`:

- replaced the unstepped `wandb.log({"log_dir": ...})` call with `wandb.run.summary["log_dir"] = ...`

Why:

- to avoid W&B step-order warnings when the first real training metrics are logged at iteration `0`

### What changed in `sru-navigation-sim`

Files:

- `/home/chongxi/Work/Astera/navigation/sru-navigation-sim/scripts/train.py`
- `/home/chongxi/Work/Astera/navigation/sru-navigation-sim/scripts/train_second_stage.py`
- `/home/chongxi/Work/Astera/navigation/sru-navigation-sim/docs/STAGGERED_RESETS.md`

Exactly what changed:

- added CLI flag:
  - `--staggered_reset_buckets`
- passed the flag into `runner.learn(...)`
- documented:
  - what was changed
  - why it helps
  - how to use it
  - which files were touched

Why:

- the simulation repo is where experiments are launched
- the runner feature had to be exposed without hard-coding it
- the documentation belongs with the project that uses the feature

## Appendix: File-by-File Changes

### `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/runners/on_policy_runner.py`

What changed:

- added startup pre-roll staggering over discrete horizon phases
- added runner-side bookkeeping for stagger buckets and bucket phases
- added rollout-boundary reset-gate application
- added runner-side recurrent memory resets after gate resets
- added invalid-step handling to rollout logging
- extended `learn(...)` with `staggered_reset_buckets`

Why:

- the runner owns rollout collection, so it is the right place to:
  - decide bucket structure,
  - apply the initial stagger,
  - trigger reset gates every `K` steps,
  - keep policy memory consistent after gate resets.

Without runner changes, we could not align staggering to rollout boundaries.

### `/home/chongxi/Work/Astera/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`

What changed:

- added optional staggered-reset gating state:
  - `_staggered_reset_enabled`
  - `_staggered_reset_pending`
- changed `step(...)` so terminated envs are not immediately reset when gated mode is enabled
- added freezing behavior for pending envs between early termination and next gate
- added extras for:
  - `staggered_invalid`
  - `staggered_pending`
- added:
  - `configure_staggered_reset_gating(...)`
  - `apply_staggered_reset_gate(...)`
- ensured `_reset_idx(...)` clears pending-reset state

Why:

- the paper’s algorithm requires deferred resets during training
- Isaac Lab’s default behavior was immediate reset inside `step(...)`
- that immediate reset destroyed the stagger structure after the first few failures

So this file had to change to make gate-based reset scheduling possible at all.

### `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/storage/rollout_storage.py`

What changed:

- added `valid_mask` to stored transitions
- saved `valid_mask` into rollout storage
- extended mini-batch generators to emit `valid_mask_batch`

Why:

- once early-terminated envs are frozen until the next gate, some collected steps are not valid learning data
- those waiting steps must be excluded from advantage normalization and losses

Storage needed to carry that mask from collection to optimization.

### `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/algorithms/ppo.py`

What changed:

- added masked helpers:
  - `_align_mask`
  - `_masked_mean`
  - `_masked_normalize`
- stored `transition.valid_mask`
- changed PPO update to use masked:
  - KL averaging
  - advantage normalization
  - surrogate loss
  - value loss
  - entropy term

Why:

- PPO would otherwise optimize on frozen invalid steps
- that would bias both the policy and value losses

Masking keeps the effective batch equal to only the valid staggered transitions.

### `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/algorithms/mdpo.py`

What changed:

- added the same masked helpers as PPO
- stored per-policy valid masks in `process_env_step(...)`
- changed PPO-style losses inside MDPO to use masked:
  - advantage normalization
  - surrogate loss
  - value loss
  - entropy term
- changed mutual distillation KL to use masked means

Why:

- MDPO uses the same rollout data problem as PPO
- both policy branches needed to ignore frozen invalid steps
- distillation also had to ignore invalid samples, otherwise the two policies would regularize each other on stale waiting states

### `/home/chongxi/Work/Astera/navigation/sru-navigation-learning/rsl_rl/algorithms/spo.py`

What changed:

- updated to the new storage interface
- added valid-mask handling and masked losses

Why:

- even though the current navigation training mostly uses MDPO or PPO, `SPO` shares the same storage interface
- once storage started returning `valid_mask_batch`, `SPO` also needed to be updated to keep the library internally consistent

### `/home/chongxi/Work/Astera/navigation/sru-navigation-sim/scripts/train.py`

What changed:

- added CLI flag:
  - `--staggered_reset_buckets`
- passed it into `runner.learn(...)`

Why:

- this exposes staggered resets for normal from-scratch training

### `/home/chongxi/Work/Astera/navigation/sru-navigation-sim/scripts/train_second_stage.py`

What changed:

- added the same CLI flag:
  - `--staggered_reset_buckets`
- passed it into `runner.learn(...)`

Why:

- second-stage continuation was the main workflow where staggered resets were being tested
- the same algorithmic option needed to be available there without duplicating runner logic

## Appendix: Why These Changes Are Split Across Files

The implementation is intentionally split because staggered resets are not only a reset policy.

They require coordinated changes across:

- environment stepping
- rollout collection
- storage
- optimization

If only one layer changed, the algorithm would be incomplete:

- only env changes: resets would be deferred, but the learner would still optimize on invalid waiting steps
- only runner changes: envs would still reset immediately, so staggering would decay
- only storage/loss changes: nothing would create or preserve the gate structure

That is why the full implementation touches both Isaac Lab and `rsl_rl`, plus the train entrypoints that expose the feature.
