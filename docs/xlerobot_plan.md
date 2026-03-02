# Plan: Add Differential Drive Robot (xlerobot) to SRU Navigation Sim

## Context

The SRU navigation sim currently supports legged-wheeled robots (B2W, AoW-D) that use a **neural low-level locomotion policy** to convert SE2 velocity commands `[vx, vy, omega]` into joint commands. The user wants to add a **2-wheel differential drive robot** called `xlerobot` for sim2real deployment. This robot is fundamentally simpler: no legs, no balancing, and an **analytical kinematic model** replaces the neural low-level policy.

**Key decisions (confirmed by user):**
- Robot name: `xlerobot`
- No existing URDF/USD — create from scratch
- Keep 3D action space `[vx, vy, omega]` but mask `vy=0` after policy output
- Replace neural low-level policy with diff-drive kinematics: `v_left = (vx - omega * L/2) / r`, `v_right = (vx + omega * L/2) / r`

---

## Architecture Overview

### Current B2W Pipeline
```
High-level policy (5Hz) → [vx, vy, omega] → scale/tanh/filter → Neural locomotion policy (50Hz) → 8 joint commands
```

### New xlerobot Pipeline
```
High-level policy (5Hz) → [vx, vy=0, omega] → scale/tanh/filter → Analytical diff-drive kinematics → 2 wheel velocities
```

---

## Files Overview

All paths relative to `isaaclab_nav_task/navigation/` unless noted.

| Action | File | Purpose |
|--------|------|---------|
| CREATE | `assets/data/Robots/Xlerobot/xlerobot.urdf` | Robot URDF (base + 2 wheels + caster) |
| CREATE | `assets/xlerobot.py` | `XLEROBOT_CFG` ArticulationCfg |
| MODIFY | `assets/__init__.py` | Export `XLEROBOT_CFG` |
| CREATE | `mdp/navigation/actions/diff_drive_se2_actions.py` | `DiffDriveNavigationSE2Action` class |
| CREATE | `mdp/navigation/actions/diff_drive_se2_actions_cfg.py` | `DiffDriveNavigationSE2ActionCfg` |
| MODIFY | `mdp/navigation/actions/__init__.py` | Export new action classes |
| MODIFY | `mdp/depth_utils/camera_config.py` | Add `"xlerobot"` to `ROBOT_CAMERA_CONFIGS` |
| CREATE | `config/xlerobot/__init__.py` | `gym.register()` for 6 envs |
| CREATE | `config/xlerobot/navigation_env_cfg.py` | `XlerobotNavigationEnvCfg` + DEV/PLAY variants |
| CREATE | `config/xlerobot/agents/__init__.py` | Agent config module init |
| CREATE | `config/xlerobot/agents/rsl_rl_cfg.py` | MDPO/PPO runner configs |
| MODIFY | `config/__init__.py` | Add `from .xlerobot import *` |

---

## Step-by-step Implementation

### Step 1: Create xlerobot URDF

**File**: `assets/data/Robots/Xlerobot/xlerobot.urdf`

A simple differential drive robot:
- **base_link**: box body (0.4 x 0.3 x 0.15m, mass ~5kg), centered at wheel axle height
- **left_wheel_joint** / **right_wheel_joint**: continuous joints, axis=Y, radius=0.05m, width=0.03m
- **caster_link**: small sphere at rear for stability (free-spinning via continuous joint)
- **Wheel base**: 0.3m (distance between wheel centers)
- **Wheel radius**: 0.05m

Then convert to USD:
```bash
python -m isaaclab.app.converters.urdf_converter --input xlerobot.urdf --output xlerobot.usd
```

### Step 2: Create robot asset config

**File to create**: `assets/xlerobot.py`

```python
XLEROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ASSETS_DIR}/Robots/Xlerobot/xlerobot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),  # wheel radius + clearance
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=50.0,   # rad/s → ~2.5 m/s at r=0.05
            stiffness={".*": 0.0},     # velocity control: zero stiffness
            damping={".*": 5.0},
        ),
    },
)
```

**File to modify**: `assets/__init__.py` — add `from .xlerobot import *` and `"XLEROBOT_CFG"` to `__all__`.

### Step 3: Create DiffDriveNavigationSE2Action class (CRITICAL)

**File to create**: `mdp/navigation/actions/diff_drive_se2_actions.py`

A new `ActionTerm` subclass that:

1. **Same 3D action interface** as `PerceptiveNavigationSE2Action` (action_dim=3)
2. **Same processing pipeline**: scale/offset → tanh → speed bias → low-pass filter
3. **Masks vy=0** at the end of `process_actions()`
4. **Analytical kinematics** in `apply_actions()`:
   ```python
   vx = processed_actions[:, 0]
   omega = processed_actions[:, 2]
   wheel_vel_left  = (vx - omega * wheel_base/2) / wheel_radius
   wheel_vel_right = (vx + omega * wheel_base/2) / wheel_radius
   ```
5. **Directly applies wheel velocities** via a `JointVelocityActionCfg` term — no neural policy, no decimation

**Duck-typing interface requirements** (used by `events.py` and `observations.py`):

| Attribute/Property | Shape | Used by |
|---|---|---|
| `_policy_scaling` | `[num_envs, 3]` | `randomize_action_scale()` in events.py |
| `_policy_bias` | `[num_envs, 3]` | `randomize_action_scale()` in events.py |
| `_per_env_per_dim_low_pass_alpha` | `[num_envs, 3]` | `randomize_low_pass_filter_alpha()` in events.py |
| `processed_actions` (property) | `[num_envs, 3]` | `generated_actions()` in observations.py |
| `low_level_actions` (property) | `[num_envs, 2]` | `last_low_level_action()` in observations.py |
| `cfg.policy_scaling` | `list[3]` | `randomize_action_scale()` in events.py |

**Key design notes:**
- No `low_level_policy_file` loading (the biggest simplification)
- No `low_level_decimation` counter — kinematics run every physics step
- `process_actions()` is called at planning frequency (5Hz), `apply_actions()` at physics rate (200Hz)
- The `observation_group` field is used by `process_actions()` to get `base_lin_vel[:, 0:3]` for speed bias

**File to create**: `mdp/navigation/actions/diff_drive_se2_actions_cfg.py`

```python
@configclass
class DiffDriveNavigationSE2ActionCfg(ActionTermCfg):
    class_type = DiffDriveNavigationSE2Action

    # Diff-drive kinematics
    wheel_base: float = 0.3          # distance between wheel centers (m)
    wheel_radius: float = 0.05       # wheel radius (m)
    wheel_velocity_action: ActionTermCfg = MISSING  # JointVelocityActionCfg for 2 wheels

    # Action processing (same as PerceptiveNavigationSE2ActionCfg)
    use_raw_actions: bool = False
    scale: list[float] = [1.0, 1.0, 1.0]
    offset: list[float] = [0.0, 0.0, 0.0]
    policy_scaling: list[float] = [1.0, 1.0, 1.0]
    observation_group: str = "low_level_policy"
    policy_distr_type: str = "gaussian"
    enable_low_pass_filter: bool = True
    low_pass_filter_alpha: float = 0.5
```

**NOT included** (vs `PerceptiveNavigationSE2ActionCfg`):
- `low_level_policy_file` (no neural policy)
- `low_level_position_action` (no legs)
- `low_level_decimation` (no decimation needed)
- `reorder_joint_list` (simple 2-joint robot)

**File to modify**: `mdp/navigation/actions/__init__.py` — export both new classes.

### Step 4: Add xlerobot camera config

**File to modify**: `mdp/depth_utils/camera_config.py`

Add entry to `ROBOT_CAMERA_CONFIGS`:
```python
"xlerobot": ZEDX_CAMERA_CONFIG,  # reuse ZedX config initially
```

### Step 5: Create xlerobot environment config

**File to create**: `config/xlerobot/navigation_env_cfg.py`

`XlerobotNavigationEnvCfg(NavigationEnvCfg)` — in `__post_init__()`:

```python
# Robot asset
self.scene.robot = XLEROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

# Camera — lower mounting for diff-drive (shorter robot)
self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
self.scene.raycast_camera.offset.pos = (0.20, 0.0, 0.12)
self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

# Actions — DiffDrive replaces PerceptiveNavigation
self.actions.velocity_command = mdp.DiffDriveNavigationSE2ActionCfg(
    asset_name="robot",
    wheel_velocity_action=mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=1.0,
        use_default_offset=True,
    ),
    wheel_base=0.3,
    wheel_radius=0.05,
    observation_group="low_level_policy",
    policy_scaling=[1.5, 1.0, 1.0],
    use_raw_actions=True,
    policy_distr_type="gaussian",
)

# Terminations — only base_link contact (no legs)
self.terminations.base_contact.params = {
    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
    "threshold": 1.0,
}

# Rewards — only wheel joints for joint acceleration
self.rewards.joint_acc_l2_joint.params = {
    "asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel_joint", "right_wheel_joint"]),
}

# Terrain — easier for differential drive
self.scene.terrain.max_init_terrain_level = 5
self.scene.terrain.terrain_generator.difficulty_range = [0.2, 0.7]
self.scene.terrain.terrain_generator.curriculum = True
```

Also create `_DEV` (2 rows, 300 iters) and `_PLAY` (20 envs) variants following B2W pattern.

### Step 6: Create training configs

**File to create**: `config/xlerobot/agents/rsl_rl_cfg.py`

| Config | Algorithm | Key settings |
|--------|-----------|-------------|
| `XlerobotNavMDPORunnerCfg` | MDPO | value_loss_coef=0.0002, schedule=exponential, gamma=0.999 |
| `XlerobotNavMDPORunnerDevCfg` | MDPO | max_iterations=300, logger=tensorboard |
| `XlerobotNavPPORunnerCfg` | PPO | value_loss_coef=0.02, schedule=adaptive, gamma=0.995 |
| `XlerobotNavPPORunnerDevCfg` | PPO | max_iterations=300, logger=tensorboard |

Common settings:
- `ActorCriticSRU` with [512, 256, 128] hidden dims, LSTM_SRU
- `image_input_dims=(64, 5, 8)`, `height_input_dims=(64, 7, 7)`
- `wandb_project="isaaclab_nav_xlerobot"`
- `num_steps_per_env=16`, `save_interval=500`

**File to create**: `config/xlerobot/agents/__init__.py` — imports rsl_rl_cfg module.

### Step 7: Register gym environments

**File to create**: `config/xlerobot/__init__.py`

Register 6 environments:
- `Isaac-Nav-MDPO-Xlerobot-v0` / `-Play-v0` / `-Dev-v0`
- `Isaac-Nav-PPO-Xlerobot-v0` / `-Play-v0` / `-Dev-v0`

Each follows the pattern:
```python
gym.register(
    id="Isaac-Nav-MDPO-Xlerobot-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.XlerobotNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.XlerobotNavMDPORunnerCfg,
    },
)
```

**File to modify**: `config/__init__.py` — add `from .xlerobot import *`.

---

## Observation Dimensions (unchanged)

The high-level navigation policy observations remain the same:

| Group | Dims | Content |
|-------|------|---------|
| policy | 2575 | gravity(3) + lin_vel(3) + ang_vel(3) + last_action(3) + goal(3) + depth(2560) |
| critic | 5712 | gravity(3) + lin_vel(3) + ang_vel(3) + last_action(3) + goal(3) + time(1) + height_scan(3136) + depth(2560) |

The `low_level_policy` observation group still exists but is only used to extract `base_lin_vel[:, 0:3]` for speed bias. The `joint_pos` / `joint_vel` / `actions` terms in that group will have different dimensions (2 joints instead of 8) but this doesn't affect anything since no neural low-level policy reads them.

---

## Compatibility Notes

These existing functions work without modification due to duck-typing:
- `events.py::randomize_action_scale()` — accesses `_policy_scaling`, `_policy_bias`
- `events.py::randomize_low_pass_filter_alpha()` — accesses `_per_env_per_dim_low_pass_alpha`
- `events.py::disable_backward_penalty_after_steps()` — guarded by `hasattr()`, safe to skip
- `observations.py::generated_actions()` — accesses `processed_actions` property
- `observations.py::last_low_level_action()` — accesses `low_level_actions[:, joint_ids]`

---

## Verification

1. **Import test**:
   ```bash
   python -c "import isaaclab_nav_task; import gymnasium; print([k for k in gymnasium.envs.registry.keys() if 'Xlerobot' in k])"
   ```

2. **Dev training** (quick smoke test):
   ```bash
   python scripts/train.py --task Isaac-Nav-PPO-Xlerobot-Dev-v0 --num_envs 64
   ```
   Verify: no crashes, observations flow, actions apply, rewards compute.

3. **Check diff-drive behavior**:
   - vy should always be 0 in logs
   - Wheel velocities should be reasonable (not saturating at limits)
   - Robot should move forward/backward and rotate, not strafe

4. **Full training**:
   ```bash
   python scripts/train.py --task Isaac-Nav-MDPO-Xlerobot-v0 --num_envs 4096
   ```