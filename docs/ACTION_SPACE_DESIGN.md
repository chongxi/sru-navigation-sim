# Action Space Design for Diff-Drive Navigation

> Lessons learned from debugging why the diff-drive policy wouldn't learn.
> The core issue: **the action space must be solvable given the observation space**.

## The Solvability Check

Before training, verify: **can the network compute the correct action from its observations?**

If the network observes `O` and must output action `A`, there must exist a function `A = f(O)` that solves the task. If `A` requires information not in `O`, the task is **unsolvable** regardless of network capacity or training time.

## Egocentric vs Allocentric

| Frame | Also called | What it means | Example |
|-------|-------------|---------------|---------|
| **Egocentric** (body frame) | local, relative | Relative to the robot's current pose | "goal is 30° to my left, 5m away" |
| **Allocentric** (world frame) | global, absolute | Fixed world coordinates | "goal is at (15, 22), I'm facing 1.2 rad" |

## Our Observation Space (Egocentric)

All observations are in **body frame**:

| Observation | Frame | Dim |
|-------------|-------|-----|
| `base_lin_vel` | body | 3D — speed relative to robot |
| `base_ang_vel` | body | 3D — rotation rate |
| `projected_gravity` | body | 3D — tilt (pitch/roll, NOT heading) |
| `target_position` | body | 4D — goal direction + log distance relative to robot |
| `depth_image` | body | 2560D — what the robot sees in front |
| `last_action` | — | 3D — previous action |

**Key: no absolute position, no absolute heading.** The agent knows "where is the goal relative to me?" but NOT "where am I in the world?" or "which direction am I facing?"

## Action Space Options: What We Tried

### Option 1: Delta Heading Offset (accumulated)

```
Action: [vx, heading_offset]
Effect: yaw_target += heading_offset (accumulated over time)
```

| Pros | Cons |
|------|------|
| Works with egocentric obs | Accumulates — can spin forever |
| Small output = go straight | Constant output → constant rotation |
| No absolute heading needed | Noisy exploration creates wild spinning |

**Result**: The robot spun in circles. With `policy_scaling=π`, random exploration produced ±2.4 rad/step heading changes, accumulating to 24 rad/s spinning. Even with reduced scaling (0.3 rad/step), the accumulation meant constant small outputs caused persistent rotation.

### Option 2: Absolute Yaw Target (world frame)

```
Action: [vx, yaw_target_world]
Effect: yaw_target = network_output * π  (world frame angle)
```

| Pros | Cons |
|------|------|
| No accumulation | **UNSOLVABLE** — needs absolute heading |
| No drift | Network can't compute world-frame angle from body-frame obs |
| Direct control | ±π wrapping discontinuity |

**Result**: Reward plateaued at ~0. The network was asked to output "face 1.2 rad in world frame" but had no observation of its current world-frame heading. The mapping from body-frame goal direction to world-frame yaw target requires `current_heading` which is not observed. **Fundamentally impossible.**

### Option 3: Relative Heading Target (body frame) ✓

```
Action: [vx, heading_relative]
Effect: yaw_target = current_yaw + network_output * π
```

| Pros | Cons |
|------|------|
| Works with egocentric obs | None significant |
| No accumulation | |
| Output 0 = go straight | |
| Directly maps to body-frame goal | |
| No drift, no spinning | |

**Result**: 48% success rate at iteration 378. The network outputs "turn 30° right from where I'm facing" which directly corresponds to "goal is 30° to my right" in the observation. **Solvable.**

## The Compatibility Matrix

For each (observation frame, action frame) pair, is the task solvable?

```
                          ACTION FRAME
                    ┌─────────────┬──────────────┬───────────────┐
                    │ Delta       │ Absolute     │ Relative to   │
                    │ (accumulate)│ (world)      │ current       │
  ┌─────────────── ┼─────────────┼──────────────┼───────────────┤
  │ Egocentric     │ Solvable*   │ UNSOLVABLE   │ SOLVABLE ✓    │
O │ (body frame    │ but drifts  │ needs heading│ best match    │
B │  obs, no       │ and spins   │ not in obs   │ for ego obs   │
S │  heading)      │             │              │               │
  ├─────────────── ┼─────────────┼──────────────┼───────────────┤
  │ Allocentric    │ Solvable    │ SOLVABLE     │ Solvable      │
  │ (world frame   │ but         │ natural      │ redundant     │
  │  obs, with     │ unnecessary │ match for    │ (has heading  │
  │  heading)      │             │ allo obs     │  but doesn't  │
  │                │             │              │  use it)      │
  └─────────────── ┴─────────────┴──────────────┴───────────────┘

* "Solvable but drifts" = correct instantaneous policy exists,
  but accumulation causes practical issues (spinning, drift)
```

## Design Rule

> **Match the action frame to the observation frame.**
>
> - Egocentric observations → relative/body-frame actions
> - Allocentric observations → absolute/world-frame actions
>
> Mixing frames (ego obs + allo actions) creates an unsolvable task.

## Our Final Design

```
Observations (egocentric):
  goal_direction_body = [dir_x, dir_y, dir_z, log(dist)]  ← "goal is 30° left, 5m"
  depth_image                                               ← "wall 2m ahead"
  base_lin_vel, base_ang_vel                                ← "moving 1m/s forward"

Action (egocentric):
  vx              = forward speed                           ← "drive at 1.5 m/s"
  heading_relative = turn angle from current heading        ← "turn 30° left"

Implementation:
  yaw_target = current_yaw + tanh(network_output) * π
  PID tracks yaw_target at 120Hz → wheel velocities
```

The network learns a simple mapping:
- See goal 30° left → output heading_relative ≈ -0.1 (turn left ~18°)
- See wall ahead → output heading_relative ≈ ±0.3 (turn to avoid)
- See clear path to goal → output heading_relative ≈ 0, vx ≈ high

## What If You Want Allocentric?

If you add absolute heading to observations (e.g., compass/magnetometer), then absolute yaw_target becomes solvable. The observation would need:

```python
# Add to ObservationsCfg:
heading_world = ObsTerm(func=mdp.root_heading_world)  # 2D: [cos(yaw), sin(yaw)]
```

Then absolute action `yaw_target = network_output * π` works because the network can compute:
```
desired_world_heading = current_heading + body_frame_goal_direction
```

But for sim-to-real, egocentric is preferred — no compass/GPS dependency.
