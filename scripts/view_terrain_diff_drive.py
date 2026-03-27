#!/usr/bin/env python3
"""Visualize the diff-drive terrain config without running any robot.

Loads the exact same terrain as DiffDriveNavigationEnvCfg and lets you
fly around in the viewport to inspect corridors, walls, pits, etc.

Usage:
    python scripts/view_terrain_diff_drive.py
    python scripts/view_terrain_diff_drive.py --num_rows 2 --num_cols 4
    python scripts/view_terrain_diff_drive.py --terrain_type maze
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View diff-drive terrain config.")
parser.add_argument("--num_rows", type=int, default=2, help="Terrain grid rows.")
parser.add_argument("--num_cols", type=int, default=4, help="Terrain grid columns.")
parser.add_argument("--terrain_type", type=str, default=None,
                    choices=["maze", "non_maze", "pits"],
                    help="Show only one terrain type (default: all).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after sim launch ───────────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from isaaclab_nav_task.terrains.patches import apply_terrain_patches
apply_terrain_patches()
from isaaclab_nav_task.terrains import HfMazeTerrainCfg

# ── Terrain config — exact copy from DiffDriveNavigationEnvCfg ────────

ALL_SUB_TERRAINS = {
    "maze": HfMazeTerrainCfg(
        proportion=0.4,
        open_probability=0.95,
        grid_size=(10, 10),
        cell_size=3.0,
        add_noise_to_flat=False,
        add_goal=True,
        randomize_wall=False,
        random_wall_ratio=0.0,
        add_stairs_to_maze=False,
    ),
    "non_maze": HfMazeTerrainCfg(
        proportion=0.3,
        open_probability=0.95,
        grid_size=(10, 10),
        cell_size=3.0,
        add_noise_to_flat=False,
        add_goal=True,
        randomize_wall=True,
        random_wall_ratio=0.2,
        non_maze_terrain=True,
    ),
    "pits": HfMazeTerrainCfg(
        proportion=0.3,
        open_probability=0.95,
        grid_size=(10, 10),
        cell_size=3.0,
        add_noise_to_flat=False,
        add_goal=True,
        randomize_wall=False,
        random_wall_ratio=0.0,
        non_maze_terrain=True,
        dynamic_obstacles=True,
    ),
}

if args_cli.terrain_type is not None:
    sub_terrains = {args_cli.terrain_type: ALL_SUB_TERRAINS[args_cli.terrain_type]}
    sub_terrains[args_cli.terrain_type].proportion = 1.0
else:
    sub_terrains = ALL_SUB_TERRAINS


class ViewSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(30.0, 30.0),
            border_width=20.0,
            num_rows=args_cli.num_rows,
            num_cols=args_cli.num_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            difficulty_range=[0.2, 0.7],
            sub_terrains=sub_terrains,
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            restitution=0.0,
            static_friction=1.0,
            dynamic_friction=0.9,
            compliant_contact_stiffness=5e5,
            compliant_contact_damping=300.0,
        ),
        debug_vis=False,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1/60.0, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.0, 0.0, 50.0], [0.0, 0.0, 0.0])  # top-down view

    scene_cfg = ViewSceneCfg(num_envs=1, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    types_str = args_cli.terrain_type or "maze+non_maze+pits"
    print("\n" + "=" * 64)
    print("  Diff-Drive Terrain Viewer")
    print("=" * 64)
    print(f"  Grid:       {args_cli.num_rows} x {args_cli.num_cols} tiles")
    print(f"  Tile size:  30m x 30m")
    print(f"  Cell size:  3.0m (corridor width ~2.9m)")
    print(f"  Grid cells: 10 x 10 per tile")
    print(f"  Types:      {types_str}")
    print(f"  Difficulty: [0.2, 0.7]")
    print(f"  Camera:     top-down, fly around with mouse/WASD")
    print("=" * 64 + "\n")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
