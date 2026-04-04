#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from pxr import Usd, UsdGeom
from scipy.ndimage import binary_dilation, label

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from isaaclab_nav_task.terrains.terrain_constants import HORIZONTAL_SCALE, PADDING, VERTICAL_SCALE


GEOM_TYPES = set(PRIMITIVE_MESH_TYPES) | {"Mesh"}


@dataclass
class UsdTerrainBuildResult:
    merged_mesh_path: str
    num_source_prims: int
    terrain_size: float
    terrain_origin_xy: tuple[float, float]
    horizontal_scale: float
    coarse_cell_size: float
    floor_height: float
    height_field_visual: torch.Tensor
    height_field_valid_mask: torch.Tensor
    height_field_platform_mask: torch.Tensor
    height_field_spawn_mask: torch.Tensor


def _path_has_name_token(path: str, token: str) -> bool:
    return token in path.strip("/").split("/")


def _should_skip_geom(path: str) -> bool:
    return _path_has_name_token(path, "Robot") or _path_has_name_token(path, "NvbloxMesh")


def _collect_geom_prims(scene_root_path: str) -> list[Usd.Prim]:
    stage = sim_utils.get_current_stage()
    scene_root = stage.GetPrimAtPath(scene_root_path)
    if not scene_root.IsValid():
        raise ValueError(f"Invalid USD scene root: {scene_root_path}")

    geom_prims: list[Usd.Prim] = []
    for prim in Usd.PrimRange(scene_root):
        if not prim.IsValid() or not prim.IsActive():
            continue
        if prim.GetTypeName() not in GEOM_TYPES:
            continue
        path = prim.GetPath().pathString
        if _should_skip_geom(path):
            continue
        geom_prims.append(prim)

    if not geom_prims:
        raise ValueError(f"No non-Nvblox geometry found under {scene_root_path}")
    return geom_prims


def _merge_geom_prims(geom_prims: list[Usd.Prim]) -> tuple[np.ndarray, np.ndarray]:
    merged_vertices: list[np.ndarray] = []
    merged_faces: list[np.ndarray] = []
    vertex_offset = 0

    for prim in geom_prims:
        if prim.GetTypeName() == "Mesh":
            mesh = create_trimesh_from_geom_mesh(prim)
        else:
            mesh = create_trimesh_from_geom_shape(prim)

        scale = sim_utils.resolve_prim_scale(prim)
        mesh.apply_scale(scale)

        translation, quat_wxyz = sim_utils.resolve_prim_pose(prim)
        rotation = math_utils.matrix_from_quat(torch.tensor(quat_wxyz, dtype=torch.float32)).cpu().numpy()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(translation, dtype=np.float64)
        mesh.apply_transform(transform)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            continue

        merged_vertices.append(vertices)
        merged_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)

    if not merged_vertices or not merged_faces:
        raise ValueError("Merged USD geometry produced no triangles.")

    return np.concatenate(merged_vertices, axis=0), np.concatenate(merged_faces, axis=0)


def _estimate_floor_height(hit_z: np.ndarray, normal_z: np.ndarray) -> float:
    finite = np.isfinite(hit_z)
    if not finite.any():
        raise ValueError("USD terrain sampling found no valid top-down ray hits.")

    candidates = hit_z[finite & (normal_z >= 0.85)]
    if candidates.size == 0:
        candidates = hit_z[finite]

    lower_half = candidates[candidates <= np.quantile(candidates, 0.5)]
    if lower_half.size == 0:
        lower_half = candidates

    bin_size = max(VERTICAL_SCALE * 4.0, 0.02)
    bins = np.arange(lower_half.min() - bin_size, lower_half.max() + 2 * bin_size, bin_size)
    if bins.size < 2:
        return float(np.median(lower_half))

    hist, edges = np.histogram(lower_half, bins=bins)
    bin_idx = int(hist.argmax())
    in_bin = (candidates >= edges[bin_idx]) & (candidates < edges[bin_idx + 1])
    floor_band = candidates[in_bin] if in_bin.any() else lower_half
    return float(np.median(floor_band))


def _apply_padding(valid_mask: np.ndarray, padding_cells: int) -> np.ndarray:
    if padding_cells <= 0:
        return valid_mask.copy()
    kernel = np.ones((2 * padding_cells + 1, 2 * padding_cells + 1), dtype=bool)
    return ~binary_dilation(~valid_mask, structure=kernel)


def _exclude_borders(mask: np.ndarray, border_cells: int) -> np.ndarray:
    result = mask.copy()
    if border_cells <= 0:
        return result
    result[:border_cells, :] = False
    result[-border_cells:, :] = False
    result[:, :border_cells] = False
    result[:, -border_cells:] = False
    return result


def _apply_height_transition_padding(
    heights_units: np.ndarray,
    valid_mask: np.ndarray,
    height_threshold_units: int,
    padding_cells: int,
) -> np.ndarray:
    grad_x = np.abs(np.diff(heights_units, axis=0, prepend=heights_units[:1, :]))
    grad_y = np.abs(np.diff(heights_units, axis=1, prepend=heights_units[:, :1]))
    grad_x_back = np.abs(np.diff(heights_units, axis=0, append=heights_units[-1:, :]))
    grad_y_back = np.abs(np.diff(heights_units, axis=1, append=heights_units[:, -1:]))
    transitions = np.maximum.reduce([grad_x, grad_y, grad_x_back, grad_y_back]) >= height_threshold_units
    if padding_cells > 0:
        kernel = np.ones((2 * padding_cells + 1, 2 * padding_cells + 1), dtype=bool)
        transitions = binary_dilation(transitions, structure=kernel)
    return valid_mask & ~transitions


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    connectivity = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    labeled, num_labels = label(mask, structure=connectivity)
    if num_labels <= 1:
        return mask.copy()

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    return labeled == int(component_sizes.argmax())


def _largest_enclosed_component(mask: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    connectivity = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    labeled, num_labels = label(mask, structure=connectivity)
    if num_labels <= 1:
        return mask.copy()

    outside_mask = ~finite_mask
    interior_labels: list[int] = []
    for component_id in range(1, num_labels + 1):
        component_mask = labeled == component_id
        if not component_mask.any():
            continue
        component_with_margin = binary_dilation(component_mask, structure=connectivity)
        if not np.any(component_with_margin & outside_mask):
            interior_labels.append(component_id)
    if not interior_labels:
        return _largest_connected_component(mask)

    component_sizes = np.bincount(labeled.ravel())
    largest_interior = max(interior_labels, key=lambda component_id: int(component_sizes[component_id]))
    return labeled == largest_interior


def build_navigation_terrain_from_usd_stage(
    scene_root_path: str,
    merged_mesh_path: str,
    device: str,
    goal_padding_cells: int | None = None,
    spawn_padding_cells: int | None = None,
    coarse_cell_size: float = 2.0,
    horizontal_scale: float = HORIZONTAL_SCALE,
    walkable_height_tolerance: float = 0.12,
) -> UsdTerrainBuildResult:
    geom_prims = _collect_geom_prims(scene_root_path)
    points, faces = _merge_geom_prims(geom_prims)

    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)
    scene_center_xy = 0.5 * (min_xy + max_xy)
    scene_span = float(np.max(max_xy - min_xy))

    margin_cells = max(
        PADDING.BORDER_CELLS + 4,
        (goal_padding_cells if goal_padding_cells is not None else PADDING.GOAL_PADDING) + 2,
        (spawn_padding_cells if spawn_padding_cells is not None else PADDING.SPAWN_PADDING) + 2,
    )
    num_cells = int(math.ceil(scene_span / horizontal_scale)) + 2 * margin_cells
    terrain_size = (num_cells + 1) * horizontal_scale

    axis = (np.arange(num_cells, dtype=np.float32) + 1.0) * horizontal_scale - 0.5 * terrain_size
    local_x, local_y = np.meshgrid(axis, axis, indexing="ij")
    world_x = scene_center_xy[0] + local_x
    world_y = scene_center_xy[1] + local_y

    z_top = float(points[:, 2].max() + 2.0)
    z_bottom = float(points[:, 2].min() - 2.0)
    max_dist = max(10.0, z_top - z_bottom + 2.0)

    ray_starts = np.stack([world_x, world_y, np.full_like(world_x, z_top)], axis=-1)
    ray_starts_t = torch.from_numpy(ray_starts).view(1, -1, 3).to(device=device, dtype=torch.float32)
    ray_dirs_t = torch.zeros_like(ray_starts_t)
    ray_dirs_t[..., 2] = -1.0

    warp_mesh = convert_to_warp_mesh(points, faces, device=device)
    ray_hits, _, ray_normals, _ = raycast_mesh(
        ray_starts_t,
        ray_dirs_t,
        mesh=warp_mesh,
        max_dist=max_dist,
        return_normal=True,
    )

    hit_z = ray_hits[0, :, 2].detach().cpu().numpy().reshape(num_cells, num_cells)
    normal_z = ray_normals[0, :, 2].detach().cpu().numpy().reshape(num_cells, num_cells)
    floor_height = _estimate_floor_height(hit_z, normal_z)

    finite = np.isfinite(hit_z)
    walkable = finite & (normal_z >= 0.85) & (np.abs(hit_z - floor_height) <= walkable_height_tolerance)
    walkable = _largest_enclosed_component(walkable, finite)
    heights_units = np.round(np.where(finite, hit_z, floor_height) / VERTICAL_SCALE).astype(np.int16)

    base_valid_mask = _apply_height_transition_padding(
        heights_units=heights_units,
        valid_mask=walkable,
        height_threshold_units=max(PADDING.HEIGHT_TRANSITION_THRESHOLD, int(round(0.10 / VERTICAL_SCALE))),
        padding_cells=PADDING.HEIGHT_TRANSITION_PADDING,
    )
    base_valid_mask = _largest_connected_component(base_valid_mask)

    goal_pad = int(goal_padding_cells) if goal_padding_cells is not None else PADDING.GOAL_PADDING
    spawn_pad = int(spawn_padding_cells) if spawn_padding_cells is not None else PADDING.SPAWN_PADDING

    valid_mask = _exclude_borders(_apply_padding(base_valid_mask, goal_pad), PADDING.BORDER_CELLS)
    while not valid_mask.any() and goal_pad > 0:
        goal_pad -= 1
        valid_mask = _exclude_borders(_apply_padding(base_valid_mask, goal_pad), PADDING.BORDER_CELLS)
    valid_mask = _largest_connected_component(valid_mask)

    spawn_mask = _exclude_borders(_apply_padding(base_valid_mask, spawn_pad), PADDING.BORDER_CELLS) & valid_mask
    while not spawn_mask.any() and spawn_pad > 0:
        spawn_pad -= 1
        spawn_mask = _exclude_borders(_apply_padding(base_valid_mask, spawn_pad), PADDING.BORDER_CELLS) & valid_mask

    if not valid_mask.any():
        valid_mask = _exclude_borders(base_valid_mask, PADDING.BORDER_CELLS)
    if not spawn_mask.any():
        spawn_mask = valid_mask.copy()
    if not valid_mask.any():
        raise ValueError("USD terrain generation produced no valid goal cells after padding.")
    if not spawn_mask.any():
        raise ValueError("USD terrain generation produced no valid spawn cells after padding.")

    stage = sim_utils.get_current_stage()
    if stage.GetPrimAtPath(merged_mesh_path):
        stage.RemovePrim(merged_mesh_path)

    merged_mesh = UsdGeom.Mesh.Define(stage, merged_mesh_path)
    merged_mesh.GetPointsAttr().Set(points.tolist())
    merged_mesh.GetFaceVertexCountsAttr().Set(np.full((faces.shape[0],), 3, dtype=np.int32).tolist())
    merged_mesh.GetFaceVertexIndicesAttr().Set(faces.reshape(-1).tolist())
    merged_mesh.GetSubdivisionSchemeAttr().Set("none")
    merged_mesh.GetExtentAttr().Set(
        [
            tuple(points.min(axis=0).tolist()),
            tuple(points.max(axis=0).tolist()),
        ]
    )
    UsdGeom.Imageable(merged_mesh.GetPrim()).MakeInvisible()

    return UsdTerrainBuildResult(
        merged_mesh_path=merged_mesh_path,
        num_source_prims=len(geom_prims),
        terrain_size=terrain_size,
        terrain_origin_xy=(float(scene_center_xy[0]), float(scene_center_xy[1])),
        horizontal_scale=horizontal_scale,
        coarse_cell_size=coarse_cell_size,
        floor_height=floor_height,
        height_field_visual=torch.from_numpy(heights_units.copy()).unsqueeze(0),
        height_field_valid_mask=torch.from_numpy(valid_mask.copy()).unsqueeze(0),
        height_field_platform_mask=torch.from_numpy(np.zeros_like(valid_mask, dtype=bool)).unsqueeze(0),
        height_field_spawn_mask=torch.from_numpy(spawn_mask.copy()).unsqueeze(0),
    )
