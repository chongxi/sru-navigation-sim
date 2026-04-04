#!/usr/bin/env python3
"""Load an nvblox PLY mesh into Isaac Sim with vertex-color OmniPBR material.

Usage:
    python scripts/load_nvblox_mesh.py [path/to/mesh.ply]

If no path is given, defaults to the DA3 streaming mesh.
"""
from __future__ import annotations

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load nvblox mesh in Isaac Sim with vertex colors.")
parser.add_argument(
    "mesh_path",
    nargs="?",
    default="/home/chongxi/Work/scene_reconstruction/Depth-Anything-3/workspace/da3_streaming_IMG_3604/mesh/mesh.ply",
    help="Path to .ply mesh file.",
)
parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale factor for the mesh.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Isaac Sim / USD imports (must come after AppLauncher) ---
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

import trimesh


def create_vertex_color_material(stage, mat_path: str):
    """Create a UsdPreviewSurface material that reads displayColor via PrimvarReader."""
    mat = UsdShade.Material.Define(stage, mat_path)

    # Surface shader
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # PrimvarReader to fetch per-vertex displayColor
    reader = UsdShade.Shader.Define(stage, f"{mat_path}/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)

    # Connect reader output -> diffuseColor input
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )

    return mat


def load_ply_to_usd(stage, mesh_path: str, prim_path: str, scale: float, mat_path: str):
    """Load a PLY file via trimesh and create a UsdGeom.Mesh with vertex colors."""
    mesh = trimesh.load(mesh_path, process=False)
    print(f"[INFO] Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Create the Xform hierarchy
    xform = UsdGeom.Xform.Define(stage, prim_path)
    if scale != 1.0:
        xform.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))

    # Create the mesh prim
    usd_mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/mesh")

    # Points
    points = [Gf.Vec3f(*v) for v in mesh.vertices.tolist()]
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray(points))

    # Face vertex counts and indices
    face_counts = Vt.IntArray([3] * len(mesh.faces))
    face_indices = Vt.IntArray(mesh.faces.flatten().tolist())
    usd_mesh.CreateFaceVertexCountsAttr(face_counts)
    usd_mesh.CreateFaceVertexIndicesAttr(face_indices)

    # Vertex normals
    if not mesh.vertex_normals.any():
        # trimesh didn't compute normals from the file; recompute
        trimesh.smoothing.filter_humphrey(mesh)
    normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals.tolist()]
    usd_mesh.CreateNormalsAttr(Vt.Vec3fArray(normals))
    usd_mesh.SetNormalsInterpolation("vertex")

    # Vertex colors as displayColor primvar (rgb, float, 0-1)
    if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
        rgba = mesh.visual.vertex_colors  # uint8 RGBA
        rgb_float = (rgba[:, :3].astype(np.float64) / 255.0).tolist()
        display_color = Vt.Vec3fArray([Gf.Vec3f(*c) for c in rgb_float])
        primvar_api = UsdGeom.PrimvarsAPI(usd_mesh)
        cpv = primvar_api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        cpv.Set(display_color)
        print(f"[INFO] Set {len(rgb_float)} vertex colors as displayColor primvar")

    # Bind the vertex-color material
    mat = create_vertex_color_material(stage, mat_path)
    UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(mat)
    print(f"[INFO] Bound OmniPBR (vertex color) material: {mat_path}")

    return usd_mesh


def main():
    stage = omni.usd.get_context().get_stage()

    # Add a dome light so the mesh is visible
    dome_light = UsdGeom.Xform.Define(stage, "/World/DomeLight")
    from pxr import UsdLux
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight/Light")
    light.CreateIntensityAttr(1500.0)

    # Load the mesh
    load_ply_to_usd(
        stage,
        mesh_path=args.mesh_path,
        prim_path="/World/NvbloxMesh",
        scale=args.scale,
        mat_path="/World/Looks/NvbloxVertexColorMat",
    )

    print("[INFO] Mesh loaded. Running simulation loop (close window to exit).")
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
