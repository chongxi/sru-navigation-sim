#!/usr/bin/env python3
r"""
Load a colored PLY mesh into Isaac Sim / Isaac Lab while preserving vertex colors.

This is intended for nvblox mesh exports such as:
  data/sample_20260402_i13_nvblox_only/nvblox_out/mesh.ply

Examples
--------
Windows Isaac Sim:
  C:\isaac-sim\python.bat G:\GithubProject\ReconS\utilities\isaacsim_load_colored_ply.py ^
      --mesh-path G:\GithubProject\ReconS\data\sample_20260402_i13_nvblox_only\nvblox_out\mesh.ply

Linux Isaac Lab:
  /mnt/g/GithubProject/IsaacLab/isaaclab.sh -p \
      /mnt/g/GithubProject/ReconS/utilities/isaacsim_load_colored_ply.py \
      --mesh-path /mnt/g/GithubProject/ReconS/data/sample_20260402_i13_nvblox_only/nvblox_out/mesh.ply
"""

from __future__ import annotations

import argparse
import os
import site
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a colored PLY in Isaac Sim / Isaac Lab.")
    parser.add_argument(
        "--mesh-path",
        required=True,
        help="Path to the colored PLY mesh.",
    )
    parser.add_argument(
        "--output-stage",
        help="Optional USD stage path to write (default: <mesh parent>/mesh_colored.usda).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim headless.",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Write the USD stage and exit without opening the interactive UI loop.",
    )
    parser.add_argument(
        "--headless-seconds",
        type=float,
        default=2.0,
        help="When --headless and not --stage-only, keep the app alive this many seconds.",
    )
    return parser.parse_args()


def _create_simulation_app(headless: bool):
    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception:
        from omni.isaac.kit import SimulationApp  # type: ignore
    return SimulationApp({"headless": headless})


def _sanitize_python_env_for_isaac() -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)

    candidates: List[str] = []
    try:
        candidates.append(str(Path(site.getusersitepackages()).resolve()).lower())
    except Exception:
        pass
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(str((Path(appdata) / "Python").resolve()).lower())

    if not candidates:
        return

    cleaned: List[str] = []
    for entry in sys.path:
        if not entry:
            cleaned.append(entry)
            continue
        try:
            resolved = str(Path(entry).resolve()).lower()
        except Exception:
            resolved = entry.lower()
        if not any(c in resolved for c in candidates):
            cleaned.append(entry)
    sys.path[:] = cleaned

    mod = sys.modules.get("torch")
    if mod is not None:
        mod_file = str(getattr(mod, "__file__", "")).lower()
        if any(c in mod_file for c in candidates):
            del sys.modules["torch"]


@dataclass
class _PlyProperty:
    name: str
    dtype: str
    is_list: bool = False
    count_dtype: str | None = None
    item_dtype: str | None = None


@dataclass
class _PlyElement:
    name: str
    count: int
    properties: List[_PlyProperty]


@dataclass
class _MeshData:
    points: List[Tuple[float, float, float]]
    face_counts: List[int]
    face_indices: List[int]
    normals: List[Tuple[float, float, float]] | None
    colors: List[Tuple[float, float, float]] | None
    opacity: List[float] | None


_PLY_TO_STRUCT = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _parse_ply_header(path: Path) -> Tuple[str, List[_PlyElement], int]:
    with path.open("rb") as f:
        first = f.readline().decode("ascii", errors="ignore").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file.")

        fmt = ""
        elements: List[_PlyElement] = []
        current: _PlyElement | None = None

        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Invalid PLY header in {path}: missing end_header.")
            line = raw.decode("ascii", errors="ignore").strip()
            if not line or line.startswith("comment"):
                continue
            if line == "end_header":
                return fmt, elements, f.tell()
            parts = line.split()
            key = parts[0]
            if key == "format":
                fmt = parts[1]
            elif key == "element":
                current = _PlyElement(name=parts[1], count=int(parts[2]), properties=[])
                elements.append(current)
            elif key == "property":
                if current is None:
                    raise ValueError(f"Property without element in {path}: {line}")
                if len(parts) >= 5 and parts[1] == "list":
                    current.properties.append(
                        _PlyProperty(
                            name=parts[4],
                            dtype="list",
                            is_list=True,
                            count_dtype=parts[2],
                            item_dtype=parts[3],
                        )
                    )
                elif len(parts) == 3:
                    current.properties.append(_PlyProperty(name=parts[2], dtype=parts[1]))
                else:
                    raise ValueError(f"Invalid property line in {path}: {line}")


def _read_scalar_binary(f, dtype: str, endian: str) -> float | int:
    fmt_char = _PLY_TO_STRUCT.get(dtype)
    if fmt_char is None:
        raise ValueError(f"Unsupported PLY dtype: {dtype}")
    size = struct.calcsize(fmt_char)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Unexpected EOF while reading binary PLY.")
    return struct.unpack(endian + fmt_char, data)[0]


def _normalize_rgb(values: Tuple[float, float, float]) -> Tuple[float, float, float]:
    scale = 255.0 if max(values) > 1.0 else 1.0
    return (float(values[0]) / scale, float(values[1]) / scale, float(values[2]) / scale)


def _normalize_opacity(value: float) -> float:
    return float(value) / 255.0 if value > 1.0 else float(value)


def _read_vertex_values_ascii(tokens: List[str], properties: List[_PlyProperty]) -> Dict[str, Any]:
    cursor = 0
    values: Dict[str, Any] = {}
    for prop in properties:
        if prop.is_list:
            n = int(tokens[cursor])
            cursor += 1
            arr = [int(tokens[cursor + i]) for i in range(n)]
            cursor += n
            values[prop.name] = arr
        else:
            values[prop.name] = float(tokens[cursor])
            cursor += 1
    return values


def _read_vertex_values_binary(f, properties: List[_PlyProperty], endian: str) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for prop in properties:
        if prop.is_list:
            if prop.count_dtype is None or prop.item_dtype is None:
                raise ValueError("Invalid list property in binary PLY.")
            n = int(_read_scalar_binary(f, prop.count_dtype, endian))
            arr = [_read_scalar_binary(f, prop.item_dtype, endian) for _ in range(n)]
            values[prop.name] = arr
        else:
            values[prop.name] = _read_scalar_binary(f, prop.dtype, endian)
    return values


def _parse_ply(path: Path) -> _MeshData:
    fmt, elements, data_offset = _parse_ply_header(path)
    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format in {path}: {fmt}")

    points: List[Tuple[float, float, float]] = []
    face_counts: List[int] = []
    face_indices: List[int] = []
    normals: List[Tuple[float, float, float]] = []
    colors: List[Tuple[float, float, float]] = []
    opacity: List[float] = []

    vertex_props: List[_PlyProperty] = []
    for elem in elements:
        if elem.name == "vertex":
            vertex_props = elem.properties
            break
    vertex_names = {prop.name for prop in vertex_props}
    has_normals = {"nx", "ny", "nz"}.issubset(vertex_names)
    has_rgb = {"red", "green", "blue"}.issubset(vertex_names) or {"r", "g", "b"}.issubset(vertex_names)
    has_alpha = "alpha" in vertex_names or "a" in vertex_names

    if fmt == "ascii":
        with path.open("r", encoding="ascii", errors="ignore") as f:
            f.seek(data_offset)
            for elem in elements:
                for _ in range(elem.count):
                    line = f.readline()
                    if not line:
                        raise ValueError("Unexpected EOF while reading ascii PLY.")
                    values = _read_vertex_values_ascii(line.strip().split(), elem.properties)
                    if elem.name == "vertex":
                        x = float(values.get("x", 0.0))
                        y = float(values.get("y", 0.0))
                        z = float(values.get("z", 0.0))
                        points.append((x, y, z))
                        if has_normals:
                            normals.append((float(values["nx"]), float(values["ny"]), float(values["nz"])))
                        if has_rgb:
                            r = float(values.get("red", values.get("r", 0.0)))
                            g = float(values.get("green", values.get("g", 0.0)))
                            b = float(values.get("blue", values.get("b", 0.0)))
                            colors.append(_normalize_rgb((r, g, b)))
                            if has_alpha:
                                a = float(values.get("alpha", values.get("a", 1.0)))
                                opacity.append(_normalize_opacity(a))
                    elif elem.name == "face":
                        idx = values.get("vertex_indices") or values.get("vertex_index")
                        if isinstance(idx, list) and idx:
                            face_counts.append(len(idx))
                            face_indices.extend(int(v) for v in idx)
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        with path.open("rb") as f:
            f.seek(data_offset)
            for elem in elements:
                for _ in range(elem.count):
                    values = _read_vertex_values_binary(f, elem.properties, endian)
                    if elem.name == "vertex":
                        x = float(values.get("x", 0.0))
                        y = float(values.get("y", 0.0))
                        z = float(values.get("z", 0.0))
                        points.append((x, y, z))
                        if has_normals:
                            normals.append((float(values["nx"]), float(values["ny"]), float(values["nz"])))
                        if has_rgb:
                            r = float(values.get("red", values.get("r", 0.0)))
                            g = float(values.get("green", values.get("g", 0.0)))
                            b = float(values.get("blue", values.get("b", 0.0)))
                            colors.append(_normalize_rgb((r, g, b)))
                            if has_alpha:
                                a = float(values.get("alpha", values.get("a", 1.0)))
                                opacity.append(_normalize_opacity(a))
                    elif elem.name == "face":
                        idx = values.get("vertex_indices") or values.get("vertex_index")
                        if isinstance(idx, list) and idx:
                            face_counts.append(len(idx))
                            face_indices.extend(int(v) for v in idx)

    if not points:
        raise ValueError(f"No vertices found in PLY: {path}")
    if not face_counts:
        raise ValueError(f"No faces found in PLY: {path}")

    return _MeshData(
        points=points,
        face_counts=face_counts,
        face_indices=face_indices,
        normals=normals if has_normals else None,
        colors=colors if has_rgb else None,
        opacity=opacity if has_rgb and has_alpha else None,
    )


def _compute_extent(points, Gf):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return [
        Gf.Vec3f(float(min(xs)), float(min(ys)), float(min(zs))),
        Gf.Vec3f(float(max(xs)), float(max(ys)), float(max(zs))),
    ]


def _build_stage(output_stage: Path, mesh_path: Path, mesh_data: _MeshData, Usd, UsdGeom, UsdLux, Gf, Sdf) -> None:
    output_stage.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_stage))
    if stage is None:
        raise RuntimeError(f"Failed to create stage: {output_stage}")

    world = UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetDefaultPrim(world.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/World/NvbloxMesh")
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in mesh_data.points])
    mesh.CreateFaceVertexCountsAttr(mesh_data.face_counts)
    mesh.CreateFaceVertexIndicesAttr(mesh_data.face_indices)
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(_compute_extent(mesh_data.points, Gf))

    if mesh_data.normals:
        mesh.CreateNormalsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in mesh_data.normals])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    prim = mesh.GetPrim()
    if mesh_data.colors:
        color_attr = prim.GetAttribute("primvars:displayColor")
        if not color_attr:
            color_attr = prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, False)
        color_var = UsdGeom.Primvar(color_attr)
        color_var.SetInterpolation(UsdGeom.Tokens.vertex)
        color_attr.Set([Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in mesh_data.colors])

        opacity_values = mesh_data.opacity or [1.0] * len(mesh_data.colors)
        opacity_attr = prim.GetAttribute("primvars:displayOpacity")
        if not opacity_attr:
            opacity_attr = prim.CreateAttribute("primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray, False)
        opacity_var = UsdGeom.Primvar(opacity_attr)
        opacity_var.SetInterpolation(UsdGeom.Tokens.vertex)
        opacity_attr.Set([float(v) for v in opacity_values])

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1200.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(5000.0)
    key.CreateAngleAttr(0.5)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    key_xform = UsdGeom.Xformable(key.GetPrim())
    key_xform.AddRotateXYZOp().Set(Gf.Vec3f(55.0, 0.0, 35.0))

    fill = UsdLux.DistantLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(1500.0)
    fill.CreateAngleAttr(1.0)
    fill.CreateColorAttr(Gf.Vec3f(0.8, 0.87, 1.0))
    fill_xform = UsdGeom.Xformable(fill.GetPrim())
    fill_xform.AddRotateXYZOp().Set(Gf.Vec3f(-30.0, 0.0, -120.0))

    stage.GetRootLayer().Save()
    print(f"[info] mesh_path: {mesh_path}")
    print(f"[info] output_stage: {output_stage}")
    print(
        "[info] mesh stats: "
        f"{len(mesh_data.points)} verts, "
        f"{len(mesh_data.face_counts)} faces, "
        f"vertex_colors={'yes' if mesh_data.colors else 'no'}"
    )


def _frame_mesh_in_viewport(mesh_prim_path: str) -> None:
    try:
        import omni.usd  # type: ignore
        from omni.kit.viewport.utility import frame_viewport_selection, get_active_viewport  # type: ignore

        viewport = get_active_viewport()
        if viewport is None:
            return
        omni.usd.get_context().get_selection().set_selected_prim_paths([mesh_prim_path], True)
        frame_viewport_selection(viewport)
    except Exception as exc:
        print(f"[warn] viewport framing failed: {exc}")


def main() -> int:
    args = _parse_args()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if mesh_path.suffix.lower() != ".ply":
        raise ValueError(f"This loader expects a .ply file, got: {mesh_path.suffix}")

    output_stage = (
        Path(args.output_stage).expanduser().resolve()
        if args.output_stage
        else (mesh_path.parent / "mesh_colored.usda").resolve()
    )

    mesh_data = _parse_ply(mesh_path)
    if mesh_data.colors is None:
        raise ValueError(f"No vertex colors found in PLY: {mesh_path}")

    _sanitize_python_env_for_isaac()
    simulation_app = _create_simulation_app(args.headless)
    try:
        import omni.usd  # type: ignore
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # type: ignore

        _build_stage(output_stage, mesh_path, mesh_data, Usd, UsdGeom, UsdLux, Gf, Sdf)

        if args.stage_only:
            return 0

        ctx = omni.usd.get_context()
        if not ctx.open_stage(output_stage.as_posix()):
            raise RuntimeError(f"Failed to open stage in Isaac Sim: {output_stage}")

        for _ in range(20):
            simulation_app.update()

        _frame_mesh_in_viewport("/World/NvbloxMesh")

        if args.headless:
            end_t = time.time() + max(0.1, float(args.headless_seconds))
            while simulation_app.is_running() and time.time() < end_t:
                simulation_app.update()
        else:
            while simulation_app.is_running():
                simulation_app.update()
    finally:
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
