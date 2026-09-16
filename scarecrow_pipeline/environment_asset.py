"""Pure-Python (no bpy) helpers for onboarding environment/set assets.

Shared by blender/import_environment_asset.py, kept bpy-free so the
extension-dispatch table and bounding-box math are unit-testable without a
live Blender process.
"""

from __future__ import annotations

from pathlib import Path

SUFFIX_HANDLERS: dict[str, str] = {
    ".dbz": "daz",
    ".duf": "daz",
    ".fbx": "fbx",
    ".gltf": "gltf",
    ".glb": "gltf",
    ".obj": "obj",
    ".usd": "usd",
    ".usda": "usd",
    ".usdc": "usd",
    ".usdz": "usd",
}


def resolve_handler(source_path: str) -> str:
    suffix = Path(source_path).suffix.lower()
    handler = SUFFIX_HANDLERS.get(suffix)
    if handler is None:
        raise ValueError(f"Unsupported environment asset extension: {suffix!r}")
    return handler


def compute_bounding_box(corners: list[tuple[float, float, float]]) -> tuple[list[float], list[float]]:
    if not corners:
        raise ValueError("compute_bounding_box requires at least one point")
    xs, ys, zs = zip(*corners)
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]
