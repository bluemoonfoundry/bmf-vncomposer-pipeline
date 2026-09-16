"""Onboard an environment/set asset (DAZ-sourced or natively Blender-
importable) into a normalized Environment_<name> collection.

Run headlessly against a fresh .blend (mirrors import_daz_artifact.py):

    blender -b -P blender/import_environment_asset.py -- \
        --source <path-to-asset> --collection-name <name> --out <output.blend> \
        [--root-paths <daz-root-paths.json>]

Supports DAZ .dbz/.duf sets and props via the same easy_import_daz path used
for characters (with posing/rig-merge options off, since a set/prop has no
posable figure to fit), and natively Blender-importable formats (glTF/GLB,
FBX, OBJ, USD) via bpy's own import_scene/wm operators. Both paths normalize
into a single Environment_<name> collection: new objects are flattened out of
whatever collection structure the importer created (see collection_utils.py),
any embedded camera/light objects are tagged with a scarecrow_env_role custom
property so Phase 3's render_scene() can find or override them, and a
world-space bounding box over all mesh objects is stored as
scarecrow_bbox_min/scarecrow_bbox_max custom properties on the collection.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent

from collection_utils import (
    flatten_into_collection,
    new_objects_since,
    remove_empty_scratch_collections,
    snapshot_names,
)

# scarecrow_pipeline/environment_asset.py is intentionally bpy-free and has no
# third-party dependencies, but importing it via the `scarecrow_pipeline`
# package would trigger scarecrow_pipeline/__init__.py, which eagerly imports
# schemas.py (requires pydantic). Blender's bundled Python has no pydantic, so
# load the module directly from its file path instead, bypassing the package
# __init__ entirely.
_ENV_ASSET_SPEC = importlib.util.spec_from_file_location(
    "environment_asset", REPO_ROOT / "scarecrow_pipeline" / "environment_asset.py"
)
_environment_asset = importlib.util.module_from_spec(_ENV_ASSET_SPEC)
_ENV_ASSET_SPEC.loader.exec_module(_environment_asset)
resolve_handler = _environment_asset.resolve_handler
compute_bounding_box = _environment_asset.compute_bounding_box


def import_daz_set(source_path, root_paths_path):
    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.import_daz")
    from bl_ext.user_default.import_daz import api

    if root_paths_path:
        from bl_ext.user_default.import_daz.settings import GS
        with open(root_paths_path, encoding="utf-8-sig") as handle:
            GS.readDazPaths(json.load(handle), None, True)

    api.set_silent_mode(True)
    result = bpy.ops.daz.easy_import_daz(
        directory=os.path.dirname(source_path),
        files=[{"name": os.path.basename(source_path)}],
        fitMeshes="UNIQUE",
        useMakePosable=False,
        useHead=False,
        useUnits=False,
        useExpressions=False,
        useVisemes=False,
        useFacs=False,
        useFacsdetails=False,
        useFacsexpr=False,
        useBody=False,
        useMergeRigs=False,
        useMergeMaterials=True,
        useTransferClothes=False,
        useTransferFace=False,
        useTransferGeografts=False,
        useTransferHD=False,
    )
    api.set_silent_mode(False)
    print("IMPORT_RESULT", result)


def import_native(source_path, handler):
    if handler == "fbx":
        bpy.ops.import_scene.fbx(filepath=source_path)
    elif handler == "gltf":
        bpy.ops.import_scene.gltf(filepath=source_path)
    elif handler == "obj":
        bpy.ops.wm.obj_import(filepath=source_path)
    elif handler == "usd":
        bpy.ops.wm.usd_import(filepath=source_path)
    else:
        raise ValueError(f"No native import operator wired for handler {handler!r}")


def tag_camera_and_light_roles(objects):
    for obj in objects:
        if obj.type == "CAMERA":
            obj["scarecrow_env_role"] = "camera"
        elif obj.type == "LIGHT":
            obj["scarecrow_env_role"] = "light"


def store_bounding_box(collection):
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in collection.objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not corners:
        print(f"WARNING: no mesh objects in {collection.name!r}; skipping bounding box")
        return
    bbox_min, bbox_max = compute_bounding_box([tuple(corner) for corner in corners])
    collection["scarecrow_bbox_min"] = bbox_min
    collection["scarecrow_bbox_max"] = bbox_max


def onboard_environment(source_path, collection_name, out_path, root_paths_path):
    source_path = os.path.abspath(source_path)
    handler = resolve_handler(source_path)

    pre_object_names, pre_collection_names = snapshot_names()

    if handler == "daz":
        import_daz_set(source_path, root_paths_path)
    else:
        if root_paths_path:
            print(
                f"WARNING: --root-paths {root_paths_path!r} has no effect for native "
                f"asset imports (handler={handler!r}); ignoring."
            )
        import_native(source_path, handler)

    new_objects = new_objects_since(pre_object_names)
    if not new_objects:
        raise RuntimeError(f"No new content found after importing {source_path!r}; nothing to onboard")

    target_name = f"Environment_{collection_name}"
    target = bpy.data.collections.new(target_name)
    bpy.context.scene.collection.children.link(target)
    flatten_into_collection(new_objects, target)
    remove_empty_scratch_collections(pre_collection_names, keep=target)

    tag_camera_and_light_roles(new_objects)
    store_bounding_box(target)

    print("ENVIRONMENT_COLLECTION", target.name, "OBJECTS", [obj.name for obj in target.objects])

    save_path = os.path.abspath(out_path)
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    print("SAVED", save_path)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--root-paths")
    args = parser.parse_args(argv)
    onboard_environment(args.source, args.collection_name, args.out, args.root_paths)


if __name__ == "__main__":
    main()
