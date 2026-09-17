"""Import a Diffeomorphic DBZ artifact in Blender background mode.

With --collection-name, the new top-level collection(s) DAZ's importer
creates are wrapped (not flattened -- see collection_utils.py) into one
collection named --collection-name, matching the bare-<CharacterName>
convention scarecrow_pipeline/registry.py's CharacterRecord.collection and
blender/worker.py's append_collection()/resolve_armature() expect. Wrapping
rather than flattening preserves the nested collection structure DAZ's
importer creates, since blender/author_outfit_variant.py later links
Outfit_<name>/Hair_<name> variant collections as children of this one.
"""

import argparse
import json
import os
import sys

import bpy


def normalize_character_collection(collection_name, pre_import_container, pre_children):
    """Wrap the collection(s) DAZ's importer created under pre_import_container.

    DAZ's importer links new content into whatever collection was *active*
    at import time (bpy.context.collection), not necessarily the scene root
    -- in a fresh default-startup Blender scene that's the pre-existing
    "Collection" holding the default Cube/Camera/Light, one level below the
    scene root. Scanning only scene.collection.children misses it.
    """
    new_top_level = [c for c in pre_import_container.children if c.name not in pre_children]
    if not new_top_level:
        raise RuntimeError(
            f"No new top-level collection found after import; nothing to name {collection_name!r}"
        )
    target = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(target)
    for collection in new_top_level:
        pre_import_container.children.unlink(collection)
        target.children.link(collection)
    return target


def main():
    # Always invoked against Blender's default startup scene (no --blend to
    # load from); the factory Cube/Camera/Light would otherwise sit unused
    # in pre_import_container, and get pulled into the imported collection
    # by normalize_character_collection() if DAZ's importer happens to
    # target that same collection.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbz", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--root-paths")
    parser.add_argument("--collection-name")
    args = parser.parse_args(argv)

    from bl_ext.user_default import import_daz

    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.import_daz")
    from bl_ext.user_default.import_daz import api
    if args.root_paths:
        from bl_ext.user_default.import_daz.settings import GS
        with open(args.root_paths, encoding="utf-8-sig") as handle:
            GS.readDazPaths(json.load(handle), None, True)
        print("ROOT_PATHS_LOADED", api.get_absolute_path("/data/daz 3d/built-in content/daz iray pbrskin/pbrskin.dsf"))

    pre_import_container = bpy.context.view_layer.active_layer_collection.collection
    pre_children = {c.name for c in pre_import_container.children}

    api.set_silent_mode(True)
    dbz = os.path.abspath(args.dbz)
    result = bpy.ops.daz.easy_import_daz(
        directory=os.path.dirname(dbz),
        files=[{"name": os.path.basename(dbz)}],
        fitMeshes="DBZFILE",
        useMakePosable=True,
        useHead=True,
        useUnits=True,
        useExpressions=True,
        useVisemes=True,
        useFacs=True,
        useFacsdetails=True,
        useFacsexpr=True,
        useBody=True,
        useMergeRigs=True,
        useMergeMaterials=True,
        useTransferClothes=True,
        useTransferFace=True,
        useTransferGeografts=True,
        useTransferHD=False,
    )
    api.set_silent_mode(False)
    print("IMPORT_RESULT", result)
    print("OBJECTS", len(bpy.data.objects), "MESHES", len(bpy.data.meshes), "ARMATURES", sum(obj.type == "ARMATURE" for obj in bpy.data.objects))

    if args.collection_name:
        target = normalize_character_collection(args.collection_name, pre_import_container, pre_children)
        print("CHARACTER_COLLECTION", target.name)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend))


if __name__ == "__main__":
    main()
