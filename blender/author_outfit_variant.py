"""Author an Outfit_<name>/Hair_<name> variant collection into an existing master .blend.

Run headlessly against the already-open master blend (mirrors import_daz_artifact.py):

    blender -b <master.blend> -P blender/author_outfit_variant.py -- \
        --dbz <outfit.dbz> --collection-name <name> --kind outfit|hair [--out <versioned.blend>]

Imports a second DAZ outfit/hair DBZ artifact (exported the same way as the base
character), fits it to the character already in the scene by merging the rig the
import creates into the existing armature (bpy.ops.daz.merge_rigs with
useMergeNonConforming='ALL_RIGS', the mode for combining independently-imported
figures rather than a rig's own conforming children -- see merge_rigs.py), then
collects whatever new mesh/empty objects remain into a freshly named
Outfit_<name>/Hair_<name> collection so set_variant_visibility (blender/worker.py:32)
can toggle it.

Because the DBZ is exported "the same way as the base character", its scene graph
typically carries the figure along with the new item, so the import also produces a
second copy of content that already exists in the master (the body, in particular).
Blender de-duplicates colliding object names with a ".NNN" suffix on import, which
doubles as a reliable duplicate signal: any freshly-imported object whose name is
`<existing-name>.NNN` for a name that was already present before this import is
treated as a re-imported duplicate and deleted rather than added to the variant
collection. This is a heuristic -- it would misfire if the new item's own object
legitimately collided by name with something unrelated already in the master -- but
matches how Diffeomorphic's own import naming behaves in practice.
"""

import argparse
import json
import os
import re
import sys

import bpy

DUP_SUFFIX_RE = re.compile(r"^(.*)\.\d{3}$")


def strip_dup_suffix(name):
    match = DUP_SUFFIX_RE.match(name)
    return match.group(1) if match else name


def find_single_armature(objects):
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(
            "Expected exactly one existing armature in the master blend, found "
            f"{len(armatures)}: {[obj.name for obj in armatures]}"
        )
    return armatures[0]


def import_dbz(dbz_path):
    from bl_ext.user_default.import_daz import api

    api.set_silent_mode(True)
    result = bpy.ops.daz.easy_import_daz(
        directory=os.path.dirname(dbz_path),
        files=[{"name": os.path.basename(dbz_path)}],
        fitMeshes="DBZFILE",
        useMakePosable=True,
        useHead=False,
        useUnits=False,
        useExpressions=False,
        useVisemes=False,
        useFacs=False,
        useFacsdetails=False,
        useFacsexpr=False,
        useBody=False,
        useMergeRigs=True,
        useMergeMaterials=True,
        useTransferClothes=True,
        useTransferFace=True,
        useTransferGeografts=True,
        useTransferHD=False,
    )
    api.set_silent_mode(False)
    print("IMPORT_RESULT", result)


def merge_into_master(master_rig, new_rigs):
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for rig in new_rigs:
        rig.select_set(True)
    master_rig.select_set(True)
    bpy.context.view_layer.objects.active = master_rig
    bpy.ops.daz.merge_rigs(
        useOnlySelected=True,
        useHiddenRigs=False,
        duplicateDistance=1.0,
        useMergeNonConforming="ALL_RIGS",
        useConvertWidgets=True,
        useTieRigs=False,
    )


def author_variant(dbz_path, collection_name, kind, out_path, root_paths_path):
    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.import_daz")

    if root_paths_path:
        from bl_ext.user_default.import_daz.settings import GS
        with open(root_paths_path, encoding="utf-8-sig") as handle:
            GS.readDazPaths(json.load(handle), None, True)

    pre_object_names = {obj.name for obj in bpy.data.objects}
    pre_collection_names = {coll.name for coll in bpy.data.collections}
    master_rig = find_single_armature(bpy.data.objects)

    import_dbz(os.path.abspath(dbz_path))

    new_objects = [obj for obj in bpy.data.objects if obj.name not in pre_object_names]
    new_rigs = [obj for obj in new_objects if obj.type == "ARMATURE"]
    if new_rigs:
        merge_into_master(master_rig, new_rigs)

    # merge_rigs deletes the rig(s) it folded in; re-read from bpy.data.objects
    # so the diff reflects what actually survived.
    new_objects = [obj for obj in bpy.data.objects if obj.name not in pre_object_names]

    duplicates = []
    survivors = []
    for obj in new_objects:
        if obj.type == "ARMATURE":
            print(f"WARNING: leftover armature {obj.name!r} was not merged into "
                  f"{master_rig.name!r}; keeping it in the variant collection.")
            survivors.append(obj)
            continue
        base_name = strip_dup_suffix(obj.name)
        if base_name != obj.name and base_name in pre_object_names:
            duplicates.append(obj)
        else:
            survivors.append(obj)

    duplicate_names = [obj.name for obj in duplicates]
    for obj in duplicates:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Armature):
                bpy.data.armatures.remove(data)

    if not survivors:
        raise RuntimeError(f"No new content found after importing {dbz_path!r}; nothing to author into a collection")

    prefix = {"outfit": "Outfit_", "hair": "Hair_"}[kind]
    target_name = f"{prefix}{collection_name}"
    target = bpy.data.collections.new(target_name)
    # Nest under the character's own collection (not the scene root): worker.py's
    # append_collection() pulls in exactly one named collection via
    # bpy.data.libraries.load, so a variant collection only reaches the render
    # scene -- and only then becomes visible to set_variant_visibility -- if it
    # is a child of that collection.
    character_collection = master_rig.users_collection[0]
    character_collection.children.link(target)
    for obj in survivors:
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        target.objects.link(obj)

    # Clean up now-empty scratch collections the import created (e.g. its own
    # per-file grouping collection) so nothing orphaned is left behind.
    for coll in list(bpy.data.collections):
        if coll.name in pre_collection_names or coll is target or len(coll.objects) != 0:
            continue
        for parent in list(bpy.data.collections) + [bpy.context.scene.collection]:
            if coll.name in parent.children:
                parent.children.unlink(coll)
        bpy.data.collections.remove(coll)

    print("VARIANT_COLLECTION", target.name, "OBJECTS", [obj.name for obj in target.objects])
    print("DUPLICATES_REMOVED", duplicate_names)

    save_path = os.path.abspath(out_path) if out_path else bpy.data.filepath
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    print("SAVED", save_path)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbz", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--kind", required=True, choices=["outfit", "hair"])
    parser.add_argument("--out")
    parser.add_argument("--root-paths")
    args = parser.parse_args(argv)
    author_variant(args.dbz, args.collection_name, args.kind, args.out, args.root_paths)


if __name__ == "__main__":
    main()
