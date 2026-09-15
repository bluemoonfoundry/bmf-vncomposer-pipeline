"""Import a Diffeomorphic DBZ artifact in Blender background mode."""

import argparse
import json
import os
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbz", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--root-paths")
    args = parser.parse_args(argv)

    from bl_ext.user_default import import_daz

    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.import_daz")
    from bl_ext.user_default.import_daz import api
    if args.root_paths:
        from bl_ext.user_default.import_daz.settings import GS
        with open(args.root_paths, encoding="utf-8-sig") as handle:
            GS.readDazPaths(json.load(handle), None, True)
        print("ROOT_PATHS_LOADED", api.get_absolute_path("/data/daz 3d/built-in content/daz iray pbrskin/pbrskin.dsf"))

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
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend))


if __name__ == "__main__":
    main()
