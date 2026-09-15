"""Load Diffeomorphic root paths from JSON in Blender background mode."""

import argparse
import json
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    args = parser.parse_args(argv)

    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.import_daz")
    from bl_ext.user_default.import_daz import api

    paths = args.paths
    with open(paths, encoding="utf-8-sig") as handle:
        root_data = json.load(handle)
    from bl_ext.user_default.import_daz.settings import GS

    # The stock operator also calls GS.saveSettings(context), which requires a
    # foreground Blender file-browser context and fails in background mode.
    # readDazPaths is the actual settings update and is safe for headless jobs.
    GS.readDazPaths(root_data, None, True)
    print("LOAD_ROOT_PATHS_RESULT", "FINISHED")
    print("DAZ_ROOT_PATHS", api.get_root_paths())
    expected = "/data/daz 3d/built-in content/daz iray pbrskin/pbrskin.dsf"
    print("PBRSKIN_PATH", api.get_absolute_path(expected))


if __name__ == "__main__":
    main()
