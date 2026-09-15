"""Generate docs/facs_controls.json from the live imported rig.

The FACS-side analog of dump_posable_bones.py. Diffeomorphic stores every
user-facing FACS/morph control as a "raw" custom property directly on the
armature OBJECT (not .data) -- e.g. "facs_bs_JawOpenWide". That raw property
feeds a "(fin)" property on the armature DATA via a driver (a straight value,
or a SUM/SCRIPTED combination of several raw/other-(fin) inputs -- see
blender/dump_facs_drivers.py for walking a single control's full chain), and
"(fin)" in turn drives shape keys and/or bone-rotation formulas. Only the raw
object-level properties are meant to be driven directly; "(fin)"/"(rst)"
properties are internal/computed and are deliberately excluded here.

Also excludes internal per-bone driver helper properties (patterns like
"<bone>:Rot:0:01" / "<bone>:Loc:1:01") which are ERC/formula plumbing, not
user-facing expression controls -- see docs/posable_bones.json for bones.

Usage:
  blender --background <master.blend> --python blender/dump_facs_controls.py -- \
      --out docs/facs_controls.json
"""

import argparse
import json
import re
import sys

import bpy

BONE_HELPER_PATTERN = re.compile(r":(Rot|Loc|Scale|ERC):\d+:\d+$")

EMOTION_PRESET_NAMES = {
    "bereft", "bored", "confident", "disgust", "drunk", "excitement", "fear",
    "flirting", "happy", "rage", "scream", "shock", "silly", "surprised",
    "tired", "triumph",
}


def category_for(name):
    lowered = name.lower()
    if lowered.startswith("facs_ctrl_v") or lowered.startswith("ctrlv"):
        return "viseme"
    stem = lowered.split("facs_ctrl_", 1)[-1].split("facs_bs_", 1)[-1]
    if stem in EMOTION_PRESET_NAMES:
        return "emotion_preset"
    if any(t in lowered for t in ("jaw",)):
        return "jaw"
    if any(t in lowered for t in ("brow",)):
        return "brow"
    if any(t in lowered for t in ("eye", "lid", "blink")):
        return "eye"
    if any(t in lowered for t in ("mouth", "lip", "smile", "frown", "corner")):
        return "mouth"
    if any(t in lowered for t in ("cheek", "nose", "nostril", "chin", "nasolabial")):
        return "cheek_nose_chin"
    if lowered.startswith(("facs_bs_", "facs_ctrl_")):
        return "other_facs"
    return "other"


def is_internal_helper(name):
    return bool(BONE_HELPER_PATTERN.search(name)) or name.endswith(("(fin)", "(rst)"))


def drives_shape_key(raw_name, meshes):
    for mesh in meshes:
        if mesh.data.shape_keys and raw_name in mesh.data.shape_keys.key_blocks:
            return True
    return False


def dump_facs_controls(armature, meshes):
    controls = []
    for name in armature.keys():
        if is_internal_helper(name):
            continue
        controls.append({
            "name": name,
            "category": category_for(name),
            "drives_shape_key_directly": drives_shape_key(name, meshes),
        })
    controls.sort(key=lambda c: (c["category"], c["name"]))
    return {
        "armature": armature.name,
        "note": (
            "User-facing FACS/morph control properties, found as raw custom "
            "properties on the armature OBJECT (not .data). Drive via "
            "armature[name] = value THEN armature.update_tag() (a plain "
            "Python write to a custom ID-property does not itself mark the ID "
            "dirty for driver re-evaluation, in background or foreground "
            "Blender). Two similarly-named controls can drive completely "
            "different things (e.g. facs_bs_JawOpen only feeds viseme/lip "
            "sub-formulas; facs_bs_JawOpenWide is what actually rotates the "
            "jaw bone and drives its own shape keys) -- drives_shape_key_directly "
            "tells you which controls have an immediate, visible shape-key "
            "effect versus ones that only contribute to other controls' "
            "combined formulas. See blender/dump_facs_drivers.py to inspect "
            "any single control's full driver chain before relying on it."
        ),
        "control_count": len(controls),
        "controls": controls,
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    out = dump_facs_controls(armature, meshes)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print("WROTE", args.out, out["control_count"], "controls")


if __name__ == "__main__":
    main()
