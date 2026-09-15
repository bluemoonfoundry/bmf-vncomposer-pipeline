"""Generate docs/posable_bones.json from the live imported rig.

Walks every posable (non-"(drv)") pose bone on the armature and records its
category, native rotation_mode, whether it carries an additive "(drv)"
corrective layer, and any anatomical rotation limits Diffeomorphic imported
as a LIMIT_ROTATION constraint. This is the machine-readable vocabulary a
future natural-language pose layer targets -- regenerate it whenever a new
character is onboarded rather than hand-editing the JSON.

Usage:
  blender --background <master.blend> --python blender/dump_posable_bones.py -- \
      --out docs/posable_bones.json
"""

import argparse
import json
import sys

import bpy

CATEGORY_TERMS = {
    "face": ["eye", "brow", "lip", "cheek", "jaw", "nose", "nostril", "chin", "ear",
             "mouth", "tongue", "teeth", "lid", "smile", "frown", "nasolabial", "philtrum", "corner"],
    "hand": ["thumb", "index", "mid", "ring", "pinky", "carpal", "hand"],
    "foot": ["toe", "foot", "heel", "metatarsal"],
    "leg": ["thigh", "shin", "calf", "hip", "pelvis"],
    "arm": ["shldr", "shoulder", "upperarm", "forearm", "arm", "clavicle"],
    "spine": ["spine", "chest", "abdomen", "neck", "head"],
}


def category_for(name):
    lowered = name.lower()
    for category, terms in CATEGORY_TERMS.items():
        if any(term in lowered for term in terms):
            return category
    return "other"


def rotation_limits_for(pose_bone):
    limit_cns = next((c for c in pose_bone.constraints if c.type == "LIMIT_ROTATION"), None)
    if limit_cns is None:
        return None
    return {
        "x": [round(limit_cns.min_x, 4), round(limit_cns.max_x, 4)] if limit_cns.use_limit_x else None,
        "y": [round(limit_cns.min_y, 4), round(limit_cns.max_y, 4)] if limit_cns.use_limit_y else None,
        "z": [round(limit_cns.min_z, 4), round(limit_cns.max_z, 4)] if limit_cns.use_limit_z else None,
    }


def dump_posable_bones(armature):
    bones = []
    for pose_bone in armature.pose.bones:
        if pose_bone.name.endswith("(drv)"):
            continue
        corrective = any(
            c.type == "COPY_TRANSFORMS" and c.subtarget.endswith("(drv)")
            for c in pose_bone.constraints
        )
        bones.append({
            "name": pose_bone.name,
            "parent": pose_bone.parent.name if pose_bone.parent else None,
            "category": category_for(pose_bone.name),
            "rotation_mode": pose_bone.rotation_mode,
            "has_corrective_layer": corrective,
            "rotation_limits_radians": rotation_limits_for(pose_bone),
        })
    bones.sort(key=lambda b: (b["category"], b["name"]))
    limited_count = sum(1 for b in bones if b["rotation_limits_radians"] is not None)
    return {
        "armature": armature.name,
        "note": (
            "Posable (non-'(drv)') pose bones on the Diffeomorphic-imported rig. "
            "Drive these via pose.bones[name].rotation_euler using EACH BONE'S OWN "
            "rotation_mode (do not force XYZ order -- DAZ's native per-bone axis order "
            "matters for multi-axis rotations). rotation_limits_radians, where present, "
            "come from a LIMIT_ROTATION constraint already on the bone (imported from DAZ's "
            "own joint limits) and are enforced automatically by Blender at evaluation time; "
            "null per-axis or null overall means unconstrained -- keep natural-language-driven "
            "rotations near documented anatomical ranges anyway to avoid clamped/ugly poses. "
            "has_corrective_layer=true bones additionally carry an additive '(drv)' JCM layer "
            "(mix_mode=BEFORE_FULL) that combines automatically -- safe to pose directly."
        ),
        "bone_count": len(bones),
        "bones_with_rotation_limits": limited_count,
        "bones": bones,
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    out = dump_posable_bones(armature)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print("WROTE", args.out, out["bone_count"], "bones,", out["bones_with_rotation_limits"], "with rotation limits")


if __name__ == "__main__":
    main()
