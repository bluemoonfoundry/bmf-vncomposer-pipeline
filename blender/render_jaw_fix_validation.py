"""Validate the real fix for the jaw-open FACS control.

Root cause (found via blender/dump_facs_drivers.py against the untouched
Diffeomorphic driver network):
  1. Earlier tests drove "facs_bs_JawOpen", which only feeds viseme/lip
     sub-formulas and has no shape key or bone-rotation output of its own.
     The channel that actually rotates the mandible (lowerjaw:Rot:0:01) and
     drives the three "facs_bs_JawOpenWide" mesh shape keys is a DIFFERENT
     property: "facs_bs_JawOpenWide".
  2. Plain Python assignment to a custom ID-property (armature["prop"] = x)
     does not itself mark the ID as needing dependency-graph re-evaluation.
     update_tag() must be called before the driver network will re-evaluate,
     in both background and foreground Blender.

This script drives only "facs_bs_JawOpenWide" (the real source property),
calls update_tag(), and renders neutral vs. open with every imported driver
left intact -- no driver removal, no Rigify conversion.
"""

import argparse
import math
import os
import sys

import bpy
from mathutils import Vector


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def set_jaw_open_wide(armature, value):
    armature["facs_bs_JawOpenWide"] = value
    armature.update_tag()
    armature.data.update_tag()
    bpy.context.view_layer.update()


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    scene = bpy.context.scene
    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_render]

    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    height = maximum.z - minimum.z
    upper = [v for v in corners if v.z >= minimum.z + height * 0.52]
    target_min = Vector((min(v.x for v in upper), min(v.y for v in upper), min(v.z for v in upper)))
    target_max = Vector((max(v.x for v in upper), max(v.y for v in upper), max(v.z for v in upper)))
    target = (target_min + target_max) / 2
    target.z += 0.05

    source = scene.camera
    direction = (target - source.location).normalized()
    width = max(target_max.x - target_min.x, target_max.y - target_min.y)
    distance = width * 0.42 / math.tan(math.radians(42.0 / 2))
    camera_data = bpy.data.cameras.new("Jason_JawFix_Camera")
    camera = bpy.data.objects.new("Jason_JawFix_Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = target - direction * distance
    camera_data.lens = 58
    look_at(camera, target)
    scene.camera = camera

    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True

    def jaw_key_values():
        out = []
        for obj in meshes:
            if not obj.data.shape_keys:
                continue
            for key in obj.data.shape_keys.key_blocks:
                if key.name == "facs_bs_JawOpenWide" and key.value > 0.001:
                    out.append((obj.name, round(key.value, 3)))
        return out

    set_jaw_open_wide(armature, 0.0)
    print("NEUTRAL fin=", armature.data.get("facs_bs_JawOpenWide(fin)"), "jaw_keys=", jaw_key_values())
    scene.render.filepath = os.path.join(args.out_dir, "JasonCross_jawfix_neutral.png")
    bpy.ops.render.render(write_still=True)

    set_jaw_open_wide(armature, 1.0)
    print("OPEN fin=", armature.data.get("facs_bs_JawOpenWide(fin)"), "jaw_keys=", jaw_key_values())
    scene.render.filepath = os.path.join(args.out_dir, "JasonCross_jawfix_open.png")
    bpy.ops.render.render(write_still=True)

    print("JAWFIX_RENDERED", args.out_dir)


if __name__ == "__main__":
    main()
