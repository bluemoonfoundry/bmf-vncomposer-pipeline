"""Create a clean character-only stage and render neutral/FACS comparisons."""

import argparse
import math
import os
import sys

import bpy
from mathutils import Vector


def belongs_to_armature(obj, armature):
    parent = obj.parent
    while parent:
        if parent == armature:
            return True
        parent = parent.parent
    return False


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def bounds(objects):
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return minimum, maximum


def area_light(name, location, energy, size, target):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    look_at(obj, target)
    return obj


def make_stage(out_dir):
    scene = bpy.context.scene
    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    character_meshes = [obj for obj in meshes if belongs_to_armature(obj, armature)]
    if not character_meshes:
        raise RuntimeError("No mesh objects were parented to the imported character armature")
    for obj in meshes:
        obj.hide_render = obj not in character_meshes
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True

    minimum, maximum = bounds(character_meshes)
    center = (minimum + maximum) / 2
    extent = maximum - minimum
    source_camera = next((obj for obj in bpy.data.objects if obj.type == "CAMERA"), None)
    direction = (center - source_camera.location).normalized() if source_camera else Vector((0, -1, 0))
    camera_data = bpy.data.cameras.new("Jason_Clean_Camera")
    camera = bpy.data.objects.new("Jason_Clean_Camera", camera_data)
    scene.collection.objects.link(camera)
    camera_data.lens = 58
    # Use an explicit field of view; camera_data.angle is not reliable until
    # Blender has evaluated the newly-created camera datablock.
    distance = max(extent.x, extent.z) * 0.72 / math.tan(math.radians(50.0 / 2))
    # The source DAZ camera is rear-facing for this character; place the clean
    # camera on the opposite side while preserving its viewing axis.
    camera.location = center + direction * distance
    look_at(camera, center + Vector((0, 0, extent.z * 0.05)))
    print("BOUNDS", tuple(round(v, 3) for v in minimum), tuple(round(v, 3) for v in maximum), "EXTENT", tuple(round(v, 3) for v in extent), "CAMERA_DISTANCE", round(distance, 3))
    scene.camera = camera

    area_light("Jason_Clean_Key", center + Vector((-3.5, -4.5, 4.5)), 1000, 4.0, center)
    area_light("Jason_Clean_Fill", center + Vector((3.5, -2.0, 2.0)), 450, 5.0, center)
    area_light("Jason_Clean_Rim", center + Vector((0.0, 3.0, 4.0)), 800, 3.0, center)
    world = scene.world or bpy.data.worlds.new("Jason_Clean_World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.04, 0.04, 0.04, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25

    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    scene.render.filepath = os.path.join(out_dir, "JasonCross_clean_neutral.png")
    os.makedirs(out_dir, exist_ok=True)
    prop = "facs_bs_JawOpen(fin)"
    if prop not in armature.data:
        raise KeyError(prop)
    armature.data[prop] = 0.0
    bpy.ops.render.render(write_still=True)
    armature.data[prop] = 0.7
    scene.render.filepath = os.path.join(out_dir, "JasonCross_clean_jaw_open.png")
    bpy.ops.render.render(write_still=True)
    scene.render.filepath = os.path.join(out_dir, "JasonCross_clean_neutral.png")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "JasonCross_clean_stage.blend"))
    print("CHARACTER_MESHES", [obj.name for obj in character_meshes])
    print("HIDDEN_SCENE_MESHES", [obj.name for obj in meshes if obj not in character_meshes])
    print("CLEAN_STAGE_SAVED", os.path.join(out_dir, "JasonCross_clean_stage.blend"))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    make_stage(os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
