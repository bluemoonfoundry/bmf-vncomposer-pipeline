"""Headless Blender worker for a single VN render.

Run: blender -b scene.blend -P blender/worker.py -- --request request.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from compositor import build_compositor
from collection_utils import strip_dup_suffix


def append_collection(master_blend, collection_name):
    """Append a collection and all of its linked contents from a master cache."""
    with bpy.data.libraries.load(master_blend, link=False) as (source, destination):
        if collection_name not in source.collections:
            raise ValueError(f"Collection {collection_name!r} not found in {master_blend}")
        destination.collections = [collection_name]
    collection = destination.collections[0]
    bpy.context.scene.collection.children.link(collection)
    return collection


def set_variant_visibility(prefixes, active_names):
    for collection in bpy.data.collections:
        if any(collection.name.startswith(prefix) for prefix in prefixes):
            collection.hide_render = not any(collection.name == name or collection.name.endswith(name) for name in active_names)
    for collection in bpy.data.collections:
        if collection.name in active_names:
            collection.hide_render = False


def resolve_armature(character):
    """Resolve the armature belonging to a specific character.

    Multiple characters can be appended into the same scene (see
    append_collection), each inside its own collection named after the
    character. Scanning bpy.data.objects for the first ARMATURE is only
    correct for a single-character scene; once a second character is
    present it silently poses/expresses the wrong rig. `character` should
    be the same identifier passed as request["character"] (the appended
    collection name).
    """
    if not character:
        raise ValueError("character identifier is required to resolve an armature")
    collection = bpy.data.collections.get(character)
    if collection is not None:
        for obj in collection.all_objects:
            if obj.type == "ARMATURE":
                return obj
    # Fallback for armatures not (yet) tracked via a matching collection name.
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and (obj.name == character or obj.name.startswith(f"{character}_")):
            return obj
    return None


def resolve_body_mesh(armature):
    """Find the mesh object Armature-deformed by `armature` -- the
    shrinkwrap collision guard (see apply_pose's IK section) needs this to
    keep an IK target from sinking the hand into the torso. Matching via
    an ARMATURE modifier pointing at this specific object is reliable
    across characters, unlike matching by object name."""
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if any(mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers):
            return obj
    return None


def apply_expression(weights, character):
    """Drive Diffeomorphic FACS/morph controls by their raw property name.

    Diffeomorphic-imported meshes do not take posed shape-key values directly:
    every FACS/morph shape key is driven by an armature custom property (the
    "raw" control, e.g. "facs_bs_JawOpenWide"), which itself feeds a "(fin)"
    property via a scripted driver, which in turn drives the shape key value
    and any coupled bone rotations (see blender/dump_facs_drivers.py for how
    to inspect this chain for a given control). Writing shape_key.value
    directly gets silently overwritten by that driver network on the next
    dependency-graph evaluation.

    A second, easy-to-miss requirement: a plain Python write to a custom
    ID-property (armature["prop"] = x) does not itself mark the ID as dirty
    for driver re-evaluation, in background or foreground Blender. Each
    changed object/data block needs an explicit update_tag() call.
    """
    armature = resolve_armature(character)
    if armature is None:
        print(f"WARNING: no armature found for character {character!r}; expression weights ignored.")
        return
    touched = set()
    for name, weight in weights.items():
        value = max(0.0, min(1.0, float(weight)))
        if name in armature.keys():
            armature[name] = value
            touched.add(armature)
        elif name in armature.data.keys():
            armature.data[name] = value
            touched.add(armature.data)
        else:
            print(f"WARNING: FACS/morph control {name!r} not found as a custom "
                  f"property on armature {armature.name!r}; expression weight ignored. "
                  f"Use blender/dump_facs_drivers.py --match <term> to find the correct name.")
    for id_block in touched:
        id_block.update_tag()
    if touched:
        bpy.context.view_layer.update()


def apply_pose(pose, character):
    """Pose the imported Diffeomorphic rig directly in Blender.

    DAZ Studio is only used to source the base character once; all posing
    happens here so it can be driven by natural language. Bone rotations are
    applied to the base-named ("posable") pose bones, e.g. "r_upperarm", not
    their hidden "(drv)" counterparts -- those carry an additive corrective
    layer (mix_mode='BEFORE_FULL') and are combined automatically, not
    overridden, so direct rotation_euler writes are safe. See
    blender/dump_facs_drivers.py and docs/posable_bones.json for how to
    verify a bone/property before driving it.
    """
    if not pose:
        return
    armature = resolve_armature(character)
    if armature is None:
        print(f"WARNING: no armature found for character {character!r}; pose ignored.")
        return

    for bone_name, euler in pose.get("bone_rotations", {}).items():
        bone = armature.pose.bones.get(bone_name)
        if bone is None:
            print(f"WARNING: pose bone {bone_name!r} not found on armature {armature.name!r}; "
                  f"rotation ignored. See docs/posable_bones.json for valid names.")
            continue
        # Each imported bone keeps DAZ's own native Euler axis order (e.g. "YZX"),
        # not Blender's "XYZ" default. Forcing rotation_mode to XYZ would silently
        # reinterpret the (x, y, z) triplet in the wrong axis order for any
        # multi-axis rotation. docs/posable_bones.json records each bone's mode.
        if bone.rotation_mode == "QUATERNION" or bone.rotation_mode.startswith("AXIS"):
            bone.rotation_mode = "YZX"
        bone.rotation_euler = Vector(euler)

    root_location = pose.get("root_location")
    if root_location is not None:
        root_bone = armature.pose.bones.get("hip")
        if root_bone is not None:
            root_bone.location = Vector(root_location)

    pole_targets = pose.get("pole_targets", {})
    ik_target_bones = pose.get("ik_targets", {})
    for bone_name in pole_targets:
        if bone_name not in ik_target_bones:
            print(f"WARNING: pole target for bone {bone_name!r} has no matching ik_targets "
                  f"entry; ignored.")

    for bone_name, coordinates in ik_target_bones.items():
        if bone_name not in armature.pose.bones:
            print(f"WARNING: IK target bone {bone_name!r} not found on armature {armature.name!r}; ignored.")
            continue
        target = bpy.data.objects.get(f"IK_Target_{bone_name}") or bpy.data.objects.new(f"IK_Target_{bone_name}", None)
        if not target.users_collection:
            bpy.context.scene.collection.objects.link(target)
        target.location = Vector(coordinates)
        bone = armature.pose.bones[bone_name]
        constraint = next((item for item in bone.constraints if item.name == "Scarecrow IK"), None) or bone.constraints.new("IK")
        constraint.name = "Scarecrow IK"
        constraint.target = target
        constraint.chain_count = 2

        body_mesh = resolve_body_mesh(armature)
        if body_mesh is not None:
            guard = next((item for item in bone.constraints if item.name == "Scarecrow Surface Guard"), None) \
                or bone.constraints.new("SHRINKWRAP")
            guard.name = "Scarecrow Surface Guard"
            guard.target = body_mesh
            guard.shrinkwrap_type = "NEAREST_SURFACE"
            guard.distance = 0.03
            guard.influence = 0.8
            # IK must resolve the base position first; the guard nudges it off
            # the surface afterward, so it must sit after "Scarecrow IK" in
            # the constraint stack -- constraints.new() already appends to
            # the end, so no explicit reordering is needed as long as this
            # block stays after the IK constraint is created above.

        pole_coordinates = pole_targets.get(bone_name)
        if pole_coordinates is not None:
            pole = bpy.data.objects.get(f"IK_Pole_{bone_name}") or bpy.data.objects.new(f"IK_Pole_{bone_name}", None)
            if not pole.users_collection:
                bpy.context.scene.collection.objects.link(pole)
            pole.location = Vector(pole_coordinates)
            constraint.pole_target = pole
        else:
            constraint.pole_target = None

    look_at_target = pose.get("look_at_target")
    look_bone_names = ["head", "l_eye", "r_eye"]
    if look_at_target is not None:
        target = bpy.data.objects.get("LookAt_Target") or bpy.data.objects.new("LookAt_Target", None)
        if not target.users_collection:
            bpy.context.scene.collection.objects.link(target)
        target.location = Vector(look_at_target)
        # TRACK_Z verified empirically (blender/dump_facs_drivers.py-style probe
        # render, see artifacts/head_track_test/) against this rig's actual bone
        # orientation -- do not assume a track axis for a differently-oriented rig.
        for bone_name in look_bone_names:
            bone = armature.pose.bones.get(bone_name)
            if bone is None:
                continue
            constraint = next((item for item in bone.constraints if item.name == "Scarecrow LookAt"), None) \
                or bone.constraints.new("DAMPED_TRACK")
            constraint.name = "Scarecrow LookAt"
            constraint.target = target
            constraint.track_axis = "TRACK_Z"
    else:
        for bone_name in look_bone_names:
            bone = armature.pose.bones.get(bone_name)
            if bone is None:
                continue
            constraint = next((item for item in bone.constraints if item.name == "Scarecrow LookAt"), None)
            if constraint is not None:
                bone.constraints.remove(constraint)

    bpy.context.view_layer.update()


def configure_camera(camera_data):
    camera = bpy.context.scene.camera or bpy.data.objects.new("Scarecrow_Camera", bpy.data.cameras.new("Scarecrow_Camera"))
    if camera.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.data.type = camera_data.get("projection", "PERSP")
    camera.data.lens = float(camera_data.get("focal_length_mm", 50.0))
    camera.location = Vector(camera_data.get("location", [0.0, -8.0, 1.6]))
    camera.rotation_euler = Vector(camera_data.get("rotation_euler", [math.pi / 2, 0.0, 0.0]))


def configure_lighting(lighting):
    world = bpy.context.scene.world or bpy.data.worlds.new("Scarecrow World")
    bpy.context.scene.world = world
    if world.node_tree is None:
        # World.use_nodes is deprecated (worlds have had mandatory node trees
        # since Blender 4.0) and is slated for removal in 6.0; only touch it
        # as a fallback for a world that genuinely lacks a node tree.
        world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*lighting.get("ambient_rgb", [0.5] * 3), 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    light_data = bpy.data.lights.get("Scarecrow_Key") or bpy.data.lights.new("Scarecrow_Key", "SUN")
    light = bpy.data.objects.get("Scarecrow_Key") or bpy.data.objects.new("Scarecrow_Key", light_data)
    if light.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(light)
    light.rotation_euler = Vector(lighting.get("direction", [0.0, -1.0, -1.0])).to_track_quat("-Z", "Y").to_euler()
    light.data.energy = float(lighting.get("intensity", 1.0))
    light.data.color = lighting.get("color_rgb", [1.0, 1.0, 1.0])


def add_shadow_catcher():
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "Scarecrow_Shadow_Catcher"
    if hasattr(plane, "is_shadow_catcher"):
        plane.is_shadow_catcher = True


def render(request):
    append_collection(request["master_blend"], request["character"])
    active = [name for name in (request.get("outfit"), request.get("hair")) if name]
    set_variant_visibility(["Outfit_", "Hair_"], active)
    character = request["character"]
    apply_expression(request.get("expression", {}).get("weights", {}), character)
    apply_pose(request.get("pose"), character)
    configure_camera(request.get("camera", {}))
    configure_lighting(request.get("lighting", {}))
    add_shadow_catcher()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(request.get("render_samples", 64))
    scene.render.film_transparent = bool(request.get("transparent", True))
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = request["output_path"]
    build_compositor(request.get("background_path"), request["output_path"])
    bpy.ops.wm.save_as_mainfile(filepath=request["output_path"] + ".blend")
    bpy.ops.render.render(write_still=True)


def character_variant_collections(character_collection):
    """All collections nested under a character's own appended collection (its
    Outfit_/Hair_ variants -- see author_outfit_variant.py), scoped to that one
    character so a second character's variants can never be touched."""
    found = []
    stack = list(character_collection.children)
    while stack:
        child = stack.pop()
        found.append(child)
        stack.extend(child.children)
    return found


def set_character_variant_visibility(character_collection, prefixes, active_names):
    """Like set_variant_visibility (used by the single-character render() path),
    but scoped to one character's own collection tree. Blender auto-suffixes a
    variant collection's name (e.g. "Outfit_Casual.001") when two characters
    share a master_blend and therefore an identically-named variant, so names
    are compared with the dup suffix stripped rather than by exact match.
    """
    for collection in character_variant_collections(character_collection):
        if not any(collection.name.startswith(prefix) for prefix in prefixes):
            continue
        stripped = strip_dup_suffix(collection.name)
        is_active = any(stripped == name or stripped.endswith(name) for name in active_names)
        collection.hide_render = not is_active


def place_character(collection, placement):
    """Apply a CharacterPlacement's world-space transform to the top-level
    (parent-less) objects of an appended character collection -- normally just
    the armature, with meshes/props following it as parented children."""
    members = set(collection.all_objects)
    roots = [obj for obj in collection.all_objects if obj.parent is None or obj.parent not in members]
    location = Vector(placement.get("location", [0.0, 0.0, 0.0]))
    rotation = Vector(placement.get("rotation_euler", [0.0, 0.0, 0.0]))
    scale = float(placement.get("scale", 1.0))
    for obj in roots:
        obj.location = location
        obj.rotation_euler = rotation
        obj.scale = (scale, scale, scale)


def append_scene_character(placement):
    """Append and configure one CharacterPlacement, isolated in its own
    collection so its armature, morphs, and materials cannot cross-talk with
    any other character already placed in the scene (scarecrow-tnf)."""
    character = placement["character"]
    if bpy.data.collections.get(character) is not None:
        raise ValueError(
            f"Character identifier {character!r} is already used by another "
            "placement in this scene; each CharacterPlacement.character must "
            "be unique so resolve_armature() cannot ambiguously match it."
        )
    collection = append_collection(placement["master_blend"], character)
    active = [name for name in (placement.get("outfit"), placement.get("hair")) if name]
    set_character_variant_visibility(collection, ["Outfit_", "Hair_"], active)
    place_character(collection, placement)
    apply_expression(placement.get("expression", {}).get("weights", {}), character)
    apply_pose(placement.get("pose"), character)
    return collection


def resolve_environment_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        raise ValueError(
            f"Environment collection {name!r} not found; it must already be "
            "present in the base .blend (see blender/import_environment_asset.py)."
        )
    collection.hide_render = False
    return collection


def build_scene_background(background, output_path):
    """Wire up the requested BackgroundSpec. image_plate reuses the same
    AlphaOver compositor as the single-character path -- it composites
    whatever the Render Layers node captured (all placed characters) over one
    shared plate, so no per-character change is needed there. 3d_environment
    skips 2D compositing entirely: the pre-imported environment collection
    supplies the background directly in 3D, and the render is left opaque.
    """
    mode = background.get("mode", "image_plate")
    if mode == "3d_environment":
        resolve_environment_collection(background["environment_collection"])
        return False
    build_compositor(background.get("background_path"), output_path)
    return bool(background.get("transparent", True))


def render_scene(scene_request):
    """Render a multi-character SceneRequest (scarecrow_pipeline/schemas.py).

    Loads and places every CharacterPlacement into its own isolated
    collection, applies the scene's shared camera/lighting, and handles
    either BackgroundSpec mode. Alongside, not replacing, render() above.
    """
    for placement in scene_request["characters"]:
        append_scene_character(placement)
    configure_camera(scene_request.get("camera", {}))
    configure_lighting(scene_request.get("lighting", {}))
    background = scene_request.get("background") or {"mode": "image_plate"}
    if background.get("mode", "image_plate") != "3d_environment":
        add_shadow_catcher()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(scene_request.get("render_samples", 64))
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = scene_request["output_path"]
    scene.render.film_transparent = build_scene_background(background, scene_request["output_path"])
    bpy.ops.wm.save_as_mainfile(filepath=scene_request["output_path"] + ".blend")
    bpy.ops.render.render(write_still=True)


def main():
    # generate_sprite_set.py launches this against Blender's default startup
    # scene (see its module docstring), which otherwise leaves the default
    # Cube/Camera/Light sitting in the render -- render()/configure_camera()/
    # configure_lighting() build their own named objects and never reuse or
    # remove those defaults.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    with open(args.request, encoding="utf-8") as handle:
        request = json.load(handle)
    if "characters" in request:
        render_scene(request)
    else:
        render(request)


if __name__ == "__main__":
    main()
