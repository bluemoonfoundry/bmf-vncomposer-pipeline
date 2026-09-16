"""Shared diff/flatten/cleanup helpers for onboarding scripts that import new
content into an existing .blend and need to collect exactly what's new into a
single named collection (blender/author_outfit_variant.py,
blender/import_environment_asset.py).
"""

import re

import bpy

DUP_SUFFIX_RE = re.compile(r"^(.*)\.\d{3}$")


def strip_dup_suffix(name):
    match = DUP_SUFFIX_RE.match(name)
    return match.group(1) if match else name


def snapshot_names():
    pre_object_names = {obj.name for obj in bpy.data.objects}
    pre_collection_names = {coll.name for coll in bpy.data.collections}
    return pre_object_names, pre_collection_names


def new_objects_since(pre_object_names):
    return [obj for obj in bpy.data.objects if obj.name not in pre_object_names]


def flatten_into_collection(objects, target):
    for obj in objects:
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        target.objects.link(obj)


def remove_empty_scratch_collections(pre_collection_names, keep):
    for coll in list(bpy.data.collections):
        if coll.name in pre_collection_names or coll is keep or len(coll.objects) != 0:
            continue
        for parent in list(bpy.data.collections) + [bpy.context.scene.collection]:
            if coll.name in parent.children:
                parent.children.unlink(coll)
        bpy.data.collections.remove(coll)
