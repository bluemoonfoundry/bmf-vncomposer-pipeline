"""Blender compositor graph construction. Execute inside Blender's Python."""

import os

import bpy


def build_compositor(background_path=None, output_path=None, use_depth_fog=False):
    scene = bpy.context.scene
    # Blender 5.x replaced Scene.node_tree (compositor) with a shared
    # Scene.compositing_node_group pointing at a CompositorNodeTree datablock
    # -- Scene.use_nodes still exists but no longer creates/exposes node_tree.
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("Compositor Nodegroup", "CompositorNodeTree")
        scene.compositing_node_group = tree
    tree.nodes.clear()
    render = tree.nodes.new("CompositorNodeRLayers")
    render.name = "Character Render"
    # CompositorNodeComposite ("Composite" node -- writes to Scene.render
    # result / the viewer, not to disk) is only valid in the legacy top-level
    # Scene.node_tree, not in a CompositorNodeTree node group -- undefined
    # error if instantiated here. Not needed: the actual PNG write happens via
    # CompositorNodeOutputFile below (compositor path) and separately via
    # Scene.render.filepath (worker.py's bpy.ops.render.render call).
    # Socket order/names on CompositorNodeAlphaOver changed in Blender's newer
    # compositor versions (now Background, Foreground, Factor, Type, Straight
    # Alpha -- not the old Fac/Image1/Image2 positional layout) -- address by
    # name, not index, so this doesn't silently wire the wrong sockets again.
    alpha = tree.nodes.new("CompositorNodeAlphaOver")
    alpha.inputs["Factor"].default_value = 1.0
    tree.links.new(render.outputs["Image"], alpha.inputs["Foreground"])

    if background_path:
        background = tree.nodes.new("CompositorNodeImage")
        background.name = "Background Plate"
        background.image = bpy.data.images.load(background_path, check_existing=True)
        tree.links.new(background.outputs["Image"], alpha.inputs["Background"])
    else:
        alpha.inputs["Background"].default_value = (0.0, 0.0, 0.0, 1.0)

    final_node = alpha
    if use_depth_fog and "Depth" in render.outputs:
        fog = tree.nodes.new("CompositorNodeFogGlow")
        fog.quality = "HIGH"
        fog.threshold = 1.0
        tree.links.new(final_node.outputs[0], fog.inputs[0])
        final_node = fog

    if output_path:
        # CompositorNodeOutputFile's single-slot API was renamed: base_path ->
        # directory, file_slots[0].path -> file_name (file_slots/file_slots[i].path
        # no longer exist -- file_output_items is only for adding extra named
        # slots on top of the node's one default input).
        file_output = tree.nodes.new("CompositorNodeOutputFile")
        file_output.name = "Final PNG"
        file_output.directory = os.path.dirname(output_path) or "."
        file_output.file_name = os.path.splitext(os.path.basename(output_path))[0]
        # format.media_type defaults to MULTI_LAYER_IMAGE, which only allows
        # OPEN_EXR_MULTILAYER for file_format -- must switch to IMAGE first
        # to unlock PNG.
        file_output.format.media_type = "IMAGE"
        file_output.format.file_format = "PNG"
        file_output.format.color_mode = "RGBA"
        tree.links.new(final_node.outputs[0], file_output.inputs[0])
    return tree
