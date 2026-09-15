"""Diagnostic: dump the actual driver network behind the DAZ jaw FACS properties.

Unlike the earlier jaw_bridge_test.py / facs_control_test.py scripts, this does not
guess a property name and probe its downstream effect. It enumerates every driver
whose data_path or variable targets touch "JawOpen", printing the full chain
(raw prop -> final "(fin)" prop -> shape key / bone rotation) so the actual wiring
is known instead of inferred.

Usage:
  blender --background <clean_stage.blend> --python blender/dump_facs_drivers.py -- \
      --match JawOpen
"""

import argparse
import sys

import bpy


def describe_driver(owner_label, fcu):
    d = fcu.driver
    print(f"  [{owner_label}] data_path={fcu.data_path!r} index={fcu.array_index} "
          f"type={d.type} valid={d.is_valid} expr={d.expression!r} mute={fcu.mute}")
    for var in d.variables:
        for trg in var.targets:
            id_name = trg.id.name if trg.id else None
            id_type = getattr(trg, "id_type", None)
            bone_target = getattr(trg, "bone_target", None)
            print(f"      var={var.name!r} vtype={var.type} -> id_type={id_type} "
                  f"id={id_name!r} data_path={trg.data_path!r} bone_target={bone_target!r} "
                  f"transform_type={getattr(trg, 'transform_type', None)}")


def find_drivers(match):
    hits = []
    for block_label, id_block in [
        ("scene", bpy.context.scene),
        *[("armature-data:" + a.name, a) for a in bpy.data.armatures],
        *[("object:" + o.name, o) for o in bpy.data.objects],
        *[("shapekeys:" + m.shape_keys.name, m.shape_keys) for m in bpy.data.meshes if m.shape_keys],
    ]:
        anim = getattr(id_block, "animation_data", None)
        if not anim:
            continue
        for fcu in anim.drivers:
            hay = fcu.data_path
            var_hit = any(
                match.lower() in (trg.data_path or "").lower()
                for var in fcu.driver.variables
                for trg in var.targets
            )
            if match.lower() in hay.lower() or var_hit:
                hits.append((block_label, fcu))
    return hits


def evaluated_readback(rig, prop_names):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_rig = rig.evaluated_get(depsgraph)
    print("\n--- Evaluated-depsgraph readback (rig object + rig.data) ---")
    for name in prop_names:
        raw_obj = rig.get(name, "<absent>")
        raw_data = rig.data.get(name, "<absent>")
        eval_obj = eval_rig.get(name, "<absent>")
        eval_data = eval_rig.data.get(name, "<absent>")
        print(f"  {name!r}: object.raw={raw_obj} object.evaluated={eval_obj} "
              f"data.raw={raw_data} data.evaluated={eval_data}")

    print("\n--- Evaluated shape key values (any key containing 'jaw') ---")
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.data.shape_keys:
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.data
        if not eval_mesh.shape_keys:
            continue
        for key in eval_mesh.shape_keys.key_blocks:
            if "jaw" in key.name.lower():
                stored = obj.data.shape_keys.key_blocks[key.name].value
                print(f"  {obj.name}.{key.name}: stored={stored:.4f} evaluated={key.value:.4f}")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default="JawOpen")
    parser.add_argument("--set-value", type=float, default=None,
                         help="If given, set --prop (exact name) to this value before dumping.")
    parser.add_argument("--prop", default=None,
                         help="Exact raw custom-property name to set, e.g. facs_bs_JawOpenWide")
    args = parser.parse_args(argv)

    rig = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")

    print(f"=== Static driver network matching {args.match!r} ===")
    hits = find_drivers(args.match)
    if not hits:
        print("  NO DRIVERS FOUND matching this term. That itself is diagnostic.")
    for label, fcu in hits:
        describe_driver(label, fcu)

    print(f"\n=== Custom properties on rig object/data matching {args.match!r} ===")
    for owner_label, owner in [("object", rig), ("data", rig.data)]:
        for key in owner.keys():
            if args.match.lower() in key.lower():
                print(f"  {owner_label}[{key!r}] = {owner[key]}")

    if args.set_value is not None:
        name = args.prop
        if not name:
            raise SystemExit("--set-value requires --prop <exact custom property name>")
        print(f"\n=== Setting {name!r} to {args.set_value} ===")
        if name in rig.keys():
            rig[name] = args.set_value
            print(f"  set object[{name!r}] = {args.set_value}")
        if name in rig.data.keys():
            rig.data[name] = args.set_value
            print(f"  set data[{name!r}] = {args.set_value}")

        # Plain Python assignment to a custom ID-property does NOT by itself
        # mark the ID as needing dependency-graph re-evaluation in background
        # mode. update_tag() is the missing step every earlier test skipped.
        rig.update_tag()
        rig.data.update_tag()
        bpy.context.view_layer.update()
        frame = bpy.context.scene.frame_current
        bpy.context.scene.frame_set(frame + 1)
        bpy.context.scene.frame_set(frame)

        prop_names = sorted({
            k for k in list(rig.keys()) + list(rig.data.keys())
            if args.match.lower() in k.lower()
        })
        evaluated_readback(rig, prop_names)


if __name__ == "__main__":
    main()
