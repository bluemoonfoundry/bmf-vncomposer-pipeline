# Phase 4: Environment/set asset onboarding — design

Tracks: scarecrow-bzy. Blocks scarecrow-8ng (Phase 3).

## Problem

Multi-character scene compositing (Phase 3) needs a way to bring set/prop
geometry into a render scene as a self-contained collection, the same way
character master blends already provide `Jasoncross_v1` and outfit variants
provide `Outfit_WorkerUniform`. Two independent sources of that geometry must
be supported:

- DAZ-sourced sets/props, importable via the same Diffeomorphic
  `easy_import_daz` path `import_daz_artifact.py` already uses for characters.
- Natively Blender-importable assets from other DCCs/engines (Unreal, Unity,
  etc.), in whatever format Blender's own `bpy.ops.import_scene.*` /
  `bpy.ops.wm.*_import` operators support.

Both must normalize into the same collection contract so Phase 3's
`render_scene()` can consume either without caring which pipeline produced it.

## Non-goals

- Feeding an onboarded environment into Phase 3's `render_scene()` — Phase 3
  owns that; this phase stops at producing a correctly-shaped `.blend`.
- Automatic `registry.json` writes. Registry updates for characters/outfits
  have so far been a manual step performed after headless verification
  (see `Onboard Worker Uniform...`, scarecrow-2oz); this phase follows the
  same convention rather than wiring registry writes into the script.
- Broader environment authoring features (variant swapping, LOD, physics) —
  out of scope; only static-geometry onboarding is covered here.

## Collection contract

Every onboarded environment produces a single `Environment_<name>` collection,
linked at the scene root of its own `.blend`, containing:

- All geometry objects imported for that environment (flattened — no nested
  scratch collections left over from the importer).
- Any embedded camera/light objects, each tagged with a custom property
  `obj["scarecrow_env_role"] = "camera"` or `"light"` so Phase 3 can find or
  override them without guessing by object type alone (type is already
  queryable, but the explicit tag keeps the two phases decoupled from how
  Phase 3 chooses to detect them — e.g. if it later needs to distinguish a
  "primary" light from a fill light added separately).
- A world-space bounding box over all mesh objects, stored as
  `collection["scarecrow_bbox_min"]` / `collection["scarecrow_bbox_max"]`
  (each a 3-tuple), computed from `obj.bound_box` transformed by
  `obj.matrix_world`.

## Components

### `blender/collection_utils.py` (new)

Extracted from `author_outfit_variant.py`, which already implements exactly
the diff/flatten/cleanup sequence this phase needs:

- `snapshot_names() -> tuple[set[str], set[str]]` — pre-import object and
  collection name sets.
- `strip_dup_suffix(name: str) -> str` — strips a Blender `.NNN`
  de-duplication suffix.
- `new_objects_since(pre_object_names: set[str]) -> list[bpy.types.Object]`
  — objects present now but not in the snapshot.
- `flatten_into_collection(objects, target: bpy.types.Collection)` — unlinks
  each object from its current collection(s) and links it into `target`.
- `remove_empty_scratch_collections(pre_collection_names: set[str], keep: bpy.types.Collection)`
  — removes collections the importer created that are now empty, skipping
  anything that predates the import or is the target itself.

`author_outfit_variant.py` is updated to import and use these instead of its
inline versions; its duplicate-detection logic (matching stripped names
against pre-existing objects) and rig-merge logic are specific to outfits and
stay where they are.

### `blender/import_environment_asset.py` (new)

CLI, mirroring `import_daz_artifact.py`'s invocation style:

```
blender -b -P blender/import_environment_asset.py -- \
    --source <path-to-asset> --collection-name <name> --out <output.blend> \
    [--root-paths <daz-root-paths.json>]
```

Logic:

1. `snapshot_names()`.
2. Dispatch on `Path(source).suffix.lower()`:
   - `.dbz`, `.duf` → enable `bl_ext.user_default.import_daz`, optionally load
     `--root-paths` (same as `import_daz_artifact.py`), then
     `bpy.ops.daz.easy_import_daz(..., fitMeshes="UNIQUE" (DAZ's import_daz addon
     documents `DBZFILE` as Characters-only, requiring a same-named `.dbz`
     fitting file beside the source `.duf`; `UNIQUE`/`SHARED` are its
     documented Environment-appropriate modes, and `UNIQUE` is used here since
     it doesn't assume the environment's objects share instanced mesh data),
     useMakePosable=False, useHead=False, useUnits=False, useExpressions=False, useVisemes=False, useFacs=False, useFacsdetails=False, useFacsexpr=False, useBody=False, useMergeRigs=False, useMergeMaterials=True, useTransferClothes=False, useTransferFace=False, useTransferGeografts=False, useTransferHD=False)` —
     posing/FACS/rig-merge options are all off since a set/prop has no
     posable figure to fit.
   - `.fbx` → `bpy.ops.import_scene.fbx(filepath=...)`
   - `.gltf`, `.glb` → `bpy.ops.import_scene.gltf(filepath=...)`
   - `.obj` → `bpy.ops.wm.obj_import(filepath=...)`
   - `.usd`, `.usda`, `.usdc`, `.usdz` → `bpy.ops.wm.usd_import(filepath=...)`
   - anything else → `raise ValueError` naming the unsupported extension.
3. `new_objects_since(...)`; if empty, raise (mirrors
   `author_outfit_variant.py`'s "nothing to author into a collection" guard).
4. Create `Environment_<name>` collection, link it to
   `bpy.context.scene.collection`, `flatten_into_collection(new_objects, target)`.
5. `remove_empty_scratch_collections(...)`.
6. Tag cameras/lights among the new objects with `scarecrow_env_role`.
7. Compute and store the bounding box custom properties on the target
   collection.
8. Print a summary line (`ENVIRONMENT_COLLECTION <name> OBJECTS [...]`,
   matching the `VARIANT_COLLECTION` / `IMPORT_RESULT` print conventions
   already used) and `bpy.ops.wm.save_as_mainfile(filepath=<out>)`.

### Tests

`tests/test_import_environment_asset.py` — pure-Python, no live Blender
required:

- Extension-to-handler dispatch table: each supported suffix maps to the
  expected handler name; an unsupported suffix raises.
- Bounding-box computation given a set of fake world-space corner points
  (extracted as a standalone function so it's testable without `bpy`).

Blender-side behavior (the actual import + collection contract) is verified
manually per the plan below, the same way `author_outfit_variant.py` was
verified against a live headless Blender rather than via automated tests.

## Verification plan

1. **DAZ path**: onboard one DAZ-sourced set/prop the user has on disk.
   Confirm `Environment_<name>` exists, is correctly bounded, and (if the
   asset has one) any embedded light is tagged.
2. **Non-DAZ path**: onboard one glTF/FBX file the user supplies. Confirm the
   same collection contract holds.
3. Both `.blend` outputs are left in `artifacts/` for Phase 3 to consume in
   its own two-character/environment verification render later.

## Open questions

None outstanding — all resolved during brainstorming (native format coverage,
test-asset sourcing, camera/light tagging mechanism, bounding-box storage
location).
