# Phase 4: Environment/set asset onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic import dispatcher that onboards environment/set assets — both DAZ-sourced and natively Blender-importable (glTF/FBX/OBJ/USD) — into a normalized `Environment_<name>` collection contract, so Phase 3's multi-character scene compositing can consume either without caring which pipeline produced it.

**Architecture:** Extract the diff/flatten/cleanup collection logic already proven in `blender/author_outfit_variant.py` into a shared `blender/collection_utils.py`. Put the bpy-free logic (extension-to-importer dispatch, bounding-box math) in a new `scarecrow_pipeline/environment_asset.py` module so it's unit-testable with plain pytest (no live Blender). Wire both into a new `blender/import_environment_asset.py` CLI script that mirrors the existing `import_daz_artifact.py` / `author_outfit_variant.py` invocation style.

**Tech Stack:** Python 3.12/3.13, Blender's `bpy` (headless, `-b -P` mode), pydantic (unused directly by this feature but present in the repo), pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-phase4-environment-onboarding-design.md`

## Global Constraints

- `scarecrow_pipeline/` modules must not import `bpy` — they're plain Python, imported both by tests and (optionally) by `blender/*.py` scripts running inside Blender.
- `blender/*.py` scripts are never covered by automated tests in this repo (no `bpy` available under pytest) — verification for bpy-touching code is manual, headless, against real assets, matching how `author_outfit_variant.py` and `import_daz_artifact.py` were verified.
- Follow existing print-diagnostic conventions (`IMPORT_RESULT`, `VARIANT_COLLECTION ... OBJECTS [...]`) for any new script's stdout, so log-scraping stays consistent.
- Registry (`artifacts/registry.json`) is updated by hand after verification, not by the onboarding script — do not add `Registry.upsert_environment()` calls to `import_environment_asset.py`.
- `EnvironmentRecord` in `scarecrow_pipeline/registry.py` already exists and needs no schema changes.

---

### Task 1: Pure-Python dispatch + bounding-box logic

**Files:**
- Create: `scarecrow_pipeline/environment_asset.py`
- Test: `tests/test_environment_asset.py`

**Interfaces:**
- Consumes: nothing (no dependency on earlier tasks).
- Produces:
  - `SUFFIX_HANDLERS: dict[str, str]` — maps a lowercase file extension (including the leading dot, e.g. `".fbx"`) to a handler tag: `"daz"` for `.dbz`/`.duf`, `"fbx"` for `.fbx`, `"gltf"` for `.gltf`/`.glb`, `"obj"` for `.obj`, `"usd"` for `.usd`/`.usda`/`.usdc`/`.usdz`.
  - `resolve_handler(source_path: str) -> str` — returns the handler tag for `source_path`'s suffix; raises `ValueError(f"Unsupported environment asset extension: {suffix!r}")` for anything else (case-insensitive on the suffix).
  - `compute_bounding_box(corners: list[tuple[float, float, float]]) -> tuple[list[float], list[float]]` — given a flat list of world-space `(x, y, z)` points, returns `(min_xyz, max_xyz)` each as a 3-element `list[float]`. Raises `ValueError("compute_bounding_box requires at least one point")` on an empty list.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_environment_asset.py
import pytest

from scarecrow_pipeline.environment_asset import compute_bounding_box, resolve_handler


@pytest.mark.parametrize("source_path,expected", [
    ("set.dbz", "daz"),
    ("prop.duf", "daz"),
    ("scene.FBX", "fbx"),
    ("scene.gltf", "gltf"),
    ("scene.glb", "gltf"),
    ("mesh.obj", "obj"),
    ("stage.usd", "usd"),
    ("stage.usda", "usd"),
    ("stage.usdc", "usd"),
    ("stage.usdz", "usd"),
])
def test_resolve_handler_maps_known_extensions(source_path, expected):
    assert resolve_handler(source_path) == expected


def test_resolve_handler_rejects_unknown_extension():
    with pytest.raises(ValueError, match="Unsupported environment asset extension"):
        resolve_handler("model.blend")


def test_compute_bounding_box_over_multiple_points():
    corners = [(-1.0, 0.0, 2.0), (3.0, -5.0, 2.0), (0.0, 4.0, -1.0)]
    bbox_min, bbox_max = compute_bounding_box(corners)
    assert bbox_min == [-1.0, -5.0, -1.0]
    assert bbox_max == [3.0, 4.0, 2.0]


def test_compute_bounding_box_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one point"):
        compute_bounding_box([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_environment_asset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scarecrow_pipeline.environment_asset'`

- [ ] **Step 3: Write the implementation**

```python
# scarecrow_pipeline/environment_asset.py
"""Pure-Python (no bpy) helpers for onboarding environment/set assets.

Shared by blender/import_environment_asset.py, kept bpy-free so the
extension-dispatch table and bounding-box math are unit-testable without a
live Blender process.
"""

from __future__ import annotations

from pathlib import Path

SUFFIX_HANDLERS: dict[str, str] = {
    ".dbz": "daz",
    ".duf": "daz",
    ".fbx": "fbx",
    ".gltf": "gltf",
    ".glb": "gltf",
    ".obj": "obj",
    ".usd": "usd",
    ".usda": "usd",
    ".usdc": "usd",
    ".usdz": "usd",
}


def resolve_handler(source_path: str) -> str:
    suffix = Path(source_path).suffix.lower()
    handler = SUFFIX_HANDLERS.get(suffix)
    if handler is None:
        raise ValueError(f"Unsupported environment asset extension: {suffix!r}")
    return handler


def compute_bounding_box(corners: list[tuple[float, float, float]]) -> tuple[list[float], list[float]]:
    if not corners:
        raise ValueError("compute_bounding_box requires at least one point")
    xs, ys, zs = zip(*corners)
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_environment_asset.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scarecrow_pipeline/environment_asset.py tests/test_environment_asset.py
git commit -m "Add bpy-free dispatch/bbox helpers for environment onboarding (scarecrow-bzy)"
```

---

### Task 2: Extract shared collection helpers from `author_outfit_variant.py`

**Files:**
- Create: `blender/collection_utils.py`
- Modify: `blender/author_outfit_variant.py`

**Interfaces:**
- Consumes: nothing new (operates purely on `bpy.data`/`bpy.context`, same as the code it's extracted from).
- Produces (for Task 4 to consume):
  - `snapshot_names() -> tuple[set[str], set[str]]` — returns `(pre_object_names, pre_collection_names)`.
  - `strip_dup_suffix(name: str) -> str` — strips a trailing Blender `.NNN` de-dup suffix if present, else returns `name` unchanged.
  - `new_objects_since(pre_object_names: set[str]) -> list["bpy.types.Object"]` — objects in `bpy.data.objects` not present in `pre_object_names`.
  - `flatten_into_collection(objects: list["bpy.types.Object"], target: "bpy.types.Collection") -> None` — for each object, unlinks it from every collection it currently belongs to and links it into `target`.
  - `remove_empty_scratch_collections(pre_collection_names: set[str], keep: "bpy.types.Collection") -> None` — removes any collection not in `pre_collection_names`, not `keep`, and with zero objects, unlinking it from any parent collection or the scene root first.

There is no automated test for this file (no `bpy` under pytest, consistent with every other file in `blender/`). Correctness is established by Task 3's behavior being unchanged, verified manually per this repo's existing convention.

- [ ] **Step 1: Read the current inline implementations**

Re-read `blender/author_outfit_variant.py` lines 37-42 (`DUP_SUFFIX_RE`, `strip_dup_suffix`), lines 108-109 (snapshot), lines 114/121 (`new_objects` diff), lines 160-163 (flatten loop), and lines 166-173 (scratch-collection cleanup) — these are the five blocks being extracted verbatim.

- [ ] **Step 2: Write `blender/collection_utils.py`**

```python
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
```

- [ ] **Step 3: Update `blender/author_outfit_variant.py` to use the shared helpers**

Remove the now-duplicated `DUP_SUFFIX_RE` and `strip_dup_suffix` definitions (lines 37-42) and add the import:

```python
from collection_utils import (
    flatten_into_collection,
    new_objects_since,
    remove_empty_scratch_collections,
    snapshot_names,
    strip_dup_suffix,
)
```

placed alongside the existing `import bpy`, matching how `blender/worker.py` imports its sibling `compositor` module (`SCRIPT_DIR` path-insert pattern) — add the same `SCRIPT_DIR` sys.path setup to `author_outfit_variant.py` since it currently has none:

```python
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
```

Replace this block in `author_variant()`:

```python
    pre_object_names = {obj.name for obj in bpy.data.objects}
    pre_collection_names = {coll.name for coll in bpy.data.collections}
    master_rig = find_single_armature(bpy.data.objects)
```

with:

```python
    pre_object_names, pre_collection_names = snapshot_names()
    master_rig = find_single_armature(bpy.data.objects)
```

Replace both occurrences of:

```python
    new_objects = [obj for obj in bpy.data.objects if obj.name not in pre_object_names]
```

with:

```python
    new_objects = new_objects_since(pre_object_names)
```

Replace:

```python
    for obj in survivors:
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        target.objects.link(obj)
```

with:

```python
    flatten_into_collection(survivors, target)
```

Replace:

```python
    for coll in list(bpy.data.collections):
        if coll.name in pre_collection_names or coll is target or len(coll.objects) != 0:
            continue
        for parent in list(bpy.data.collections) + [bpy.context.scene.collection]:
            if coll.name in parent.children:
                parent.children.unlink(coll)
        bpy.data.collections.remove(coll)
```

with:

```python
    remove_empty_scratch_collections(pre_collection_names, keep=target)
```

- [ ] **Step 4: Verify no syntax errors**

Run: `python -m py_compile blender/collection_utils.py blender/author_outfit_variant.py`
Expected: exits 0, no output (this only checks syntax — `bpy` isn't importable outside Blender, so this can't execute the module; full behavioral verification happens in Task 5's headless run).

- [ ] **Step 5: Commit**

```bash
git add blender/collection_utils.py blender/author_outfit_variant.py
git commit -m "Extract shared collection diff/flatten helpers into collection_utils.py (scarecrow-bzy)"
```

---

### Task 3: `import_environment_asset.py` dispatcher

**Files:**
- Create: `blender/import_environment_asset.py`

**Interfaces:**
- Consumes:
  - `scarecrow_pipeline.environment_asset.resolve_handler(source_path: str) -> str` (Task 1)
  - `scarecrow_pipeline.environment_asset.compute_bounding_box(corners: list[tuple[float, float, float]]) -> tuple[list[float], list[float]]` (Task 1)
  - `blender.collection_utils.snapshot_names()`, `new_objects_since()`, `flatten_into_collection()`, `remove_empty_scratch_collections()` (Task 2)
- Produces: the `Environment_<name>` collection contract described in the spec — no other task consumes this directly (Phase 3 will, later, out of scope here).

- [ ] **Step 1: Write `blender/import_environment_asset.py`**

```python
"""Onboard an environment/set asset (DAZ-sourced or natively Blender-
importable) into a normalized Environment_<name> collection.

Run headlessly against a fresh .blend (mirrors import_daz_artifact.py):

    blender -b -P blender/import_environment_asset.py -- \
        --source <path-to-asset> --collection-name <name> --out <output.blend> \
        [--root-paths <daz-root-paths.json>]

Supports DAZ .dbz/.duf sets and props via the same easy_import_daz path used
for characters (with posing/rig-merge options off, since a set/prop has no
posable figure to fit), and natively Blender-importable formats (glTF/GLB,
FBX, OBJ, USD) via bpy's own import_scene/wm operators. Both paths normalize
into a single Environment_<name> collection: new objects are flattened out of
whatever collection structure the importer created (see collection_utils.py),
any embedded camera/light objects are tagged with a scarecrow_env_role custom
property so Phase 3's render_scene() can find or override them, and a
world-space bounding box over all mesh objects is stored as
scarecrow_bbox_min/scarecrow_bbox_max custom properties on the collection.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collection_utils import (
    flatten_into_collection,
    new_objects_since,
    remove_empty_scratch_collections,
    snapshot_names,
)
from scarecrow_pipeline.environment_asset import compute_bounding_box, resolve_handler


def import_daz_set(source_path, root_paths_path):
    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.import_daz")
    from bl_ext.user_default.import_daz import api

    if root_paths_path:
        from bl_ext.user_default.import_daz.settings import GS
        with open(root_paths_path, encoding="utf-8-sig") as handle:
            GS.readDazPaths(json.load(handle), None, True)

    api.set_silent_mode(True)
    result = bpy.ops.daz.easy_import_daz(
        directory=os.path.dirname(source_path),
        files=[{"name": os.path.basename(source_path)}],
        fitMeshes="DBZFILE",
        useMakePosable=False,
        useHead=False,
        useUnits=False,
        useExpressions=False,
        useVisemes=False,
        useFacs=False,
        useFacsdetails=False,
        useFacsexpr=False,
        useBody=False,
        useMergeRigs=False,
        useMergeMaterials=True,
        useTransferClothes=False,
        useTransferFace=False,
        useTransferGeografts=False,
        useTransferHD=False,
    )
    api.set_silent_mode(False)
    print("IMPORT_RESULT", result)


def import_native(source_path, handler):
    if handler == "fbx":
        bpy.ops.import_scene.fbx(filepath=source_path)
    elif handler == "gltf":
        bpy.ops.import_scene.gltf(filepath=source_path)
    elif handler == "obj":
        bpy.ops.wm.obj_import(filepath=source_path)
    elif handler == "usd":
        bpy.ops.wm.usd_import(filepath=source_path)
    else:
        raise ValueError(f"No native import operator wired for handler {handler!r}")


def tag_camera_and_light_roles(objects):
    for obj in objects:
        if obj.type == "CAMERA":
            obj["scarecrow_env_role"] = "camera"
        elif obj.type == "LIGHT":
            obj["scarecrow_env_role"] = "light"


def store_bounding_box(collection):
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in collection.objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not corners:
        print(f"WARNING: no mesh objects in {collection.name!r}; skipping bounding box")
        return
    bbox_min, bbox_max = compute_bounding_box([tuple(corner) for corner in corners])
    collection["scarecrow_bbox_min"] = bbox_min
    collection["scarecrow_bbox_max"] = bbox_max


def onboard_environment(source_path, collection_name, out_path, root_paths_path):
    source_path = os.path.abspath(source_path)
    handler = resolve_handler(source_path)

    pre_object_names, pre_collection_names = snapshot_names()

    if handler == "daz":
        import_daz_set(source_path, root_paths_path)
    else:
        import_native(source_path, handler)

    new_objects = new_objects_since(pre_object_names)
    if not new_objects:
        raise RuntimeError(f"No new content found after importing {source_path!r}; nothing to onboard")

    target_name = f"Environment_{collection_name}"
    target = bpy.data.collections.new(target_name)
    bpy.context.scene.collection.children.link(target)
    flatten_into_collection(new_objects, target)
    remove_empty_scratch_collections(pre_collection_names, keep=target)

    tag_camera_and_light_roles(new_objects)
    store_bounding_box(target)

    print("ENVIRONMENT_COLLECTION", target.name, "OBJECTS", [obj.name for obj in target.objects])

    save_path = os.path.abspath(out_path)
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    print("SAVED", save_path)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--root-paths")
    args = parser.parse_args(argv)
    onboard_environment(args.source, args.collection_name, args.out, args.root_paths)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify no syntax errors**

Run: `python -m py_compile blender/import_environment_asset.py`
Expected: exits 0, no output.

- [ ] **Step 3: Verify the bpy-free imports resolve**

Run: `python -c "import sys; sys.path.insert(0, '.'); from scarecrow_pipeline.environment_asset import resolve_handler, compute_bounding_box; print(resolve_handler('x.gltf'))"`
Expected: prints `gltf` (confirms Task 1's module is wired correctly; the full script still needs live Blender to run end-to-end, covered in Task 4).

- [ ] **Step 4: Commit**

```bash
git add blender/import_environment_asset.py
git commit -m "Add import_environment_asset.py dispatcher for DAZ and native assets (scarecrow-bzy)"
```

---

### Task 4: Headless verification against real assets

**Files:**
- None created by this task except verification artifacts under `artifacts/` and, if it reveals bugs, fixes to the files from Tasks 2-3.

**Interfaces:**
- Consumes: `blender/import_environment_asset.py` (Task 3), a live Blender install with the `import_daz` addon, a DAZ-sourced set/prop file on disk, and a glTF/FBX file on disk (user-supplied per the design's resolved questions).

This task requires the user to point at real files — it cannot be scripted end-to-end without them. Steps below assume the user provides `<daz-set-path>` and `<native-asset-path>` when this task starts.

- [ ] **Step 1: Onboard the DAZ-sourced asset**

Run (substituting the real path and, if needed, the same `--root-paths` value used for `import_daz_artifact.py`):

```bash
blender -b -P blender/import_environment_asset.py -- \
    --source <daz-set-path> --collection-name TestSet \
    --out artifacts/environment_onboarding/TestSet.blend \
    --root-paths <daz-root-paths.json>
```

Expected stdout includes `IMPORT_RESULT`, `ENVIRONMENT_COLLECTION Environment_TestSet OBJECTS [...]` with a non-empty object list, and `SAVED artifacts/environment_onboarding/TestSet.blend`.

- [ ] **Step 2: Inspect the DAZ output collection**

Run:

```bash
blender -b artifacts/environment_onboarding/TestSet.blend -P - <<'EOF'
import bpy
coll = bpy.data.collections["Environment_TestSet"]
print("BBOX_MIN", coll.get("scarecrow_bbox_min"))
print("BBOX_MAX", coll.get("scarecrow_bbox_max"))
print("ROLES", [(obj.name, obj.get("scarecrow_env_role")) for obj in coll.objects if obj.type in ("CAMERA", "LIGHT")])
EOF
```

Expected: `BBOX_MIN`/`BBOX_MAX` are each 3 numbers spanning a plausible extent for the asset; any camera/light objects in the source asset show up with the matching `scarecrow_env_role`.

- [ ] **Step 3: Onboard the native (glTF/FBX) asset**

Run:

```bash
blender -b -P blender/import_environment_asset.py -- \
    --source <native-asset-path> --collection-name TestNative \
    --out artifacts/environment_onboarding/TestNative.blend
```

Expected stdout includes `ENVIRONMENT_COLLECTION Environment_TestNative OBJECTS [...]` with a non-empty object list and `SAVED artifacts/environment_onboarding/TestNative.blend`.

- [ ] **Step 4: Inspect the native output collection**

Repeat Step 2's inline-Python check against `artifacts/environment_onboarding/TestNative.blend` and `Environment_TestNative`.

- [ ] **Step 5: Regression-check `author_outfit_variant.py` after the Task 2 refactor**

Re-run the same command used to onboard `Outfit_WorkerUniform` (see the `scarecrow-2oz` commit message for the exact invocation, or `bd show scarecrow-2oz` if still queryable) against the same source `.dbz` and a scratch `--out` path. Confirm `VARIANT_COLLECTION Outfit_WorkerUniform OBJECTS [...]` and `DUPLICATES_REMOVED [...]` print the same object lists as before the refactor (compare against the reference file already checked in at `artifacts/outfit_variant_validation/request_workeruniform.json` and its paired output, or re-render and diff against `artifacts/outfit_variant_validation/request_reference.json` the same way the original commit did).

- [ ] **Step 6: Commit any fixes plus verification artifacts**

```bash
git add artifacts/environment_onboarding/
git commit -m "Verify Phase 4 environment onboarding against live DAZ and native assets (scarecrow-bzy)"
```

If Steps 1-5 required code fixes, include those files in the same commit or a preceding one, whichever keeps the history clear.

- [ ] **Step 7: Close the beads issue**

```bash
bd close scarecrow-bzy --reason="Environment/set onboarding dispatcher implemented and verified against DAZ + native assets"
```

---

## Post-plan

Closing `scarecrow-bzy` unblocks `scarecrow-8ng` (Phase 3: Multi-character CG scene compositing) — `bd ready` should surface it next.
