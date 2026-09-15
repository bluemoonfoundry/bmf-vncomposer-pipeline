# Scarecrow VN rendering pipeline

This repository provides a small, headless-first pipeline for turning semantic VN scene data into Blender renders. Daz Studio is an offline authoring/export step; Blender owns runtime posing, lighting, Cycles rendering, and compositing.

## Layout

- `scarecrow_pipeline/schemas.py` — validated request and metadata models.
- `scarecrow_pipeline/vision.py` — dependency-light background light estimation (Pillow, optional OpenCV).
- `scripts/daz_export.py` — timeout-safe Daz/Diffeomorphic export wrapper.
- `blender/worker.py` — Blender 4.x background worker (`blender -b -P ... -- ...`).
- `blender/compositor.py` — reusable compositor graph builder.
- `schemas/*.json` — JSON Schema documents for external LLM/tool integrations.

Install with `pip install -e .` (and `pip install -e '.[vision]'` for OpenCV). Blender scripts are loaded by Blender's Python and do not require Blender at package-install time.

Machine-specific Blender/DAZ paths and pinned integration versions are in [`config/toolchain.json`](config/toolchain.json). Follow [`docs/integration-setup.md`](docs/integration-setup.md) before attempting the first Diffeomorphic bake.

Example worker invocation:

```text
blender -b scene.blend -P blender/worker.py -- --request request.json
```

The worker accepts either a complete `RenderRequest` JSON file or the CLI overrides `--character`, `--outfit`, and `--emotion`.

Synchronize DAZ content roots and import a DBZ artifact:

```powershell
python scripts/sync_root_paths.py `
  --daz-script daz_scarecrow_project/Scripts/Diffeomorphic/save_root_paths.dsa `
  --paths artifacts/daz-root-paths.json `
  --blender "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe" `
  --blender-loader blender/load_daz_root_paths.py

blender -b -P blender/import_daz_artifact.py -- `
  --dbz X:/path/to/Character.dbz `
  --blend artifacts/Character.blend `
  --root-paths artifacts/daz-root-paths.json
```

The root paths are applied in memory for the headless Blender process that performs the import. This avoids the foreground-only settings-save context in Diffeomorphic 5.1.

Author an outfit/hair variant into an existing master blend (fits the new DBZ's meshes to the character already in the scene and renames its collection to the `Outfit_<name>`/`Hair_<name>` convention `set_variant_visibility` expects, see `blender/worker.py`):

```powershell
blender -b artifacts/Character.blend -P blender/author_outfit_variant.py -- `
  --dbz X:/path/to/Outfit.dbz `
  --collection-name casual `
  --kind outfit `
  --out artifacts/Character_casual.blend
```

Omit `--out` to save the variant back into the master blend in place.
