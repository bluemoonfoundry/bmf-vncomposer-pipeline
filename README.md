# bmf-vncomposer-pipeline

This repository provides a small, headless-first pipeline for turning semantic VN scene data into Blender renders. Daz Studio is an offline authoring/export step; Blender owns runtime posing, lighting, Cycles rendering, and compositing.

## Layout

- `scarecrow_pipeline/schemas.py` — validated request and metadata models.
- `scarecrow_pipeline/vision.py` — dependency-light background light estimation (Pillow, optional OpenCV).
- `scarecrow_pipeline/nl_appearance.py` — translates a natural-language appearance description into a validated `PosePayload`/`FACSExpression` pair via a provider-agnostic `LLMClient`; `AnthropicClient` (optional `pip install -e '.[llm]'`, `ANTHROPIC_API_KEY`) is the one concrete implementation.
- `scripts/run_diffeomorphic_export.py` — export of a saved `.duf` scene to `.dbz` via a GUI-resident DazScriptServer (see [`docs/outfit-onboarding-workflow.md`](docs/outfit-onboarding-workflow.md)).
- `blender/worker.py` — Blender 4.x background worker (`blender -b -P ... -- ...`).
- `blender/compositor.py` — reusable compositor graph builder.
- `schemas/*.json` — JSON Schema documents for external LLM/tool integrations.

Install with `pip install -e .` (and `pip install -e '.[vision]'` for OpenCV). Blender scripts are loaded by Blender's Python and do not require Blender at package-install time.

Machine-specific Blender/DAZ paths and pinned integration versions are in [`config/toolchain.json`](config/toolchain.json). Follow [`docs/integration-setup.md`](docs/integration-setup.md) before attempting the first Diffeomorphic bake.

The onboarded characters/outfits/hair/environments that make up the master
cache are tracked in `artifacts/registry.json`; see
[`docs/master_cache_conventions.md`](docs/master_cache_conventions.md) for
its directory layout, collection naming, and versioning conventions.

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
  --collection-name Character `
  --root-paths artifacts/daz-root-paths.json
```

The root paths are applied in memory for the headless Blender process that performs the import. This avoids the foreground-only settings-save context in Diffeomorphic 5.1. `--collection-name` wraps the new content into one top-level collection under that name, matching what `scarecrow_pipeline/registry.py`'s `CharacterRecord.collection` and `blender/worker.py`'s `append_collection()` expect; omit it to keep the importer's own collection naming.

`scripts/onboard_character.py` chains this whole sequence (root-path sync, DBZ export, import, and registry write) for onboarding a new character in one step.

To produce the `.dbz` for a new outfit/hair variant in the first place, see
[`docs/outfit-onboarding-workflow.md`](docs/outfit-onboarding-workflow.md):
build and save the scene in Daz Studio's GUI, keep Daz Studio running, then
run `scripts/run_diffeomorphic_export.py` against that saved `.duf` for a
dialog-free export through the local DazScriptServer.

Author an outfit/hair variant into an existing master blend (fits the new DBZ's meshes to the character already in the scene and renames its collection to the `Outfit_<name>`/`Hair_<name>` convention `set_variant_visibility` expects, see `blender/worker.py`):

```powershell
blender -b artifacts/Character.blend -P blender/author_outfit_variant.py -- `
  --dbz X:/path/to/Outfit.dbz `
  --collection-name casual `
  --kind outfit `
  --out artifacts/Character_casual.blend
```

Omit `--out` to save the variant back into the master blend in place.
