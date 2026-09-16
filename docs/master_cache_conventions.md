# Master-cache registry conventions

`artifacts/registry.json` (loaded/saved via `scarecrow_pipeline/registry.py`'s
`Registry`) is the source of truth for what's been onboarded and where it
lives. Downstream tools (`scarecrow_pipeline/sprite_set.py`,
`blender/author_outfit_variant.py`, multi-character scene assembly) use it to
validate that a request references a character/outfit/environment/hair combo
that actually exists before spending render time on it -- the same role
`docs/posable_bones.json` and `docs/facs_controls.json` play for bone/control
names.

These conventions describe what Phase 0-4 usage has settled in practice, not
a spec designed up front -- see the caveats below where practice and the
tidy version diverge.

## Directory layout under `artifacts/`

- `artifacts/registry.json` -- the single `Registry` JSON document (character
  and environment records).
- `artifacts/<Character>_imported.blend` -- a character's master blend, the
  reusable cache `CharacterRecord.master_blend` points at. Produced by
  `blender/import_daz_artifact.py` (directly, or via
  `scripts/onboard_character.py`).
- `artifacts/dbz/<Name>.dbz` -- staging location for raw Diffeomorphic
  exports (`scripts/run_diffeomorphic_export.py --out`) before they're
  imported into a master blend. **Caveat:** the original JasonCross onboarding
  predates this convention and left its `.dbz`/`.duf` files directly under
  `artifacts/`; new onboarding (e.g. `AbandonedSubwayStation.dbz`) uses the
  `artifacts/dbz/` subdirectory. Prefer `artifacts/dbz/` going forward.
- `artifacts/environment_onboarding/<Name>.blend` -- environment master
  blends, produced by `blender/import_environment_asset.py`.
- `artifacts/<Character>.blend1` -- Blender's own crash-recovery backup of
  the previous save. Not a project convention; safe to ignore/delete.
- Ad hoc validation subdirectories (`artifacts/jaw_fix_validation/`,
  `artifacts/outfit_variant_validation/`, `artifacts/scene_verify/`,
  `artifacts/clean_stage/`) hold one-off verification renders from earlier
  phases. They aren't referenced by the registry and aren't a convention to
  continue -- route new verification output through
  `scripts/generate_sprite_set.py`'s `--output-dir` (manifest + PNGs)
  instead of a hand-named directory.

## Collection naming

Each registry record's `collection` field names a specific top-level
collection inside its `.blend` -- the identifier `blender/worker.py`'s
`append_collection()`/`resolve_armature()` and `RenderRequest.character` key
off of:

- **Characters:** `blender/import_daz_artifact.py --collection-name <Name>`
  (added for `scripts/onboard_character.py`) wraps whatever DAZ's importer
  creates into one collection literally named `<Name>` -- e.g. `JasonCross`.
  **Caveat:** the existing `JasonCross` record predates `--collection-name`
  and was onboarded by hand before this normalization existed, so its
  registry `collection` value is `Jasoncross_v1` (DAZ's own import naming,
  case-folded, plus an ad hoc `_v1` suffix) rather than the bare
  `JasonCross`. Don't infer a versioning suffix convention from this one
  record -- it's a naming accident, not a scheme to replicate. New
  characters onboarded via `onboard_character.py` get the bare
  `<CharacterName>` as their collection name.
- **Outfits/hair:** `Outfit_<name>` / `Hair_<name>`, nested as *children* of
  the character's own collection (not flattened into it) so
  `set_variant_visibility` (`blender/worker.py`) can toggle them per-render.
  Written by `blender/author_outfit_variant.py --collection-name <name>
  --kind outfit|hair`. Registered in `CharacterRecord.outfits`/`.hair`.
- **Environments:** `Environment_<name>`, written by
  `blender/import_environment_asset.py --collection-name <name>`.

## Versioning

There's no automatic version-bumping. `author_outfit_variant.py` and
`import_daz_artifact.py`/`onboard_character.py` always take an explicit
output path:

- Pass `--out <new_path>.blend` to save a *separate* versioned copy (e.g.
  `JasonCross_B_imported.blend` for an alternate take), leaving the original
  master blend untouched.
- Omit `--out` (outfit authoring only) to update the existing master blend
  in place.

Because the registry just stores whatever path was last written for a given
name, re-running onboarding against the same `--blend-out`/`--out` path is
how an existing character/outfit/environment gets updated; pointing it at a
new path and re-registering under the same name is how an alternate version
supersedes the old one in the registry (the old `.blend` file itself is not
deleted).

## See also

- [`docs/outfit-onboarding-workflow.md`](outfit-onboarding-workflow.md) --
  scene-first DAZ export workflow feeding into `author_outfit_variant.py`.
- [`README.md`](../README.md) -- CLI invocations for each step.
