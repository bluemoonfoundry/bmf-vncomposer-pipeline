# Outfit/hair onboarding workflow (scene-first)

Onboarding a new outfit or hair variant means getting a DAZ Studio-authored
`.dbz` for it, then fitting that `.dbz` onto an existing character master
blend with `blender/author_outfit_variant.py`. This document covers the first
half: producing the `.dbz` without driving a live DAZ Studio GUI session
programmatically.

## Why scene-first

An earlier approach scripted DAZ Studio's content manager directly (load the
base figure, apply the outfit via Smart Content, confirm auto-fit dialogs,
save). That's fragile: DAZ Studio is single-threaded, and an unexpected
auto-fit confirmation dialog can hang or crash the app with no way to
recover programmatically. This happened during the Worker Uniform onboarding
for Jason Cross (scarecrow-2oz).

The supported workflow instead starts from a `.duf` scene a person has
already built and saved by hand:

1. **Build the scene interactively in DAZ Studio.** Load the character,
   apply the outfit or hair via Smart Content as usual, let auto-fit run and
   confirm any dialogs yourself, then `File > Save As` a `.duf` scene file.
   This is the only step that touches DAZ Studio's GUI or content manager.
2. **Export the `.dbz` non-interactively** by running `scripts/daz_export.py`
   against that saved scene:

   ```powershell
   python scripts/daz_export.py `
     --daz-exe "X:/DAZNext/Applications/64-bit/DAZ 3D/DAZStudio4/DAZStudio.exe" `
     --scene artifacts/JasonCross_worker_uniform.duf `
     --script daz_scarecrow_project/Scripts/Diffeomorphic/export_to_blender.dsa `
     --out artifacts/dbz
   ```

   This launches DAZ Studio with `-noPrompt` against the already-saved scene,
   writes the exported `.dbz` to `artifacts/dbz/<scene-stem>.dbz`, and exits.
   No `openFile()`/content-manager calls and no auto-fit dialog handling are
   needed at this step, because the scene arrives fully prepared.
3. **Onboard the `.dbz` into the master blend** with
   `blender/author_outfit_variant.py` (see the [README](../README.md) for
   the invocation) and register the new outfit/hair name in
   `scarecrow_pipeline/registry.py` so `sprite_set.py` can validate combos
   against it.

## Why this is headless-safe

Diffeomorphic's own `export_to_blender.dsa` ends by opening an options
dialog (`createDialog()`) so a person can toggle HD export before picking a
save path — that dialog would hang forever under `-noPrompt` with no one to
click it. `scripts/daz_export.py` works around this the same way
`scripts/run_diffeomorphic_export.py` already does for the live-server path:
it loads the installed `.dsa` source, strips the dialog's entry point, and
calls `exportToBlender()` directly against a fixed output path derived from
the scene filename. The installed Diffeomorphic script itself is never
modified.

## CAUTION: headless mode is still fragile

`-noPrompt`/batch mode is not a guarantee of reliability — DAZ Studio can
crash or silently produce no output on some scenes or machines, independent
of the dialog problem this works around. `scripts/daz_export.py` therefore
checks that the expected `.dbz` was actually written and is non-empty after
the subprocess exits, and raises if not (in addition to the existing
`timeout` and non-zero exit code checks).

**If the headless export fails or the check above raises:** open the same
saved `.duf` scene in DAZ Studio interactively and run
`export_to_blender.dsa` from the File menu by hand (the same script the
headless path drives), then point `author_outfit_variant.py` at the `.dbz`
it produces. Don't assume a headless failure means the scene itself is bad —
retry interactively before concluding that.
