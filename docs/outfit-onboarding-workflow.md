# Outfit/hair onboarding workflow (scene-first)

Onboarding a new outfit or hair variant means getting a DAZ Studio-authored
`.dbz` for it, then fitting that `.dbz` onto an existing character master
blend with `blender/author_outfit_variant.py`. This document covers the first
half: producing the `.dbz` from a scene a person has already built, through a
live, GUI-resident DAZ Studio instance.

## Why scene-first, and why GUI-resident

An earlier approach scripted DAZ Studio's content manager directly (load the
base figure, apply the outfit via Smart Content, confirm auto-fit dialogs,
save). That's fragile: DAZ Studio is single-threaded, and an unexpected
auto-fit confirmation dialog can hang or crash the app with no way to
recover programmatically. This happened during the Worker Uniform onboarding
for Jason Cross (scarecrow-2oz).

A later attempt tried to fix that by dropping DAZ Studio's GUI entirely and
driving a standalone `-noPrompt` process against a pre-built `.duf` scene.
That made things worse, not better: `-noPrompt`/batch mode relies on large,
undocumented parts of DAZ Studio's codebase that implicitly assume a GUI is
present. Live verification (scarecrow-97w, scarecrow-n4c) reproduced a
standalone `-noPrompt` process running an initial Iray preview render and
then going fully idle indefinitely, never reaching the injected export
script -- and `subprocess.run`'s own `timeout` did not fire while the child
sat idle-but-alive, requiring a manual process kill. There is no reliable
way to bound a standalone headless DAZ Studio process, and the underlying
GUI dependency is not something this project can fix upstream.

**The supported workflow keeps DAZ Studio's GUI running at all times and
drives it through the local DazScriptServer.** There is no headless
fallback: if DazScriptServer/DAZ Studio isn't reachable, the fix is to get
the GUI instance running again, not to fall back to a standalone process.

1. **Build the scene interactively in DAZ Studio, then `File > Save As` a
   `.duf` scene file.** Load the character, apply the outfit or hair via
   Smart Content as usual, let auto-fit run and confirm any dialogs
   yourself, then save. This step is not optional: the exported `.dbz`
   embeds the scene's on-disk `filepath`, and Diffeomorphic's Blender
   importer uses that to locate a same-named `.dbz`/`.json` sibling next to
   the `.duf` for mesh fitting. A scene that was never saved exports with an
   empty `filepath` and the importer silently imports nothing (verified in
   scarecrow-czw). `scripts/onboard_character.py` handles staging the `.dbz`
   next to the `.duf` automatically -- it just needs the `.duf` to exist.
2. **Export the `.dbz`** by running `scripts/run_diffeomorphic_export.py`
   against that scene while DAZ Studio (with the DazScriptServer plugin,
   see [`docs/integration-setup.md`](integration-setup.md)) is still running:

   ```powershell
   python scripts/run_diffeomorphic_export.py `
     --script daz_scarecrow_project/Scripts/Diffeomorphic/export_to_blender.dsa `
     --out artifacts/dbz/JasonCross_worker_uniform.dbz
   ```

   This sends the export script through the running DazScriptServer, which
   executes it against the currently loaded scene. No `openFile()`/content
   manager calls and no auto-fit dialog handling are needed at this step,
   because the scene arrives fully prepared and DAZ Studio's own GUI process
   is doing the work.
3. **Onboard the `.dbz` into the master blend** with
   `blender/author_outfit_variant.py` (see the [README](../README.md) for
   the invocation) and register the new outfit/hair name in
   `scarecrow_pipeline/registry.py` so `sprite_set.py` can validate combos
   against it.

## Why this is dialog-safe

Diffeomorphic's own `export_to_blender.dsa` ends by opening an options
dialog (`createDialog()`) so a person can toggle HD export before picking a
save path -- that dialog would hang forever with no one to click it in an
automated request. `scripts/run_diffeomorphic_export.py` works around this
via `daz_headless_script.build_headless_script()`: it loads the installed
`.dsa` source, strips the dialog's entry point, and calls
`exportToBlender()` directly against a fixed output path. The installed
Diffeomorphic script itself is never modified. This only strips a modal
dialog from a script run inside an already-running GUI session -- it is not
a standalone headless DAZ Studio invocation.
