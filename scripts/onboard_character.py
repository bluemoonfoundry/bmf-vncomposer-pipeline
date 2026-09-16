"""CLI orchestrator: onboard a new character into the master-cache registry.

Pure orchestration -- chains three existing standalone steps, no new render
logic:

    1. run_diffeomorphic_export.export() sends the installed Diffeomorphic
       export script through a running DazScriptServer against the scene
       already loaded in DAZ Studio's GUI (see
       docs/outfit-onboarding-workflow.md), producing a .dbz.
    2. `blender -b -P blender/import_daz_artifact.py` imports that .dbz into
       a fresh master .blend, wrapping the new content into a collection
       named after the character (--collection-name).
    3. scarecrow_pipeline/registry.py's Registry gains a CharacterRecord
       pointing at the resulting master_blend/collection.

Usage:
    python scripts/onboard_character.py --character JasonCross `
      --daz-script daz_scarecrow_project/Scripts/Diffeomorphic/export_to_blender.dsa `
      --dbz-out artifacts/dbz/JasonCross.dbz `
      --blend-out artifacts/JasonCross_imported.blend `
      [--root-paths artifacts/daz-root-paths.json] [--blender-exe path/to/blender.exe]

Requires a live, GUI-resident DAZ Studio + DazScriptServer instance for step
1 -- there is no headless fallback (see docs/outfit-onboarding-workflow.md).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from scarecrow_pipeline.registry import CharacterRecord, Registry

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_TOOLCHAIN_PATH = REPO_ROOT / "config" / "toolchain.json"
IMPORT_SCRIPT = REPO_ROOT / "blender" / "import_daz_artifact.py"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from run_diffeomorphic_export import export as export_dbz  # noqa: E402

CHARACTER_COLLECTION_RE = re.compile(r"^CHARACTER_COLLECTION (.+)$", re.MULTILINE)


def resolve_blender_exe(explicit: str | None, toolchain_path: Path = DEFAULT_TOOLCHAIN_PATH) -> str:
    if explicit:
        return explicit
    toolchain = json.loads(toolchain_path.read_text(encoding="utf-8"))
    return toolchain["blender"]["executable"]


def _resolve_absolute(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def import_dbz(
    dbz_path: Path,
    blend_path: Path,
    collection_name: str,
    root_paths: Path | None,
    blender_exe: str,
    run,
) -> str:
    """Run import_daz_artifact.py headlessly; return the CHARACTER_COLLECTION it reports."""
    command = [
        blender_exe,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(IMPORT_SCRIPT.resolve()),
        "--",
        "--dbz",
        str(dbz_path),
        "--blend",
        str(blend_path),
        "--collection-name",
        collection_name,
    ]
    if root_paths:
        command += ["--root-paths", str(root_paths)]

    completed = run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"import_daz_artifact.py failed (exit {completed.returncode}):\n"
            f"{completed.stdout or ''}\n{completed.stderr or ''}"
        )

    match = CHARACTER_COLLECTION_RE.search(completed.stdout or "")
    if not match:
        raise RuntimeError(
            "import_daz_artifact.py did not report a CHARACTER_COLLECTION line; "
            "check that --collection-name is supported by the installed script.\n"
            f"{completed.stdout or ''}"
        )
    return match.group(1).strip()


def onboard_character(
    character: str,
    daz_script: Path,
    dbz_out: str,
    blend_out: str,
    *,
    blender_exe: str,
    root_paths: str | None = None,
    export_timeout: float = 1800,
    registry_path: Path | str | None = None,
    export_fn=export_dbz,
    run=subprocess.run,
) -> dict:
    dbz_path = _resolve_absolute(dbz_out)
    blend_path = _resolve_absolute(blend_out)
    root_paths_path = _resolve_absolute(root_paths) if root_paths else None

    export_fn(Path(daz_script), dbz_path, export_timeout)

    collection_name = import_dbz(dbz_path, blend_path, character, root_paths_path, blender_exe, run)

    registry = Registry.load(registry_path) if registry_path else Registry.load()
    registry.upsert_character(CharacterRecord(
        name=character,
        master_blend=str(blend_path),
        collection=collection_name,
    ))
    registry.save(registry_path) if registry_path else registry.save()

    return {
        "character": character,
        "dbz": str(dbz_path),
        "master_blend": str(blend_path),
        "collection": collection_name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True)
    parser.add_argument("--daz-script", required=True, type=Path)
    parser.add_argument("--dbz-out", required=True)
    parser.add_argument("--blend-out", required=True)
    parser.add_argument("--root-paths")
    parser.add_argument("--blender-exe")
    parser.add_argument("--export-timeout", type=float, default=1800)
    parser.add_argument("--registry-path")
    args = parser.parse_args(argv)

    blender_exe = resolve_blender_exe(args.blender_exe)
    result = onboard_character(
        args.character,
        args.daz_script,
        args.dbz_out,
        args.blend_out,
        blender_exe=blender_exe,
        root_paths=args.root_paths,
        export_timeout=args.export_timeout,
        registry_path=args.registry_path,
    )
    print(f"Onboarded {result['character']}: {result['master_blend']} ({result['collection']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
