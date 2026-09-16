"""Run a Daz/Diffeomorphic export against an already-prepared scene, headlessly.

The workflow this supports: a user builds and saves a .duf scene interactively
in Daz Studio (character + outfit already applied, no further scene-graph
edits needed), then this script launches Daz Studio non-interactively
(-noPrompt) against that saved scene to produce the .dbz Diffeomorphic/Blender
import expects. It never drives the content manager or scene graph itself.

Daz Studio's own export_to_blender.dsa ends by opening an options dialog
(createDialog()), which would hang under -noPrompt with no one to click it.
We reuse daz_headless_script.build_headless_script() (shared with
run_diffeomorphic_export.py's live-server path) to strip that dialog and
call exportToBlender() directly with a fixed output path, writing the
patched script to a temp file passed via -script.

CAUTION: Daz Studio's -noPrompt/headless mode is itself fragile -- it can
crash or silently produce no output on some scenes/machines, independent of
the dialog issue this works around. This script verifies the expected .dbz
was actually written and is non-empty; if that check fails, fall back to
running the export interactively from Daz Studio's File menu against the
same saved scene.

CONFIRMED (scarecrow-97w): on at least one real character+outfit scene, Daz
Studio ran an initial Iray preview render and then went idle indefinitely
before ever reaching the injected -script -- and subprocess.run's own
`timeout` did NOT fire while the child sat idle-but-alive, requiring a
manual process kill. The `timeout` argument here is not a reliable ceiling;
see scarecrow-n4c for the follow-up (a hard external watchdog or an
upstream fix is still needed).

Usage: python scripts/daz_export.py --daz-exe DazStudio.exe --scene figure.duf --script export_to_blender.dsa --out cache
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from daz_headless_script import build_headless_script


def export_scene(daz_exe: str, scene: Path, script: Path, output_dir: Path, timeout: int) -> Path:
    scene = scene.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dbz_path = output_dir / (scene.stem + ".dbz")
    headless_source = build_headless_script(script.read_text(encoding="utf-8"), str(dbz_path))

    with tempfile.NamedTemporaryFile(
        "w", suffix=".dsa", delete=False, encoding="utf-8", dir=output_dir
    ) as handle:
        handle.write(headless_source)
        headless_script = Path(handle.name)

    try:
        command = [daz_exe, "-noPrompt", "-noSplash", str(scene), "-script", str(headless_script)]
        try:
            completed = subprocess.run(command, check=False, timeout=timeout, text=True, capture_output=True)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Daz export exceeded {timeout}s") from exc
    finally:
        headless_script.unlink(missing_ok=True)

    if completed.returncode:
        raise RuntimeError(f"Daz export failed ({completed.returncode}): {completed.stderr[-2000:]}")
    if not dbz_path.exists() or dbz_path.stat().st_size == 0:
        raise RuntimeError(
            f"Daz export reported success but {dbz_path} is missing or empty -- "
            "headless mode may have failed silently; retry the export interactively "
            "from Daz Studio's File menu against the same saved scene."
        )
    return dbz_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daz-exe", required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    dbz_path = export_scene(args.daz_exe, args.scene, args.script, args.out, args.timeout)
    print(dbz_path)


if __name__ == "__main__":
    main()
