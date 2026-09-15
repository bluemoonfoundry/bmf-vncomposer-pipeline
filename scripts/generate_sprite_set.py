"""CLI orchestrator: expand a SpriteSetSpec and render each combo headlessly.

Usage:
    python scripts/generate_sprite_set.py --spec spec.json [--blender-exe path/to/blender.exe] [--master-blend override.blend]

One Blender subprocess per combo (see scarecrow_pipeline/sprite_set.py and
blender/worker.py's module docstring for why: worker.py:render() mutates
shared scene/compositor/shadow-catcher state that isn't proven safe to
repeat across calls in one process). Blender is launched against its default
(blank) startup scene, NOT the character's master_blend -- worker.py's
append_collection() loads the character collection FROM master_blend via
bpy.data.libraries.load(), which raises "Cannot load from the current blend
file" if master_blend is already the open scene:

    blender --background --python-exit-code 1 --python blender/worker.py -- --request <request.json>

--python-exit-code makes Blender exit non-zero when worker.py raises --
without it, Blender's default background-mode behavior is to print the
traceback and still exit 0, which would make the manifest report "ok" for
a combo that never actually rendered.

All file paths passed to the Blender subprocess (worker script, request
JSON) are made absolute first -- a relative --out-dir/output path handed to
a headless Blender subprocess on Windows can resolve against the wrong
drive/cwd.

Never halts on the first failed combo: every combo is attempted, and
failures are recorded in the manifest rather than aborting the batch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scarecrow_pipeline.sprite_set import SpriteSetSpec
from scarecrow_pipeline.schemas import RenderRequest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLCHAIN_PATH = REPO_ROOT / "config" / "toolchain.json"
WORKER_SCRIPT = REPO_ROOT / "blender" / "worker.py"


def resolve_blender_exe(explicit: str | None, toolchain_path: Path = DEFAULT_TOOLCHAIN_PATH) -> str:
    if explicit:
        return explicit
    toolchain = json.loads(toolchain_path.read_text(encoding="utf-8"))
    return toolchain["blender"]["executable"]


def _resolve_absolute(path_str: str) -> Path:
    path = Path(path_str)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def render_combo(
    filename: str,
    request: RenderRequest,
    *,
    output_dir: Path,
    blender_exe: str,
    master_blend_override: str | None,
    run,
) -> dict:
    """Render one combo in its own Blender subprocess; never raises on failure."""
    request_dict = request.model_dump()
    request_dict["output_path"] = str((output_dir / filename).resolve())

    master_blend_path = _resolve_absolute(master_blend_override or request.master_blend)
    request_dict["master_blend"] = str(master_blend_path)

    requests_dir = output_dir / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    request_json_path = (requests_dir / f"{Path(filename).stem}.request.json").resolve()
    request_json_path.write_text(json.dumps(request_dict, indent=2), encoding="utf-8")

    command = [
        blender_exe,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(WORKER_SCRIPT.resolve()),
        "--",
        "--request",
        str(request_json_path),
    ]

    entry = {
        "outfit": request.outfit,
        "hair": request.hair,
        "emotion": request.emotion,
        "output_path": request_dict["output_path"],
    }
    try:
        completed = run(command, capture_output=True, text=True)
    except OSError as exc:
        entry["status"] = "failed"
        entry["stderr"] = str(exc)
        return entry

    if completed.returncode == 0:
        entry["status"] = "ok"
    else:
        entry["status"] = "failed"
        entry["stderr"] = (completed.stderr or "")[-4000:]
    return entry


def generate_sprite_set(
    spec: SpriteSetSpec,
    *,
    blender_exe: str,
    master_blend_override: str | None = None,
    run=subprocess.run,
) -> list[dict]:
    output_dir = Path(spec.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = [
        render_combo(
            filename,
            request,
            output_dir=output_dir,
            blender_exe=blender_exe,
            master_blend_override=master_blend_override,
            run=run,
        )
        for filename, request in spec.expand()
    ]

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--blender-exe")
    parser.add_argument("--master-blend")
    args = parser.parse_args(argv)

    spec = SpriteSetSpec.model_validate(json.loads(args.spec.read_text(encoding="utf-8")))
    blender_exe = resolve_blender_exe(args.blender_exe)

    manifest = generate_sprite_set(spec, blender_exe=blender_exe, master_blend_override=args.master_blend)

    failures = [entry for entry in manifest if entry["status"] != "ok"]
    for entry in failures:
        print(f"FAILED: {entry['output_path']}: {entry.get('stderr', '')[:500]}", file=sys.stderr)
    print(f"{len(manifest) - len(failures)}/{len(manifest)} renders succeeded")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
