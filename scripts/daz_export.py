"""Run a Daz/Diffeomorphic export with a hard timeout.

Usage: python scripts/daz_export.py --daz-exe DazStudio.exe --scene figure.duf --script export_to_blender.dsa --out cache
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def export_scene(daz_exe: str, scene: Path, script: Path, output_dir: Path, timeout: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Daz's script arguments are passed as script variables; the DSA is responsible
    # for creating the .dbz companion required by Diffeomorphic.
    command = [daz_exe, "-noPrompt", "-noSplash", str(scene), "-script", str(script), str(output_dir)]
    try:
        completed = subprocess.run(command, check=False, timeout=timeout, text=True, capture_output=True)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Daz export exceeded {timeout}s") from exc
    if completed.returncode:
        raise RuntimeError(f"Daz export failed ({completed.returncode}): {completed.stderr[-2000:]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daz-exe", required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    export_scene(args.daz_exe, args.scene, args.script, args.out, args.timeout)


if __name__ == "__main__":
    main()
