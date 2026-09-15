"""Synchronize DAZ Studio roots into Diffeomorphic's Blender settings."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from save_daz_root_paths import save_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daz-script", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--blender-loader", type=Path, required=True)
    args = parser.parse_args()

    save_paths(args.daz_script, args.paths)
    command = [
        str(args.blender), "--background", "--python", str(args.blender_loader),
        "--", "--paths", str(args.paths),
    ]
    subprocess.run(command, check=True)
    print(f"Synchronized DAZ roots through {args.paths}")


if __name__ == "__main__":
    main()
