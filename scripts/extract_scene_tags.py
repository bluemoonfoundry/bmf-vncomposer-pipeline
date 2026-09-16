"""Scan a Ren'Py project for '# scarecrow: <description>' tagged comments."""

import argparse
import json
from pathlib import Path

from scarecrow_pipeline.scene_tags import extract_scene_tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    manifest = []
    for rpy_path in sorted(args.project_root.rglob("*.rpy")):
        source_file = rpy_path.relative_to(args.project_root).as_posix()
        text = rpy_path.read_text(encoding="utf-8")
        manifest.extend(
            entry.model_dump(mode="json")
            for entry in extract_scene_tags(text, source_file=source_file)
        )

    serialized = json.dumps(manifest, indent=2) + "\n"
    if args.out:
        args.out.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
