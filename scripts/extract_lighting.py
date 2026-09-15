"""Extract Blender-friendly lighting metadata from a background plate."""

import argparse
import json
from pathlib import Path

from scarecrow_pipeline.vision import estimate_lighting


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("background", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = estimate_lighting(args.background).model_dump(mode="json")
    serialized = json.dumps(payload, indent=2) + "\n"
    if args.out:
        args.out.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
