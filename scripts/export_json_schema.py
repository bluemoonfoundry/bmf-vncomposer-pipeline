"""Regenerate schemas/render_request.schema.json from the live Pydantic model.

The JSON Schema file previously drifted from scarecrow_pipeline.schemas.RenderRequest
(it still described an abandoned base_pose: str pose design after PosePayload was
redesigned around bone_rotations). Generating it from the model directly makes
that class of drift impossible -- edit the Pydantic model, then rerun this script.

Usage:
  python scripts/export_json_schema.py [--out schemas/render_request.schema.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scarecrow_pipeline.schemas import RenderRequest  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "schemas" / "render_request.schema.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    schema = RenderRequest.model_json_schema()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
