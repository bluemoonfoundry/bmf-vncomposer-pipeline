"""Run Diffeomorphic's installed DAZ exporter without opening its dialog.

This does not modify the installed Diffeomorphic script. It loads that script,
replaces only its final interactive entry point, and sends the resulting source
through the local DazScriptServer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dazpy import DazClient

from daz_headless_script import build_headless_script


def export(script_path: Path, output_path: Path, timeout: float) -> object:
    source = script_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = DazClient(timeout=timeout)
    try:
        result = client.execute(build_headless_script(source, str(output_path)), max_wait=timeout)
        return result.value
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    print(export(args.script, args.out, args.timeout))


if __name__ == "__main__":
    main()
