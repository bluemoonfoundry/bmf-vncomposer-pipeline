"""Save DAZ content-manager paths through the local DazScriptServer."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from dazpy import DazClient


def build_headless_script(source: str, output_path: str) -> str:
    escaped = output_path.replace("\\", "\\\\").replace('"', '\\"')
    # The stock script asks for a path with FileDialog. Replace only that
    # expression, retaining Daz Studio's own content-manager enumeration.
    pattern = r"var filepath = FileDialog\.doFileDialog\([\s\S]*?\);"
    replacement = f'var filepath = "{escaped}";'
    script, count = re.subn(pattern, replacement, source, count=1)
    if count != 1:
        raise ValueError("Could not locate the save_root_paths file dialog")
    script = script.replace("MessageBox.information( msg, appName, \"&OK\" );", "print(msg);")
    return script


def save_paths(script_path: Path, output_path: Path, timeout: float = 120.0) -> object:
    source = script_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daz_output_path = output_path.resolve().as_posix()
    client = DazClient(timeout=timeout)
    try:
        result = client.execute(
            build_headless_script(source, daz_output_path),
            max_wait=timeout,
        )
        return result.value
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    print(save_paths(args.script, args.out, args.timeout))


if __name__ == "__main__":
    main()
