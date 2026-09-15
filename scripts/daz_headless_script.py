"""Strip Diffeomorphic's interactive export dialog for headless invocation.

Shared by scripts/daz_export.py (CLI export against a saved .duf) and
scripts/run_diffeomorphic_export.py (export via a live DazScriptServer).
Has no dependency on dazpy or any live DAZ Studio connection.
"""

from __future__ import annotations


def build_headless_script(source: str, output_path: str) -> str:
    marker = "createDialog()"
    if marker not in source:
        raise ValueError("Diffeomorphic exporter entry point was not found")
    prefix, _ = source.rsplit(marker, 1)
    # Avoid a modal dialog that would leave a server request waiting forever.
    prefix = prefix.replace("MessageBox.information( msg, appName, \"&OK\" );", "print(msg);")
    escaped = output_path.replace("\\", "\\\\").replace('"', '\\"')
    return prefix + f'exportToBlender("{escaped}");\n'
