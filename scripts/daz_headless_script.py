"""Strip Diffeomorphic's interactive export dialog for scripted invocation.

Used by scripts/run_diffeomorphic_export.py, which sends the patched script
through a live, GUI-resident DazScriptServer -- the only supported export
path (see scarecrow-n4c: standalone -noPrompt Daz Studio processes can wedge
indefinitely with no automatic recovery, so that path was removed).
Has no dependency on dazpy or any live DAZ Studio connection itself.
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
