import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import daz_export  # noqa: E402

FAKE_DIALOG_SCRIPT = (
    'var appName = "export_to_blender";\n'
    "function exportToBlender(filepath) {\n"
    '    MessageBox.information( msg, appName, "&OK" );\n'
    "}\n"
    "createDialog()\n"
)


def _write_fake_script(tmp_path) -> Path:
    script = tmp_path / "export_to_blender.dsa"
    script.write_text(FAKE_DIALOG_SCRIPT, encoding="utf-8")
    return script


def test_export_scene_writes_headless_script_and_returns_dbz_path(tmp_path, monkeypatch):
    script = _write_fake_script(tmp_path)
    scene = tmp_path / "JasonCross_worker_uniform.duf"
    scene.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"

    captured_commands = []

    def fake_run(command, check, timeout, text, capture_output):
        captured_commands.append(command)
        headless_script_path = Path(command[command.index("-script") + 1])
        source = headless_script_path.read_text(encoding="utf-8")
        assert "createDialog()" not in source
        assert "exportToBlender(" in source
        # Simulate Daz Studio actually producing the .dbz.
        dbz_path = output_dir / (scene.stem + ".dbz")
        dbz_path.write_bytes(b"fake-dbz-bytes")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(daz_export.subprocess, "run", fake_run)

    dbz_path = daz_export.export_scene(
        daz_exe="DazStudio.exe", scene=scene, script=script, output_dir=output_dir, timeout=60
    )

    assert dbz_path == output_dir / "JasonCross_worker_uniform.dbz"
    assert dbz_path.exists()
    assert len(captured_commands) == 1
    command = captured_commands[0]
    assert command[0] == "DazStudio.exe"
    assert "-noPrompt" in command
    assert str(scene) in command
    # The temp headless script is cleaned up after the run.
    headless_script_path = Path(command[command.index("-script") + 1])
    assert not headless_script_path.exists()


def test_export_scene_raises_when_dbz_never_written(tmp_path, monkeypatch):
    script = _write_fake_script(tmp_path)
    scene = tmp_path / "scene.duf"
    scene.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"

    def fake_run(command, check, timeout, text, capture_output):
        # Daz Studio "succeeds" but silently fails to produce the .dbz.
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(daz_export.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="missing or empty"):
        daz_export.export_scene(
            daz_exe="DazStudio.exe", scene=scene, script=script, output_dir=output_dir, timeout=60
        )


def test_export_scene_raises_on_nonzero_exit(tmp_path, monkeypatch):
    script = _write_fake_script(tmp_path)
    scene = tmp_path / "scene.duf"
    scene.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"

    def fake_run(command, check, timeout, text, capture_output):
        return SimpleNamespace(returncode=1, stderr="boom")

    monkeypatch.setattr(daz_export.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="boom"):
        daz_export.export_scene(
            daz_exe="DazStudio.exe", scene=scene, script=script, output_dir=output_dir, timeout=60
        )
