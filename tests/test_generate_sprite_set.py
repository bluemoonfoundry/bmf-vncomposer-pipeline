import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_sprite_set  # noqa: E402

from scarecrow_pipeline.registry import CharacterRecord, Registry  # noqa: E402
from scarecrow_pipeline.sprite_set import SpriteCombo, SpriteSetSpec  # noqa: E402


def _registry_with_jason():
    registry = Registry()
    registry.upsert_character(CharacterRecord(
        name="JasonCross",
        master_blend="artifacts/JasonCross_imported.blend",
        collection="Jasoncross_v1",
    ))
    return registry


def _fake_run_factory(fail_on_filenames=frozenset()):
    calls = []

    def fake_run(command, capture_output=True, text=True):
        calls.append(command)
        request_path = Path(command[-1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        filename = Path(request["output_path"]).name
        if filename in fail_on_filenames:
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    fake_run.calls = calls
    return fake_run


def test_generate_sprite_set_writes_manifest_for_every_combo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scarecrow_pipeline.sprite_set.Registry.load",
        lambda *a, **k: _registry_with_jason(),
    )
    spec = SpriteSetSpec(
        character="JasonCross",
        output_dir=str(tmp_path / "sprites"),
        combos=[
            SpriteCombo(emotion_label="happy"),
            SpriteCombo(emotion_label="sad"),
        ],
    )

    fake_run = _fake_run_factory()
    manifest = generate_sprite_set.generate_sprite_set(
        spec, blender_exe="blender.exe", run=fake_run
    )

    assert len(manifest) == 2
    assert all(entry["status"] == "ok" for entry in manifest)
    assert len(fake_run.calls) == 2

    manifest_path = tmp_path / "sprites" / "manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_generate_sprite_set_uses_absolute_paths_in_subprocess_command(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scarecrow_pipeline.sprite_set.Registry.load",
        lambda *a, **k: _registry_with_jason(),
    )
    spec = SpriteSetSpec(
        character="JasonCross",
        output_dir=str(tmp_path / "sprites"),
        combos=[SpriteCombo(emotion_label="happy")],
    )

    fake_run = _fake_run_factory()
    generate_sprite_set.generate_sprite_set(spec, blender_exe="blender.exe", run=fake_run)

    command = fake_run.calls[0]
    worker_path = Path(command[command.index("--python") + 1])
    request_path = Path(command[-1])
    assert worker_path.is_absolute()
    assert request_path.is_absolute()


def test_generate_sprite_set_continues_past_failures_and_reports_them(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scarecrow_pipeline.sprite_set.Registry.load",
        lambda *a, **k: _registry_with_jason(),
    )
    spec = SpriteSetSpec(
        character="JasonCross",
        output_dir=str(tmp_path / "sprites"),
        combos=[
            SpriteCombo(emotion_label="happy"),
            SpriteCombo(emotion_label="sad"),
            SpriteCombo(emotion_label="angry"),
        ],
    )

    failing_filename = spec.expand()[1][0]
    fake_run = _fake_run_factory(fail_on_filenames={failing_filename})
    manifest = generate_sprite_set.generate_sprite_set(spec, blender_exe="blender.exe", run=fake_run)

    assert len(manifest) == 3
    assert len(fake_run.calls) == 3
    statuses = [entry["status"] for entry in manifest]
    assert statuses.count("ok") == 2
    assert statuses.count("failed") == 1
    failed_entry = next(entry for entry in manifest if entry["status"] == "failed")
    assert "boom" in failed_entry["stderr"]
