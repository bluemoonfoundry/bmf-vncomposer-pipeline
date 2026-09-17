import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import onboard_character  # noqa: E402

from scarecrow_pipeline.registry import Registry  # noqa: E402


def _fake_export_fn(calls):
    def export_fn(daz_script, dbz_path, timeout):
        calls.append((Path(daz_script), Path(dbz_path), timeout))
    return export_fn


def _fake_stage_fn(dbz_path):
    return Path(dbz_path)


def _fake_run_factory(collection_name="JasonCross", returncode=0, stderr=""):
    calls = []

    def fake_run(command, capture_output=True, text=True):
        calls.append(command)
        stdout = f"IMPORT_RESULT {{'FINISHED'}}\nCHARACTER_COLLECTION {collection_name}\n"
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    fake_run.calls = calls
    return fake_run


def test_onboard_character_writes_registry_entry(tmp_path):
    export_calls = []
    fake_run = _fake_run_factory()
    registry_path = tmp_path / "registry.json"

    result = onboard_character.onboard_character(
        "JasonCross",
        tmp_path / "export_to_blender.dsa",
        tmp_path / "dbz" / "JasonCross.dbz",
        tmp_path / "JasonCross_imported.blend",
        blender_exe="blender.exe",
        registry_path=registry_path,
        export_fn=_fake_export_fn(export_calls),
        stage_fn=_fake_stage_fn,
        run=fake_run,
    )

    assert result["character"] == "JasonCross"
    assert result["collection"] == "JasonCross"
    assert len(export_calls) == 1
    assert len(fake_run.calls) == 1

    registry = Registry.load(registry_path)
    record = registry.characters["JasonCross"]
    assert record.collection == "JasonCross"
    assert record.master_blend == result["master_blend"]


def test_onboard_character_uses_absolute_paths_in_blender_command(tmp_path):
    fake_run = _fake_run_factory()

    onboard_character.onboard_character(
        "JasonCross",
        tmp_path / "export_to_blender.dsa",
        "relative/JasonCross.dbz",
        "relative/JasonCross_imported.blend",
        blender_exe="blender.exe",
        registry_path=tmp_path / "registry.json",
        export_fn=_fake_export_fn([]),
        stage_fn=_fake_stage_fn,
        run=fake_run,
    )

    command = fake_run.calls[0]
    dbz_path = Path(command[command.index("--dbz") + 1])
    blend_path = Path(command[command.index("--blend") + 1])
    assert dbz_path.is_absolute()
    assert blend_path.is_absolute()


def test_onboard_character_passes_root_paths_when_given(tmp_path):
    fake_run = _fake_run_factory()
    root_paths = tmp_path / "daz-root-paths.json"

    onboard_character.onboard_character(
        "JasonCross",
        tmp_path / "export_to_blender.dsa",
        tmp_path / "JasonCross.dbz",
        tmp_path / "JasonCross_imported.blend",
        blender_exe="blender.exe",
        root_paths=root_paths,
        registry_path=tmp_path / "registry.json",
        export_fn=_fake_export_fn([]),
        stage_fn=_fake_stage_fn,
        run=fake_run,
    )

    command = fake_run.calls[0]
    assert "--root-paths" in command
    assert Path(command[command.index("--root-paths") + 1]) == root_paths.resolve()


def test_onboard_character_omits_root_paths_when_not_given(tmp_path):
    fake_run = _fake_run_factory()

    onboard_character.onboard_character(
        "JasonCross",
        tmp_path / "export_to_blender.dsa",
        tmp_path / "JasonCross.dbz",
        tmp_path / "JasonCross_imported.blend",
        blender_exe="blender.exe",
        registry_path=tmp_path / "registry.json",
        export_fn=_fake_export_fn([]),
        stage_fn=_fake_stage_fn,
        run=fake_run,
    )

    assert "--root-paths" not in fake_run.calls[0]


def test_onboard_character_raises_on_nonzero_blender_exit(tmp_path):
    fake_run = _fake_run_factory(returncode=1, stderr="boom")

    with pytest.raises(RuntimeError, match="boom"):
        onboard_character.onboard_character(
            "JasonCross",
            tmp_path / "export_to_blender.dsa",
            tmp_path / "JasonCross.dbz",
            tmp_path / "JasonCross_imported.blend",
            blender_exe="blender.exe",
            registry_path=tmp_path / "registry.json",
            export_fn=_fake_export_fn([]),
            stage_fn=_fake_stage_fn,
            run=fake_run,
        )


def test_onboard_character_raises_when_character_collection_missing(tmp_path):
    def fake_run(command, capture_output=True, text=True):
        return SimpleNamespace(returncode=0, stdout="IMPORT_RESULT {'FINISHED'}\n", stderr="")

    with pytest.raises(RuntimeError, match="CHARACTER_COLLECTION"):
        onboard_character.onboard_character(
            "JasonCross",
            tmp_path / "export_to_blender.dsa",
            tmp_path / "JasonCross.dbz",
            tmp_path / "JasonCross_imported.blend",
            blender_exe="blender.exe",
            registry_path=tmp_path / "registry.json",
            export_fn=_fake_export_fn([]),
            stage_fn=_fake_stage_fn,
            run=fake_run,
        )


def test_stage_dbz_for_fitting_copies_next_to_source_duf(tmp_path):
    import gzip
    import json

    duf_dir = tmp_path / "duf"
    duf_dir.mkdir()
    duf_path = duf_dir / "JasonCross.duf"
    duf_path.write_text("", encoding="utf-8")

    dbz_dir = tmp_path / "dbz"
    dbz_dir.mkdir()
    dbz_path = dbz_dir / "JasonCross.dbz"
    with gzip.open(dbz_path, "wt", encoding="utf-8") as handle:
        json.dump({"filepath": str(duf_path)}, handle)

    staged_path = onboard_character.stage_dbz_for_fitting(dbz_path)

    assert staged_path == duf_path.with_suffix(".dbz")
    assert staged_path.exists()


def test_stage_dbz_for_fitting_raises_when_scene_never_saved(tmp_path):
    import gzip
    import json

    dbz_path = tmp_path / "JasonCross.dbz"
    with gzip.open(dbz_path, "wt", encoding="utf-8") as handle:
        json.dump({"filepath": ""}, handle)

    with pytest.raises(RuntimeError, match="must be saved"):
        onboard_character.stage_dbz_for_fitting(dbz_path)


def test_resolve_blender_exe_prefers_explicit_override(tmp_path):
    toolchain_path = tmp_path / "toolchain.json"
    toolchain_path.write_text('{"blender": {"executable": "from-config.exe"}}', encoding="utf-8")

    assert onboard_character.resolve_blender_exe("explicit.exe", toolchain_path) == "explicit.exe"
    assert onboard_character.resolve_blender_exe(None, toolchain_path) == "from-config.exe"
