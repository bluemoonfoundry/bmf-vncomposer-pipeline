import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import apply_appearance  # noqa: E402

from scarecrow_pipeline.nl_appearance import AppearanceVocabulary  # noqa: E402
from scarecrow_pipeline.registry import CharacterRecord, Registry  # noqa: E402


class FakeLLMClient:
    """Returns queued dict responses in order; records every call made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, json_schema):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "json_schema": json_schema,
        })
        return self._responses.pop(0)


def _fake_run_factory(returncode=0, stderr=""):
    calls = []

    def fake_run(command, capture_output=True, text=True):
        calls.append(command)
        return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)

    fake_run.calls = calls
    return fake_run


def _small_vocab():
    return AppearanceVocabulary(
        bones=[{"name": "hip", "category": "spine", "rotation_mode": "XYZ"}],
        facs_controls=[{"name": "facs_bs_JawOpenWide", "category": "jaw"}],
    )


def _write_registry(registry_path, master_blend="somewhere/JasonCross_imported.blend"):
    registry = Registry()
    registry.upsert_character(CharacterRecord(
        name="JasonCross",
        master_blend=master_blend,
        collection="Jasoncross_v1",
    ))
    registry.save(registry_path)


def test_apply_appearance_writes_translated_pose_and_expression_into_request(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path)
    client = FakeLLMClient([
        {
            "pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}},
            "expression": {"weights": {"facs_bs_JawOpenWide": 0.4}},
        }
    ])
    fake_run = _fake_run_factory()
    output_path = tmp_path / "out.png"

    entry = apply_appearance.apply_appearance(
        "JasonCross",
        "standing at ease, jaw slightly open",
        str(output_path),
        client=client,
        blender_exe="blender.exe",
        registry_path=registry_path,
        vocab=_small_vocab(),
        run=fake_run,
    )

    assert entry["status"] == "ok"
    request_path = Path(entry["request_path"])
    request_dict = json.loads(request_path.read_text(encoding="utf-8"))
    # RenderRequest.character is the master_blend collection name, not the
    # registry's human-facing key -- same convention as sprite_set.py.
    assert request_dict["character"] == "Jasoncross_v1"
    assert request_dict["pose"]["bone_rotations"] == {"hip": [0.1, 0.0, 0.0]}
    assert request_dict["expression"]["weights"] == {"facs_bs_JawOpenWide": 0.4}


def test_apply_appearance_invokes_blender_with_worker_script_and_request(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path)
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {}}},
    ])
    fake_run = _fake_run_factory()
    output_path = tmp_path / "out.png"

    apply_appearance.apply_appearance(
        "JasonCross",
        "neutral",
        str(output_path),
        client=client,
        blender_exe="blender.exe",
        registry_path=registry_path,
        vocab=_small_vocab(),
        run=fake_run,
    )

    command = fake_run.calls[0]
    assert command[0] == "blender.exe"
    assert "--background" in command
    assert str(apply_appearance.WORKER_SCRIPT.resolve()) in command
    request_index = command.index("--request") + 1
    assert Path(command[request_index]).exists()


def test_apply_appearance_raises_for_unregistered_character(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path)
    client = FakeLLMClient([])
    fake_run = _fake_run_factory()

    with pytest.raises(ValueError, match="AbbyMorgan"):
        apply_appearance.apply_appearance(
            "AbbyMorgan",
            "neutral",
            str(tmp_path / "out.png"),
            client=client,
            blender_exe="blender.exe",
            registry_path=registry_path,
            vocab=_small_vocab(),
            run=fake_run,
        )

    assert len(fake_run.calls) == 0


def test_apply_appearance_reports_failed_status_on_nonzero_blender_exit(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path)
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {}}},
    ])
    fake_run = _fake_run_factory(returncode=1, stderr="blender exploded")

    entry = apply_appearance.apply_appearance(
        "JasonCross",
        "neutral",
        str(tmp_path / "out.png"),
        client=client,
        blender_exe="blender.exe",
        registry_path=registry_path,
        vocab=_small_vocab(),
        run=fake_run,
    )

    assert entry["status"] == "failed"
    assert "blender exploded" in entry["stderr"]


class FakeVisionClient:
    """Returns queued CritiqueDeltaFeedback verdicts in order."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = []

    def critique_pose(self, description, image_bytes, current_intent):
        self.calls.append({"description": description, "image_bytes": image_bytes})
        return self._verdicts.pop(0)


def test_apply_appearance_with_vision_client_critiques_and_returns_manifest(tmp_path):
    """When vision_client is supplied, apply_appearance() delegates to the
    pose_critique loop instead of its single-shot path -- see scarecrow-57h."""
    from scarecrow_pipeline.nl_appearance import CritiqueDeltaFeedback

    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path)
    client = FakeLLMClient([
        {"pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}}, "expression": {"weights": {}}},
    ])
    vision_client = FakeVisionClient([CritiqueDeltaFeedback(pose_is_satisfactory=True, critique_summary="matches")])
    fake_run = _fake_run_factory()
    output_path = tmp_path / "out.png"
    # The real Blender subprocess writes output_path; _fake_run_factory only
    # simulates the process exit code, so the render_fn's read_bytes() needs
    # a real file already there.
    output_path.write_bytes(b"fake-png-bytes")

    entry = apply_appearance.apply_appearance(
        "JasonCross",
        "standing at ease, arms crossed",
        str(output_path),
        client=client,
        blender_exe="blender.exe",
        registry_path=registry_path,
        vocab=_small_vocab(),
        run=fake_run,
        vision_client=vision_client,
    )

    assert entry["status"] == "ok"
    assert entry["matches_description"] is True
    assert entry["attempts"] == 1
    assert len(vision_client.calls) == 1


@pytest.mark.skipif(
    os.environ.get("RUN_BLENDER_SMOKE_TEST") != "1",
    reason="opt-in smoke test: set RUN_BLENDER_SMOKE_TEST=1 to run (launches real Blender)",
)
def test_apply_appearance_smoke_updates_pose_in_real_blender():
    """Not run by default -- actually launches Blender headlessly against the
    real onboarded JasonCross master_blend and confirms the pose/expression
    translate_appearance() produced was genuinely applied and rendered.

    Uses FakeLLMClient (no ANTHROPIC_API_KEY needed) but the real blender.exe
    from config/toolchain.json and the real docs/posable_bones.json +
    docs/facs_controls.json vocabulary, so this is the one test in the suite
    that proves the whole NL-description -> Blender-pose path actually works.
    """
    from scarecrow_pipeline.nl_appearance import load_default_vocabulary

    client = FakeLLMClient([
        {
            "pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}},
            "expression": {"weights": {"facs_bs_JawOpenWide": 0.5}},
        }
    ])
    blender_exe = apply_appearance.resolve_blender_exe(None)
    output_path = REPO_ROOT / "artifacts" / "nl_appearance_smoke_test.png"

    entry = apply_appearance.apply_appearance(
        "JasonCross",
        "leaning forward slightly, jaw half open",
        str(output_path),
        client=client,
        blender_exe=blender_exe,
        vocab=load_default_vocabulary(),
        render_samples=1,
    )

    assert entry["status"] == "ok", entry.get("stderr")
    assert output_path.exists()
    assert Path(str(output_path) + ".blend").exists()
