from pathlib import Path

from scarecrow_pipeline.nl_appearance import AppearanceVocabulary, PoseCritique
from scarecrow_pipeline.pose_critique import (
    RenderFailedError,
    RenderOutcome,
    run_with_critique,
)


class FakeLLMClient:
    """Returns queued dict responses in order; records every call made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, json_schema):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self._responses.pop(0)


class FakeVisionClient:
    """Returns queued PoseCritique verdicts in order; records every call made."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = []

    def critique_pose(self, description, image_bytes):
        self.calls.append({"description": description, "image_bytes": image_bytes})
        return self._verdicts.pop(0)


def _small_vocab():
    return AppearanceVocabulary(
        bones=[
            {"name": "hip", "category": "spine", "rotation_mode": "XYZ"},
            {"name": "r_forearm", "category": "arm", "rotation_mode": "XYZ"},
        ],
        facs_controls=[],
    )


def _fake_render_fn(image_path: Path, statuses=None):
    """Builds a render_fn that writes a trivial PNG-shaped file each call and
    returns RenderOutcome with the given per-call status ("ok" by default)."""
    statuses = list(statuses) if statuses is not None else None
    calls = []

    def render_fn(result, attempt):
        calls.append({"result": result, "attempt": attempt})
        status = statuses.pop(0) if statuses else "ok"
        image_path.write_bytes(b"fake-png-bytes")
        entry = {"status": status, "output_path": str(image_path)}
        if status != "ok":
            raise RenderFailedError("blender exploded")
        return RenderOutcome(entry=entry, image_path=image_path)

    render_fn.calls = calls
    return render_fn


def test_run_with_critique_passes_on_first_attempt(tmp_path):
    vocab = _small_vocab()
    translate_client = FakeLLMClient([
        {"pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}}, "expression": {"weights": {}}},
    ])
    vision_client = FakeVisionClient([
        PoseCritique(matches_description=True, critique="looks right"),
    ])
    render_fn = _fake_render_fn(tmp_path / "out.png")

    outcome = run_with_critique(
        "JasonCross",
        "leaning forward",
        translate_client=translate_client,
        vision_client=vision_client,
        render_fn=render_fn,
        vocab=vocab,
    )

    assert outcome["status"] == "ok"
    assert outcome["matches_description"] is True
    assert outcome["attempts"] == 1
    assert len(render_fn.calls) == 1


def test_run_with_critique_retries_with_corrective_context_then_passes(tmp_path):
    vocab = _small_vocab()
    translate_client = FakeLLMClient([
        {"pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}}, "expression": {"weights": {}}},
        {"pose": {"bone_rotations": {"hip": [0.2, 0.0, 0.0]}}, "expression": {"weights": {}}},
    ])
    vision_client = FakeVisionClient([
        PoseCritique(matches_description=False, critique="arms are not crossed"),
        PoseCritique(matches_description=True, critique="looks right now"),
    ])
    render_fn = _fake_render_fn(tmp_path / "out.png")

    outcome = run_with_critique(
        "JasonCross",
        "arms crossed",
        translate_client=translate_client,
        vision_client=vision_client,
        render_fn=render_fn,
        vocab=vocab,
        max_attempts=2,
    )

    assert outcome["status"] == "ok"
    assert outcome["matches_description"] is True
    assert outcome["attempts"] == 2
    # The critique from attempt 1 must be folded into attempt 2's translate call.
    assert "arms are not crossed" in translate_client.calls[1]["user_prompt"]


def test_run_with_critique_feeds_previous_ik_and_pole_targets_back_numerically(tmp_path):
    vocab = _small_vocab()
    translate_client = FakeLLMClient([
        {
            "pose": {
                "ik_targets": {"r_forearm": [0.3, -0.1, 1.2]},
                "pole_targets": {"r_forearm": [-0.3, -0.15, 0.95]},
            },
            "expression": {"weights": {}},
        },
        {
            "pose": {
                "ik_targets": {"r_forearm": [0.08, -0.03, 1.15]},
                "pole_targets": {"r_forearm": [-0.3, -0.15, 0.95]},
            },
            "expression": {"weights": {}},
        },
    ])
    vision_client = FakeVisionClient([
        PoseCritique(matches_description=False, critique="right arm is out too far, not crossed"),
        PoseCritique(matches_description=True, critique="looks right now"),
    ])
    render_fn = _fake_render_fn(tmp_path / "out.png")

    outcome = run_with_critique(
        "JasonCross",
        "arms crossed",
        translate_client=translate_client,
        vision_client=vision_client,
        render_fn=render_fn,
        vocab=vocab,
        max_attempts=2,
    )

    assert outcome["status"] == "ok"
    retry_prompt = translate_client.calls[1]["user_prompt"]
    # The retry must include the previous attempt's actual numeric targets,
    # not just the prose critique.
    assert "[0.3, -0.1, 1.2]" in retry_prompt
    assert "[-0.3, -0.15, 0.95]" in retry_prompt
    assert "ik_targets" in retry_prompt
    assert "pole_targets" in retry_prompt


def test_run_with_critique_returns_best_effort_after_exhausting_attempts(tmp_path):
    vocab = _small_vocab()
    translate_client = FakeLLMClient([
        {"pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}}, "expression": {"weights": {}}},
        {"pose": {"bone_rotations": {"hip": [0.2, 0.0, 0.0]}}, "expression": {"weights": {}}},
    ])
    vision_client = FakeVisionClient([
        PoseCritique(matches_description=False, critique="still wrong"),
        PoseCritique(matches_description=False, critique="still wrong again"),
    ])
    render_fn = _fake_render_fn(tmp_path / "out.png")

    outcome = run_with_critique(
        "JasonCross",
        "arms crossed",
        translate_client=translate_client,
        vision_client=vision_client,
        render_fn=render_fn,
        vocab=vocab,
        max_attempts=2,
    )

    assert outcome["status"] == "ok"
    assert outcome["matches_description"] is False
    assert outcome["attempts"] == 2
    assert outcome["critique"] == "still wrong again"


def test_run_with_critique_returns_render_failed_without_raising(tmp_path):
    vocab = _small_vocab()
    translate_client = FakeLLMClient([
        {"pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}}, "expression": {"weights": {}}},
    ])
    vision_client = FakeVisionClient([])
    render_fn = _fake_render_fn(tmp_path / "out.png", statuses=["failed"])

    outcome = run_with_critique(
        "JasonCross",
        "arms crossed",
        translate_client=translate_client,
        vision_client=vision_client,
        render_fn=render_fn,
        vocab=vocab,
        max_attempts=2,
    )

    assert outcome["status"] == "render_failed"
    assert outcome["attempts"] == 1
    assert len(vision_client.calls) == 0
