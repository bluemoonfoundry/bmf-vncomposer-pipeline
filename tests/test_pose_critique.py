from pathlib import Path

from scarecrow_pipeline.nl_appearance import AppearanceVocabulary, CritiqueDeltaFeedback, LimbDelta
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
    """Returns queued CritiqueDeltaFeedback verdicts in order; records every call made."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = []

    def critique_pose(self, description, image_bytes, current_intent):
        self.calls.append({"description": description, "image_bytes": image_bytes, "current_intent": current_intent})
        return self._verdicts.pop(0)


def _small_vocab():
    return AppearanceVocabulary(
        bones=[
            {"name": "hip", "category": "spine", "rotation_mode": "XYZ"},
            {"name": "r_forearm", "category": "arm", "rotation_mode": "XYZ",
             "rest_head_world": [-0.3334, 0.0495, 1.1769], "rest_tail_world": [-0.45, 0.05, 0.95]},
            {"name": "r_upperarm", "category": "arm", "rotation_mode": "XYZ",
             "rest_head_world": [-0.1774, 0.0524, 1.3542], "rest_tail_world": [-0.3334, 0.0495, 1.1769]},
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
        CritiqueDeltaFeedback(pose_is_satisfactory=True, critique_summary="looks right"),
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
    # Only translated once -- no re-translation needed to pass on attempt 1.
    assert len(translate_client.calls) == 1


def test_run_with_critique_applies_numeric_delta_without_retranslating_then_passes(tmp_path):
    vocab = _small_vocab()
    translate_client = FakeLLMClient([
        {
            "pose": {"right_arm": {"target_anchor": "ANCHOR_BICEP_LATERAL_R", "character_local_offset": [0.0, 0.0, 0.0]}},
            "expression": {"weights": {}},
        },
    ])
    vision_client = FakeVisionClient([
        CritiqueDeltaFeedback(
            pose_is_satisfactory=False,
            critique_summary="right arm too low",
            adjustments=[LimbDelta(limb="right_arm", delta_meters=[0.0, 0.0, 0.05], notes="raise it")],
        ),
        CritiqueDeltaFeedback(pose_is_satisfactory=True, critique_summary="looks right now"),
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
    # The pose-generation LLM is called exactly once -- the second attempt's
    # pose comes from apply_critique_delta, not a fresh translate call.
    assert len(translate_client.calls) == 1
    # attempt 2's resolved ik_target must have moved up (+z) from attempt 1's.
    attempt_1_target = render_fn.calls[0]["result"].pose.ik_targets["r_forearm"]
    attempt_2_target = render_fn.calls[1]["result"].pose.ik_targets["r_forearm"]
    assert attempt_2_target[2] > attempt_1_target[2]


def test_run_with_critique_returns_best_effort_after_exhausting_attempts(tmp_path):
    vocab = _small_vocab()
    translate_client = FakeLLMClient([
        {
            "pose": {"right_arm": {"target_anchor": "ANCHOR_BICEP_LATERAL_R"}},
            "expression": {"weights": {}},
        },
    ])
    vision_client = FakeVisionClient([
        CritiqueDeltaFeedback(
            pose_is_satisfactory=False,
            critique_summary="still wrong",
            adjustments=[LimbDelta(limb="right_arm", delta_meters=[0.0, 0.0, 0.02])],
        ),
        CritiqueDeltaFeedback(
            pose_is_satisfactory=False,
            critique_summary="still wrong again",
            adjustments=[LimbDelta(limb="right_arm", delta_meters=[0.0, 0.0, 0.02])],
        ),
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
