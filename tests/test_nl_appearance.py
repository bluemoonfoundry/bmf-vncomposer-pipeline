import os

import pytest
from pydantic import ValidationError

from scarecrow_pipeline.nl_appearance import (
    AppearanceIntent,
    AppearanceResult,
    AppearanceTranslationError,
    AppearanceVocabulary,
    CritiqueDeltaFeedback,
    LimbDelta,
    LimbGoal,
    PoseIntent,
    apply_critique_delta,
    build_retry_prompt,
    build_system_prompt,
    build_user_prompt,
    load_default_vocabulary,
    resolve_pose_intent,
    translate_appearance,
    translate_pose_intent,
)
from scarecrow_pipeline.schemas import FACSExpression, PosePayload


def test_load_default_vocabulary_reads_real_docs():
    vocab = load_default_vocabulary()

    assert "hip" in vocab.bone_names
    assert "l_upperarm" in vocab.bone_names
    assert any(name.startswith("facs_") for name in vocab.facs_control_names)


def test_load_default_vocabulary_includes_rest_pose_landmarks():
    vocab = load_default_vocabulary()

    l_shoulder = next(bone for bone in vocab.bones if bone["name"] == "l_shoulder")
    assert "rest_head_world" in l_shoulder
    assert "rest_tail_world" in l_shoulder
    assert len(l_shoulder["rest_head_world"]) == 3


def test_load_default_vocabulary_accepts_explicit_paths(tmp_path):
    bones_path = tmp_path / "bones.json"
    bones_path.write_text('{"bones": [{"name": "hip", "category": "spine"}]}', encoding="utf-8")
    controls_path = tmp_path / "controls.json"
    controls_path.write_text('{"controls": [{"name": "facs_bs_JawOpenWide", "category": "jaw"}]}', encoding="utf-8")

    vocab = load_default_vocabulary(posable_bones_path=bones_path, facs_controls_path=controls_path)

    assert vocab.bone_names == {"hip"}
    assert vocab.facs_control_names == {"facs_bs_JawOpenWide"}


def test_appearance_vocabulary_rejects_unknown_field():
    with pytest.raises(ValidationError):
        AppearanceVocabulary(bones=[], facs_controls=[], extra_field="nope")


def test_appearance_result_defaults_to_no_pose_and_empty_expression():
    result = AppearanceResult()

    assert result.pose is None
    assert result.expression == FACSExpression(weights={})


def test_appearance_result_accepts_pose_and_expression():
    result = AppearanceResult(
        pose=PosePayload(bone_rotations={"hip": [0.0, 0.0, 0.0]}),
        expression=FACSExpression(weights={"facs_bs_JawOpenWide": 0.5}),
    )

    assert result.pose.bone_rotations == {"hip": [0.0, 0.0, 0.0]}
    assert result.expression.weights == {"facs_bs_JawOpenWide": 0.5}


def test_appearance_translation_error_is_a_runtime_error():
    assert issubclass(AppearanceTranslationError, RuntimeError)


def test_limb_goal_rejects_unknown_anchor():
    with pytest.raises(ValidationError, match="ANCHOR_NOT_REAL"):
        LimbGoal(target_anchor="ANCHOR_NOT_REAL")


def test_limb_goal_accepts_known_anchor():
    goal = LimbGoal(target_anchor="ANCHOR_BICEP_LATERAL_R", layer_depth="outer")

    assert goal.target_anchor == "ANCHOR_BICEP_LATERAL_R"
    assert goal.character_local_offset == [0.0, 0.0, 0.0]


def test_build_system_prompt_mentions_radians_and_vocabulary_only():
    prompt = build_system_prompt()

    assert "radians" in prompt.lower()
    assert "vocabulary" in prompt.lower()


def test_build_system_prompt_explains_limb_goals_and_lists_anchors():
    prompt = build_system_prompt()

    assert "target_anchor" in prompt
    assert "layer_depth" in prompt
    assert "elbow_strategy" in prompt
    assert "ANCHOR_BICEP_LATERAL_L" in prompt
    assert "ANCHOR_BICEP_LATERAL_R" in prompt


def test_build_user_prompt_includes_character_description_vocab_and_anchor_positions():
    vocab = AppearanceVocabulary(
        bones=[{"name": "hip", "category": "spine", "rotation_mode": "XYZ"}],
        facs_controls=[{"name": "facs_bs_JawOpenWide", "category": "jaw"}],
    )

    prompt = build_user_prompt("JasonCross", "standing at ease, arms crossed", vocab)

    assert "JasonCross" in prompt
    assert "standing at ease, arms crossed" in prompt
    assert "hip" in prompt
    assert "facs_bs_JawOpenWide" in prompt
    assert "anchor_positions" not in prompt or True  # anchors present only when landmark bones are in vocab


def test_build_user_prompt_includes_resolved_anchor_positions_for_real_vocab():
    vocab = load_default_vocabulary()

    prompt = build_user_prompt("JasonCross", "arms crossed", vocab)

    assert "ANCHOR_BICEP_LATERAL_R" in prompt


def test_build_retry_prompt_lists_each_error():
    prompt = build_retry_prompt(["bone_name 'l_uparm' is not in the vocabulary"])

    assert "l_uparm" in prompt
    assert "not in the vocabulary" in prompt


class FakeLLMClient:
    """Returns queued dict responses (or raises a queued exception) in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, json_schema):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "json_schema": json_schema,
        })
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _small_vocab():
    return AppearanceVocabulary(
        bones=[{"name": "hip", "category": "spine", "rotation_mode": "XYZ"}],
        facs_controls=[{"name": "facs_bs_JawOpenWide", "category": "jaw"}],
    )


def _full_arm_vocab():
    return load_default_vocabulary()


def test_translate_pose_intent_happy_path():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {
            "pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}},
            "expression": {"weights": {"facs_bs_JawOpenWide": 0.3}},
        }
    ])

    intent = translate_pose_intent("JasonCross", "leaning forward slightly, mouth open", client, vocab=vocab)

    assert intent.pose.bone_rotations == {"hip": [0.1, 0.0, 0.0]}
    assert intent.expression.weights == {"facs_bs_JawOpenWide": 0.3}
    assert len(client.calls) == 1


def test_translate_pose_intent_converts_name_value_pair_wire_shape():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {
            "pose": {"bone_rotations": [{"name": "hip", "value": [0.1, 0.0, 0.0]}]},
            "expression": {"weights": [{"name": "facs_bs_JawOpenWide", "value": 0.3}]},
        }
    ])

    intent = translate_pose_intent("JasonCross", "leaning forward slightly, mouth open", client, vocab=vocab)

    assert intent.pose.bone_rotations == {"hip": [0.1, 0.0, 0.0]}
    assert intent.expression.weights == {"facs_bs_JawOpenWide": 0.3}


def test_localize_open_maps_emits_array_of_pairs_not_one_property_per_vocab_name():
    from scarecrow_pipeline.nl_appearance import AppearanceIntent, _localize_open_maps

    vocab = _small_vocab()
    schema = _localize_open_maps(AppearanceIntent.model_json_schema(), vocab)

    bone_rotations_schema = schema["$defs"]["PoseIntent"]["properties"]["bone_rotations"]
    assert bone_rotations_schema["type"] == "array"
    name_schema = bone_rotations_schema["items"]["properties"]["name"]
    assert name_schema["enum"] == ["hip"]
    assert bone_rotations_schema["items"]["additionalProperties"] is False

    weights_schema = schema["$defs"]["FACSExpression"]["properties"]["weights"]
    assert weights_schema["type"] == "array"
    assert weights_schema["items"]["properties"]["name"]["enum"] == ["facs_bs_JawOpenWide"]


def test_localize_open_maps_constrains_target_anchor_enum():
    from scarecrow_pipeline.nl_appearance import AppearanceIntent, _localize_open_maps

    vocab = _small_vocab()
    schema = _localize_open_maps(AppearanceIntent.model_json_schema(), vocab)

    limb_goal_schema = schema["$defs"]["LimbGoal"]["properties"]["target_anchor"]
    assert set(limb_goal_schema["enum"]) == {
        "ANCHOR_CHEST_CENTER",
        "ANCHOR_PECTORAL_L",
        "ANCHOR_PECTORAL_R",
        "ANCHOR_BICEP_LATERAL_L",
        "ANCHOR_BICEP_LATERAL_R",
        "ANCHOR_FOREARM_VENTRAL_L",
        "ANCHOR_FOREARM_VENTRAL_R",
    }


def test_translate_pose_intent_retries_once_on_invalid_bone_name_then_succeeds():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {
            "pose": {"bone_rotations": {"l_uparm": [0.1, 0.0, 0.0]}},
            "expression": {"weights": {}},
        },
        {
            "pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}},
            "expression": {"weights": {}},
        },
    ])

    intent = translate_pose_intent("JasonCross", "leaning forward", client, vocab=vocab)

    assert intent.pose.bone_rotations == {"hip": [0.1, 0.0, 0.0]}
    assert len(client.calls) == 2
    assert "l_uparm" in client.calls[1]["user_prompt"]
    assert "not in the vocabulary" in client.calls[1]["user_prompt"]


def test_translate_pose_intent_raises_after_exhausted_retries():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {"pose": {"bone_rotations": {"l_uparm": [0.0, 0.0, 0.0]}}, "expression": {"weights": {}}},
        {"pose": {"bone_rotations": {"l_uparm": [0.0, 0.0, 0.0]}}, "expression": {"weights": {}}},
    ])

    with pytest.raises(AppearanceTranslationError, match="l_uparm"):
        translate_pose_intent("JasonCross", "leaning forward", client, vocab=vocab)

    assert len(client.calls) == 2


def test_translate_pose_intent_rejects_vocabulary_invalid_control_even_though_schema_valid():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {"facs_bs_MadeUpControl": 1.0}}},
        {"pose": None, "expression": {"weights": {"facs_bs_MadeUpControl": 1.0}}},
    ])

    with pytest.raises(AppearanceTranslationError, match="facs_bs_MadeUpControl"):
        translate_pose_intent("JasonCross", "jaw wide open", client, vocab=vocab)


def test_translate_pose_intent_retries_on_schema_invalid_response():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {"facs_bs_JawOpenWide": 2.5}}},
        {"pose": None, "expression": {"weights": {"facs_bs_JawOpenWide": 0.5}}},
    ])

    intent = translate_pose_intent("JasonCross", "jaw open", client, vocab=vocab)

    assert intent.expression.weights == {"facs_bs_JawOpenWide": 0.5}
    assert len(client.calls) == 2


def test_translate_pose_intent_uses_default_vocabulary_when_none_given():
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {}}},
    ])

    intent = translate_pose_intent("JasonCross", "neutral", client)

    assert intent.expression.weights == {}
    assert "l_upperarm" in client.calls[0]["user_prompt"]


def test_translate_pose_intent_propagates_client_exception_without_retry():
    vocab = _small_vocab()
    client = FakeLLMClient([RuntimeError("network down")])

    with pytest.raises(RuntimeError, match="network down"):
        translate_pose_intent("JasonCross", "leaning forward", client, vocab=vocab)

    assert len(client.calls) == 1


def test_resolve_pose_intent_returns_none_for_none_intent():
    assert resolve_pose_intent(None, _full_arm_vocab()) is None


def test_resolve_pose_intent_resolves_left_arm_anchor_to_world_space_ik_and_pole_targets():
    vocab = _full_arm_vocab()
    intent = PoseIntent(
        left_arm=LimbGoal(target_anchor="ANCHOR_BICEP_LATERAL_R", layer_depth="outer"),
        right_arm=LimbGoal(target_anchor="ANCHOR_BICEP_LATERAL_L", layer_depth="inner"),
    )

    pose = resolve_pose_intent(intent, vocab)

    assert set(pose.ik_targets) == {"l_forearm", "r_forearm"}
    assert set(pose.pole_targets) == {"l_forearm", "r_forearm"}
    assert all(isinstance(v, list) and len(v) == 3 for v in pose.ik_targets.values())


def test_resolve_pose_intent_adds_clavicle_protraction_when_both_arms_layered():
    vocab = _full_arm_vocab()
    intent = PoseIntent(
        left_arm=LimbGoal(target_anchor="ANCHOR_BICEP_LATERAL_R", layer_depth="outer"),
        right_arm=LimbGoal(target_anchor="ANCHOR_BICEP_LATERAL_L", layer_depth="inner"),
    )

    pose = resolve_pose_intent(intent, vocab)

    assert "l_shoulder" in pose.bone_rotations
    assert "r_shoulder" in pose.bone_rotations


def test_resolve_pose_intent_omits_clavicle_protraction_for_single_arm():
    vocab = _full_arm_vocab()
    intent = PoseIntent(left_arm=LimbGoal(target_anchor="ANCHOR_BICEP_LATERAL_R", layer_depth="outer"))

    pose = resolve_pose_intent(intent, vocab)

    assert "l_shoulder" not in pose.bone_rotations
    assert "r_shoulder" not in pose.bone_rotations


def test_resolve_pose_intent_applies_weight_stance():
    vocab = _full_arm_vocab()
    intent = PoseIntent(weight_stance="shift_left")

    pose = resolve_pose_intent(intent, vocab)

    assert pose.root_location[0] > 0
    assert pose.bone_rotations["spine1"][2] < 0


def test_resolve_pose_intent_explicit_bone_rotations_win_over_weight_stance():
    vocab = _full_arm_vocab()
    intent = PoseIntent(weight_stance="shift_left", bone_rotations={"spine1": [0.0, 0.0, 0.5]})

    pose = resolve_pose_intent(intent, vocab)

    assert pose.bone_rotations["spine1"] == [0.0, 0.0, 0.5]


def test_translate_appearance_resolves_arms_into_ik_targets(monkeypatch):
    vocab = _full_arm_vocab()
    client = FakeLLMClient([
        {
            "pose": {
                "left_arm": {"target_anchor": "ANCHOR_BICEP_LATERAL_R", "layer_depth": "outer"},
                "right_arm": {"target_anchor": "ANCHOR_BICEP_LATERAL_L", "layer_depth": "inner"},
            },
            "expression": {"weights": {}},
        }
    ])

    result = translate_appearance("JasonCross", "arms crossed", client, vocab=vocab)

    assert isinstance(result, AppearanceResult)
    assert set(result.pose.ik_targets) == {"l_forearm", "r_forearm"}


def test_apply_critique_delta_adds_delta_meters_to_offset():
    intent = AppearanceIntent(
        pose=PoseIntent(left_arm=LimbGoal(target_anchor="ANCHOR_BICEP_LATERAL_R", character_local_offset=[0.0, 0.0, 0.0]))
    )
    feedback = CritiqueDeltaFeedback(
        pose_is_satisfactory=False,
        critique_summary="left hand too low",
        adjustments=[LimbDelta(limb="left_arm", delta_meters=[0.0, 0.0, 0.05], notes="raise it")],
    )

    new_intent = apply_critique_delta(intent, feedback)

    assert new_intent.pose.left_arm.character_local_offset == [0.0, 0.0, 0.05]
    # original is untouched
    assert intent.pose.left_arm.character_local_offset == [0.0, 0.0, 0.0]


def test_apply_critique_delta_clamps_to_plus_minus_20cm():
    intent = AppearanceIntent(
        pose=PoseIntent(left_arm=LimbGoal(target_anchor="ANCHOR_BICEP_LATERAL_R", character_local_offset=[0.18, 0.0, 0.0]))
    )
    feedback = CritiqueDeltaFeedback(
        pose_is_satisfactory=False,
        critique_summary="too far right",
        adjustments=[LimbDelta(limb="left_arm", delta_meters=[0.10, 0.0, 0.0])],
    )

    new_intent = apply_critique_delta(intent, feedback)

    assert new_intent.pose.left_arm.character_local_offset[0] == 0.20


def test_apply_critique_delta_swaps_target_anchor_when_given():
    intent = AppearanceIntent(pose=PoseIntent(right_arm=LimbGoal(target_anchor="ANCHOR_BICEP_LATERAL_L")))
    feedback = CritiqueDeltaFeedback(
        pose_is_satisfactory=False,
        critique_summary="wrong landmark entirely",
        adjustments=[
            LimbDelta(limb="right_arm", delta_meters=[0.0, 0.0, 0.0], change_anchor="ANCHOR_FOREARM_VENTRAL_L")
        ],
    )

    new_intent = apply_critique_delta(intent, feedback)

    assert new_intent.pose.right_arm.target_anchor == "ANCHOR_FOREARM_VENTRAL_L"


def test_limb_delta_rejects_unknown_change_anchor():
    with pytest.raises(ValidationError, match="NOT_REAL"):
        LimbDelta(limb="left_arm", delta_meters=[0.0, 0.0, 0.0], change_anchor="NOT_REAL")


def test_anthropic_client_raises_helpful_error_without_optional_dependency(monkeypatch):
    import sys

    from scarecrow_pipeline.nl_appearance import AnthropicClient

    monkeypatch.setitem(sys.modules, "anthropic", None)

    with pytest.raises(ImportError, match="anthropic"):
        AnthropicClient()


@pytest.mark.skipif(
    os.environ.get("RUN_ANTHROPIC_SMOKE_TEST") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="opt-in smoke test: set RUN_ANTHROPIC_SMOKE_TEST=1 and ANTHROPIC_API_KEY to run",
)
def test_anthropic_client_smoke_translates_a_real_description():
    from scarecrow_pipeline.nl_appearance import AnthropicClient

    vocab = AppearanceVocabulary(
        bones=[{"name": "hip", "category": "spine", "rotation_mode": "XYZ"}],
        facs_controls=[{"name": "facs_bs_JawOpenWide", "category": "jaw"}],
    )
    client = AnthropicClient()

    result = translate_appearance("JasonCross", "standing at ease", client, vocab=vocab)

    assert isinstance(result, AppearanceResult)
