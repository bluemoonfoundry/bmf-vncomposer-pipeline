import os

import pytest

from scarecrow_pipeline.nl_appearance import (
    AppearanceVocabulary,
    AppearanceResult,
    AppearanceTranslationError,
    load_default_vocabulary,
)
from scarecrow_pipeline.schemas import FACSExpression, PosePayload


def test_load_default_vocabulary_reads_real_docs():
    vocab = load_default_vocabulary()

    assert "hip" in vocab.bone_names
    assert "l_upperarm" in vocab.bone_names
    assert any(name.startswith("facs_") for name in vocab.facs_control_names)


def test_load_default_vocabulary_includes_rest_pose_landmarks():
    """scarecrow-57h: the LLM needs a spatial reference frame to compute
    ik_targets/pole_targets -- every bone must carry its rest-pose head/tail
    position in the same world-space coordinate system those fields use."""
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
    import pytest
    from pydantic import ValidationError

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


from scarecrow_pipeline.nl_appearance import (
    build_retry_prompt,
    build_system_prompt,
    build_user_prompt,
)


def test_build_system_prompt_mentions_radians_and_vocabulary_only():
    prompt = build_system_prompt()

    assert "radians" in prompt.lower()
    assert "vocabulary" in prompt.lower()


def test_build_system_prompt_explains_ik_and_pole_targets():
    prompt = build_system_prompt()

    assert "pole_targets" in prompt
    assert "ik_targets" in prompt
    assert "rest_head_world" in prompt


def test_build_user_prompt_includes_character_description_and_vocab():
    vocab = AppearanceVocabulary(
        bones=[{"name": "hip", "category": "spine"}],
        facs_controls=[{"name": "facs_bs_JawOpenWide", "category": "jaw"}],
    )

    prompt = build_user_prompt("JasonCross", "standing at ease, arms crossed", vocab)

    assert "JasonCross" in prompt
    assert "standing at ease, arms crossed" in prompt
    assert "hip" in prompt
    assert "facs_bs_JawOpenWide" in prompt


def test_build_retry_prompt_lists_each_error():
    prompt = build_retry_prompt(["bone_name 'l_uparm' is not in the vocabulary"])

    assert "l_uparm" in prompt
    assert "not in the vocabulary" in prompt


from scarecrow_pipeline.nl_appearance import translate_appearance


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


def test_translate_appearance_happy_path():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {
            "pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}},
            "expression": {"weights": {"facs_bs_JawOpenWide": 0.3}},
        }
    ])

    result = translate_appearance("JasonCross", "leaning forward slightly, mouth open", client, vocab=vocab)

    assert result.pose.bone_rotations == {"hip": [0.1, 0.0, 0.0]}
    assert result.expression.weights == {"facs_bs_JawOpenWide": 0.3}
    assert len(client.calls) == 1


def test_translate_appearance_converts_name_value_pair_wire_shape():
    """The schema sent to the LLM represents bone_rotations/ik_targets/
    weights as arrays of {name, value} pairs (see _localize_open_maps),
    not dicts -- a real provider response comes back in that shape and
    must be converted before AppearanceResult validation."""
    vocab = _small_vocab()
    client = FakeLLMClient([
        {
            "pose": {"bone_rotations": [{"name": "hip", "value": [0.1, 0.0, 0.0]}]},
            "expression": {"weights": [{"name": "facs_bs_JawOpenWide", "value": 0.3}]},
        }
    ])

    result = translate_appearance("JasonCross", "leaning forward slightly, mouth open", client, vocab=vocab)

    assert result.pose.bone_rotations == {"hip": [0.1, 0.0, 0.0]}
    assert result.expression.weights == {"facs_bs_JawOpenWide": 0.3}


def test_localize_open_maps_emits_array_of_pairs_not_one_property_per_vocab_name():
    """Regression test for the "compiled grammar is too large" failure hit
    against the real API with the production vocabulary (143 bones, 446
    FACS controls): the schema must stay small regardless of vocabulary
    size, so bone_rotations/weights/ik_targets become a single array
    schema with an enum, not one named property per vocabulary entry."""
    from scarecrow_pipeline.nl_appearance import AppearanceResult, _localize_open_maps

    vocab = _small_vocab()
    schema = _localize_open_maps(AppearanceResult.model_json_schema(), vocab)

    bone_rotations_schema = schema["$defs"]["PosePayload"]["properties"]["bone_rotations"]
    assert bone_rotations_schema["type"] == "array"
    name_schema = bone_rotations_schema["items"]["properties"]["name"]
    assert name_schema["enum"] == ["hip"]
    assert bone_rotations_schema["items"]["additionalProperties"] is False

    weights_schema = schema["$defs"]["FACSExpression"]["properties"]["weights"]
    assert weights_schema["type"] == "array"
    assert weights_schema["items"]["properties"]["name"]["enum"] == ["facs_bs_JawOpenWide"]


def test_translate_appearance_converts_pole_targets_name_value_pair_wire_shape():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {
            "pose": {
                "ik_targets": [{"name": "hip", "value": [0.1, 0.2, 0.3]}],
                "pole_targets": [{"name": "hip", "value": [0.0, -0.1, 0.2]}],
            },
            "expression": {"weights": {}},
        }
    ])

    result = translate_appearance("JasonCross", "reaching forward", client, vocab=vocab)

    assert result.pose.ik_targets == {"hip": [0.1, 0.2, 0.3]}
    assert result.pose.pole_targets == {"hip": [0.0, -0.1, 0.2]}


def test_localize_open_maps_localizes_pole_targets_too():
    from scarecrow_pipeline.nl_appearance import AppearanceResult, _localize_open_maps

    vocab = _small_vocab()
    schema = _localize_open_maps(AppearanceResult.model_json_schema(), vocab)

    pole_targets_schema = schema["$defs"]["PosePayload"]["properties"]["pole_targets"]
    assert pole_targets_schema["type"] == "array"
    assert pole_targets_schema["items"]["properties"]["name"]["enum"] == ["hip"]


def test_translate_appearance_rejects_pole_target_with_no_matching_ik_target():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {
            "pose": {"pole_targets": {"hip": [0.0, 0.0, 0.0]}},
            "expression": {"weights": {}},
        },
        {
            "pose": {},
            "expression": {"weights": {}},
        },
    ])

    result = translate_appearance("JasonCross", "reaching forward", client, vocab=vocab)

    assert result.pose.pole_targets == {}
    assert len(client.calls) == 2
    assert "pole_targets" in client.calls[1]["user_prompt"]
    assert "no matching ik_targets" in client.calls[1]["user_prompt"]


def test_translate_appearance_rejects_unknown_pole_target_bone_name():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {
            "pose": {
                "ik_targets": {"l_uparm": [0.0, 0.0, 0.0]},
                "pole_targets": {"l_uparm": [0.0, 0.0, 0.0]},
            },
            "expression": {"weights": {}},
        },
        {"pose": {}, "expression": {"weights": {}}},
    ])

    translate_appearance("JasonCross", "reaching forward", client, vocab=vocab)

    assert "pole_targets bone_name 'l_uparm' is not in the vocabulary" in client.calls[1]["user_prompt"]


def test_translate_appearance_retries_once_on_invalid_bone_name_then_succeeds():
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

    result = translate_appearance("JasonCross", "leaning forward", client, vocab=vocab)

    assert result.pose.bone_rotations == {"hip": [0.1, 0.0, 0.0]}
    assert len(client.calls) == 2
    assert "l_uparm" in client.calls[1]["user_prompt"]
    assert "not in the vocabulary" in client.calls[1]["user_prompt"]


def test_translate_appearance_raises_after_exhausted_retries():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {"pose": {"bone_rotations": {"l_uparm": [0.0, 0.0, 0.0]}}, "expression": {"weights": {}}},
        {"pose": {"bone_rotations": {"l_uparm": [0.0, 0.0, 0.0]}}, "expression": {"weights": {}}},
    ])

    import pytest
    from scarecrow_pipeline.nl_appearance import AppearanceTranslationError

    with pytest.raises(AppearanceTranslationError, match="l_uparm"):
        translate_appearance("JasonCross", "leaning forward", client, vocab=vocab)

    assert len(client.calls) == 2


def test_translate_appearance_rejects_vocabulary_invalid_control_even_though_schema_valid():
    """A response can be perfectly valid PosePayload/FACSExpression JSON --
    Pydantic alone has no way to know 'facs_bs_MadeUpControl' isn't real."""
    vocab = _small_vocab()
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {"facs_bs_MadeUpControl": 1.0}}},
        {"pose": None, "expression": {"weights": {"facs_bs_MadeUpControl": 1.0}}},
    ])

    import pytest
    from scarecrow_pipeline.nl_appearance import AppearanceTranslationError

    with pytest.raises(AppearanceTranslationError, match="facs_bs_MadeUpControl"):
        translate_appearance("JasonCross", "jaw wide open", client, vocab=vocab)


def test_translate_appearance_retries_on_schema_invalid_response():
    """A response that fails Pydantic validation entirely (e.g. a weight out
    of [0,1]) also triggers the retry path, not just vocabulary mismatches."""
    vocab = _small_vocab()
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {"facs_bs_JawOpenWide": 2.5}}},
        {"pose": None, "expression": {"weights": {"facs_bs_JawOpenWide": 0.5}}},
    ])

    result = translate_appearance("JasonCross", "jaw open", client, vocab=vocab)

    assert result.expression.weights == {"facs_bs_JawOpenWide": 0.5}
    assert len(client.calls) == 2


def test_translate_appearance_uses_default_vocabulary_when_none_given():
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {}}},
    ])

    result = translate_appearance("JasonCross", "neutral", client)

    assert result.expression.weights == {}
    # The default vocabulary (docs/posable_bones.json) is much larger than
    # the 1-bone fixture used elsewhere in this file -- a real bone name
    # proves load_default_vocabulary() was used, not an empty vocabulary.
    assert "l_upperarm" in client.calls[0]["user_prompt"]


def test_translate_appearance_propagates_client_exception_without_retry():
    """A client/provider-level failure (e.g. a network error) is not a
    validation failure -- it must propagate immediately on the first
    attempt rather than being swallowed into a retry."""
    vocab = _small_vocab()
    client = FakeLLMClient([RuntimeError("network down")])

    import pytest

    with pytest.raises(RuntimeError, match="network down"):
        translate_appearance("JasonCross", "leaning forward", client, vocab=vocab)

    assert len(client.calls) == 1


def test_anthropic_client_raises_helpful_error_without_optional_dependency(monkeypatch):
    """This repo's default install does not include the 'anthropic' package
    (it's an optional extra) -- constructing AnthropicClient without it
    installed must fail with a clear message, not a bare ModuleNotFoundError
    from deep inside the client.

    Setting sys.modules["anthropic"] = None forces the next `import anthropic`
    to raise ImportError, regardless of whether the real package happens to
    be installed in this environment (e.g. as a transitive dependency of an
    unrelated package) -- this makes the test deterministic everywhere.
    """
    import sys

    import pytest

    from scarecrow_pipeline.nl_appearance import AnthropicClient

    monkeypatch.setitem(sys.modules, "anthropic", None)

    with pytest.raises(ImportError, match="anthropic"):
        AnthropicClient()


@pytest.mark.skipif(
    os.environ.get("RUN_ANTHROPIC_SMOKE_TEST") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="opt-in smoke test: set RUN_ANTHROPIC_SMOKE_TEST=1 and ANTHROPIC_API_KEY to run",
)
def test_anthropic_client_smoke_translates_a_real_description():
    """Not run by default -- makes one real Anthropic API call.

    Proves the $ref/$defs-bearing, vocabulary-localized JSON Schema built
    by translate_appearance (see _localize_open_maps) is actually accepted
    as output_config.format.schema by the real API, which no other test in
    this file verifies (FakeLLMClient never touches the real API).
    """
    from scarecrow_pipeline.nl_appearance import AnthropicClient, translate_appearance

    vocab = AppearanceVocabulary(
        bones=[{"name": "hip", "category": "spine", "rotation_mode": "XYZ"}],
        facs_controls=[{"name": "facs_bs_JawOpenWide", "category": "jaw"}],
    )
    client = AnthropicClient()

    result = translate_appearance("JasonCross", "standing at ease", client, vocab=vocab)

    assert isinstance(result, AppearanceResult)
