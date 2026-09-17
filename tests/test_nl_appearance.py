from pathlib import Path

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
