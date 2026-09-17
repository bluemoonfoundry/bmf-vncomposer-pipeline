"""Translate a natural-language appearance description into a validated
PosePayload/FACSExpression pair, via a provider-agnostic LLM client.

DAZ Studio is only ever used to source a character once (see
scripts/onboard_character.py); all posing and expression happen in
Blender (blender/worker.py's apply_pose/apply_expression) so they can be
driven by natural language. This module is the translation step
scarecrow_pipeline/scene_tags.py calls "a separate, out-of-scope LLM
step" -- see
docs/superpowers/specs/2026-09-16-nl-appearance-translator-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scarecrow_pipeline.schemas import FACSExpression, PosePayload

MAX_ATTEMPTS = 2

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSABLE_BONES_PATH = REPO_ROOT / "docs" / "posable_bones.json"
DEFAULT_FACS_CONTROLS_PATH = REPO_ROOT / "docs" / "facs_controls.json"


class AppearanceTranslationError(RuntimeError):
    """Raised when the LLM's output cannot be validated after MAX_ATTEMPTS tries."""


class AppearanceResult(BaseModel):
    """One description's translated appearance.

    Assign both fields straight onto a CharacterPlacement/RenderRequest
    (scarecrow_pipeline/schemas.py), which carry pose and expression as
    the same two side-by-side fields.
    """

    model_config = ConfigDict(extra="forbid")
    pose: PosePayload | None = None
    expression: FACSExpression = Field(default_factory=FACSExpression)


class LLMClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> dict:
        """Return a dict parsed from the model's structured-output response,
        constrained to json_schema. Raises on a provider/network failure;
        does not itself validate the dict against the schema."""
        ...


class AppearanceVocabulary(BaseModel):
    """The bone/FACS-control name vocabulary the LLM is constrained to.

    Shared across all onboarded characters (same Genesis-based rig/FACS
    set) -- see docs/posable_bones.json and docs/facs_controls.json,
    regenerated per-character via blender/dump_posable_bones.py and
    blender/dump_facs_controls.py.
    """

    model_config = ConfigDict(extra="forbid")
    bones: list[dict]
    facs_controls: list[dict]

    @property
    def bone_names(self) -> set[str]:
        return {bone["name"] for bone in self.bones}

    @property
    def facs_control_names(self) -> set[str]:
        return {control["name"] for control in self.facs_controls}


def load_default_vocabulary(
    posable_bones_path: Path = DEFAULT_POSABLE_BONES_PATH,
    facs_controls_path: Path = DEFAULT_FACS_CONTROLS_PATH,
) -> AppearanceVocabulary:
    bones = json.loads(Path(posable_bones_path).read_text(encoding="utf-8"))["bones"]
    facs_controls = json.loads(Path(facs_controls_path).read_text(encoding="utf-8"))["controls"]
    return AppearanceVocabulary(bones=bones, facs_controls=facs_controls)


def build_system_prompt() -> str:
    return (
        "You translate a natural-language character appearance description "
        "into precise pose and facial-expression control values for a "
        "Diffeomorphic-imported DAZ character rig in Blender.\n\n"
        "Respond with a single JSON object matching the given schema:\n"
        "- pose.bone_rotations: a mapping of bone name to an [x, y, z] Euler "
        "rotation IN RADIANS. Each bone has its OWN native rotation axis "
        "order (see each bone's rotation_mode in the provided vocabulary) "
        "-- do not assume XYZ order; provide the three angles as if applied "
        "in that bone's own order.\n"
        "- pose.ik_targets and pose.look_at_target are optional -- omit "
        "them entirely unless the description specifically calls for an IK "
        "target or a gaze direction.\n"
        "- expression.weights: a mapping of FACS/morph control name to a "
        "weight in [0.0, 1.0].\n"
        "- Only use bone and control names that appear in the provided "
        "vocabulary, exactly as spelled there. Never invent a name.\n"
        "- Omit any field you have no information for rather than guessing "
        "a value."
    )


def build_user_prompt(character: str, description: str, vocab: AppearanceVocabulary) -> str:
    return (
        f"Character: {character}\n"
        f"Description: {description}\n\n"
        f"Available pose bones (JSON):\n{json.dumps(vocab.bones)}\n\n"
        f"Available expression controls (JSON):\n{json.dumps(vocab.facs_controls)}"
    )


def build_retry_prompt(errors: list[str]) -> str:
    joined = "\n".join(f"- {error}" for error in errors)
    return (
        "Your previous response was invalid:\n"
        f"{joined}\n\n"
        "Respond again with a corrected JSON object matching the same "
        "schema, using only bone/control names from the vocabulary already "
        "provided."
    )


def _vocabulary_errors(result: AppearanceResult, vocab: AppearanceVocabulary) -> list[str]:
    errors = []
    bone_names = vocab.bone_names
    facs_control_names = vocab.facs_control_names
    if result.pose is not None:
        for bone_name in result.pose.bone_rotations:
            if bone_name not in bone_names:
                errors.append(f"bone_name {bone_name!r} is not in the vocabulary")
        for bone_name in result.pose.ik_targets:
            if bone_name not in bone_names:
                errors.append(f"ik_targets bone_name {bone_name!r} is not in the vocabulary")
    for control_name in result.expression.weights:
        if control_name not in facs_control_names:
            errors.append(f"expression control {control_name!r} is not in the vocabulary")
    return errors


def translate_appearance(
    character: str,
    description: str,
    client: LLMClient,
    *,
    vocab: AppearanceVocabulary | None = None,
) -> AppearanceResult:
    vocab = vocab or load_default_vocabulary()
    schema = AppearanceResult.model_json_schema()
    system_prompt = build_system_prompt()
    base_user_prompt = build_user_prompt(character, description, vocab)

    errors: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        # LLMClient.complete_json is single-turn (no conversation history), so
        # a retry is folded into one user-turn prompt rather than the design
        # spec's literal system/user/assistant/user multi-turn shape.
        prompt = base_user_prompt if attempt == 0 else base_user_prompt + "\n\n" + build_retry_prompt(errors)
        raw = client.complete_json(system_prompt, prompt, schema)
        try:
            result = AppearanceResult.model_validate(raw)
        except ValidationError as exc:
            errors = [str(exc)]
            continue
        errors = _vocabulary_errors(result, vocab)
        if not errors:
            return result

    raise AppearanceTranslationError(
        f"Could not produce a valid appearance for {character!r} after "
        f"{MAX_ATTEMPTS} attempt(s): " + "; ".join(errors)
    )


class AnthropicClient:
    """LLMClient backed by the Anthropic Messages API's native structured
    outputs (`output_config.format`), not forced tool-use -- this call has
    no tool surface of its own, just a JSON Schema-constrained text response,
    which is the currently recommended mechanism for pure JSON extraction.

    Requires the optional 'llm' dependency group
    (`pip install -e '.[llm]'`) and an ANTHROPIC_API_KEY environment
    variable. Never imported at module scope by nl_appearance.py or
    exercised by the default test suite -- see
    test_anthropic_client_raises_helpful_error_without_optional_dependency.

    Defaults to the most capable current model and a non-streaming-safe
    max_tokens, per current Anthropic guidance: prefer the top-tier model
    and let the caller downgrade for cost, rather than pre-downgrading.
    """

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 16000):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicClient requires the 'anthropic' package -- install "
                "with `pip install -e '.[llm]'`"
            ) from exc
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def complete_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> dict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": json_schema}},
        )
        # output_config.format guarantees the first content block is text
        # containing valid JSON matching json_schema.
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise RuntimeError("Anthropic response contained no text block")
        return json.loads(text)
