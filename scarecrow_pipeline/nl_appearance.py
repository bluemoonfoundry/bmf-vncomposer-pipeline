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

import copy
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


class PoseCritique(BaseModel):
    """Verdict from a vision-model critique of a rendered pose against its
    natural-language description -- see VisionCritiqueClient/pose_critique.py."""

    model_config = ConfigDict(extra="forbid")
    matches_description: bool
    critique: str


class VisionCritiqueClient(Protocol):
    def critique_pose(self, description: str, image_bytes: bytes) -> PoseCritique:
        """Return a critique of a rendered PNG against its natural-language
        description. Raises on a provider/network failure."""
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
        "- pose.bone_rotations: a list of {name, value} entries, one per "
        "bone you're setting, where name is the bone name and value is an "
        "[x, y, z] Euler rotation IN RADIANS, applied directly as that "
        "bone's rotation_euler.x/.y/.z -- value[0] is always the X-axis "
        "angle, value[1] always Y, value[2] always Z. Each bone's "
        "rotation_mode in the provided vocabulary (e.g. \"YZX\") only "
        "tells you the ORDER Blender composes those three axis rotations "
        "into the bone's final orientation, which matters for figuring "
        "out what the bone will actually do -- it does NOT change which "
        "position in the array holds which axis. Never reorder the "
        "triplet to match rotation_mode. Prefer bone_rotations only for "
        "spine/neck/torso/twist bones, or any bone with no clear spatial "
        "target -- for a limb whose pose has a spatial goal, use ik_targets "
        "and pole_targets instead (see below). Never set both "
        "bone_rotations and an ik_targets entry for the same bone: the IK "
        "constraint is evaluated after the manual rotation and will "
        "silently override it, so the two fight each other and only the "
        "IK result survives.\n"
        "- pose.ik_targets: a list of {name, value} entries, one per limb "
        "you're posing with inverse kinematics, where name is the "
        "end-effector bone (e.g. \"r_forearm\", \"l_forearm\") and value is "
        "an [x, y, z] WORLD-SPACE point for that bone's tip to reach for. "
        "This drives a 2-bone chain (e.g. r_upperarm+r_forearm together) "
        "toward the target -- you do not need to compute individual joint "
        "angles yourself, only where the hand/foot should end up in space. "
        "This is almost always the right tool for arm/leg gestures with a "
        "spatial goal (crossing arms, hand on hip, reaching for something), "
        "since guessing raw multi-joint Euler angles for such gestures is "
        "unreliable.\n"
        "- pose.pole_targets: a list of {name, value} entries, SAME name "
        "as the ik_targets entry it belongs to, where value is an [x, y, z] "
        "WORLD-SPACE point that the elbow/knee should bend toward. A "
        "2-bone IK chain's bend direction is otherwise ambiguous/unstable, "
        "so always provide a pole_targets entry alongside every "
        "ik_targets entry for a limb. Place the pole roughly where the "
        "elbow/knee itself should point away from the body -- e.g. for an "
        "arm bending forward and across the chest, the pole point should "
        "be in front of and below the shoulder, not out to the side or "
        "behind. A pole_targets entry with no matching ik_targets entry is "
        "invalid.\n"
        "- Every bone in the provided vocabulary includes rest_head_world "
        "and rest_tail_world: that bone's rest-pose head/tail position, in "
        "the SAME world-space coordinate system as ik_targets, "
        "pole_targets, and look_at_target. Use these as landmarks -- e.g. "
        "spine4 (upper chest/collar root), l_shoulder, r_shoulder (and "
        "their _tail_world, which is further out at the actual point of "
        "the shoulder, not the spine) -- to reason about where a hand or "
        "foot target should actually go, rather than guessing coordinates "
        "blind. For example, if spine4.rest_head_world is [0, 0.05, 1.27], "
        "l_shoulder.rest_tail_world is [0.15, 0.07, 1.38], and "
        "r_shoulder.rest_tail_world is [-0.15, 0.06, 1.36], then \"arms "
        "crossed over the chest\" means: hands rest FLAT AGAINST the "
        "ribcage, tucked between the body's centerline and the opposite "
        "shoulder tip -- roughly HALFWAY between x=0 and the opposite "
        "shoulder tip's x, not out at the shoulder tip itself or beyond "
        "it. The right arm's end effector (ik_targets[\"r_forearm\"]) "
        "might land around [0.08, -0.03, 1.15] (about halfway toward "
        "l_shoulder's x, a little in front of the ribcage on the "
        "front-facing axis, resting at lower-chest/upper-abdomen height, "
        "well below shoulder height) with pole_targets[\"r_forearm\"] "
        "placed in front of and below the right elbow's rest position "
        "(e.g. [-0.30, -0.15, 0.95]) so the forearm bends across the body "
        "rather than swinging out to the side or lifting up; "
        "symmetrically, ik_targets[\"l_forearm\"] lands around "
        "[-0.08, -0.03, 1.10] (mirrored, and set slightly LOWER in world "
        "z than the other forearm so the two forearms stack -- one resting "
        "just above the other -- rather than colliding at the same "
        "height) with pole_targets[\"l_forearm\"] mirrored too. Keep both "
        "targets close together near the body's centerline and at a "
        "similar (chest/upper-abdomen) height -- a common mistake is "
        "placing them too far out (toward or past the shoulder) or too "
        "high (up near the collarbone), which reads as a raised guard or "
        "flinch instead of a relaxed crossed-arms stance. Do NOT add "
        "bone_rotations for finger bones (e.g. l_index1, r_thumb2) for a "
        "relaxed or neutral pose -- leaving them unset keeps the hand "
        "naturally relaxed/open; only pose fingers into a fist or grip "
        "shape when the description explicitly calls for one. Treat all "
        "of this as illustrative reasoning, not literal numbers to copy "
        "for a different character, rig, or gesture -- always compute "
        "from the actual rest_head_world/rest_tail_world values in the "
        "vocabulary you were given.\n"
        "- pose.look_at_target is optional -- omit it entirely unless the "
        "description specifically calls for a gaze direction. It is a "
        "single [x, y, z] world-space point, not a list of pairs.\n"
        "- expression.weights: a list of {name, value} entries, one per "
        "FACS/morph control you're setting, where value is a weight in "
        "[0.0, 1.0].\n"
        "- Only use bone and control names that appear in the provided "
        "vocabulary, exactly as spelled there. Never invent a name.\n"
        "- Omit any field you have no information for rather than guessing "
        "a value."
    )


def build_user_prompt(
    character: str,
    description: str,
    vocab: AppearanceVocabulary,
    *,
    extra_context: str | None = None,
) -> str:
    prompt = (
        f"Character: {character}\n"
        f"Description: {description}\n\n"
        f"Available pose bones (JSON):\n{json.dumps(vocab.bones)}\n\n"
        f"Available expression controls (JSON):\n{json.dumps(vocab.facs_controls)}"
    )
    if extra_context:
        prompt += f"\n\n{extra_context}"
    return prompt


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
        for bone_name in result.pose.pole_targets:
            if bone_name not in bone_names:
                errors.append(f"pole_targets bone_name {bone_name!r} is not in the vocabulary")
            elif bone_name not in result.pose.ik_targets:
                errors.append(
                    f"pole_targets bone_name {bone_name!r} has no matching ik_targets "
                    "entry -- a pole target is meaningless without an IK target on the same bone"
                )
    for control_name in result.expression.weights:
        if control_name not in facs_control_names:
            errors.append(f"expression control {control_name!r} is not in the vocabulary")
    return errors


_OPEN_MAP_VOCAB_TITLES = {
    "Bone Rotations": "bone_names",
    "Ik Targets": "bone_names",
    "Pole Targets": "bone_names",
    "Weights": "facs_control_names",
}


def _localize_open_maps(schema: dict, vocab: AppearanceVocabulary) -> dict:
    """Rewrite bone_rotations/ik_targets/weights from an open-ended
    ``additionalProperties``-keyed map into a JSON array of {name, value}
    objects, with "name" constrained to an enum of the vocabulary.

    Providers' native structured-output modes (e.g. Anthropic's
    output_config.format) require additionalProperties: false on every
    object schema and have no way to express "any key, this value shape".
    The obvious fix -- emit one named property per vocabulary entry -- was
    tried first, but at this project's real vocabulary size (143 bones,
    446 FACS controls) it produces ~700+ duplicated property schemas and
    Anthropic rejects the compiled result as "too large". A single enum
    listed once, reused by one item schema, stays small regardless of
    vocabulary size. The response is converted back into dict form by
    _pairs_to_dicts before AppearanceResult validation, so PosePayload/
    FACSExpression and every other consumer never see this wire shape.
    """
    field_vocab = {
        title: getattr(vocab, attr) for title, attr in _OPEN_MAP_VOCAB_TITLES.items()
    }

    def walk(node):
        if isinstance(node, dict):
            title = node.get("title")
            if title in field_vocab and isinstance(node.get("additionalProperties"), dict):
                value_schema = walk(node["additionalProperties"])
                item_schema = {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": sorted(field_vocab[title])},
                        "value": value_schema,
                    },
                    "required": ["name", "value"],
                    "additionalProperties": False,
                }
                array_schema = {"type": "array", "items": item_schema}
                if "description" in node:
                    array_schema["description"] = node["description"]
                return array_schema
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(copy.deepcopy(schema))


def _pairs_to_dicts(raw: dict) -> dict:
    """Reverse _localize_open_maps: convert the {name, value}-array wire
    shape the LLM actually returned back into the dict shape
    AppearanceResult/PosePayload/FACSExpression expect."""
    raw = copy.deepcopy(raw)

    def pairs_to_dict(pairs):
        return {pair["name"]: pair["value"] for pair in pairs}

    pose = raw.get("pose")
    if isinstance(pose, dict):
        for field in ("bone_rotations", "ik_targets", "pole_targets"):
            if isinstance(pose.get(field), list):
                pose[field] = pairs_to_dict(pose[field])
    expression = raw.get("expression")
    if isinstance(expression, dict) and isinstance(expression.get("weights"), list):
        expression["weights"] = pairs_to_dict(expression["weights"])
    return raw


_UNSUPPORTED_SCHEMA_KEYWORDS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
}


def _strip_unsupported_keywords(schema: dict) -> dict:
    """Drop JSON Schema keywords that Anthropic's native structured-output
    validator rejects outright rather than merely ignoring: numeric
    "minimum"/"maximum" on a "number" property, and "minItems"/"maxItems"
    values other than 0 or 1 on an array (so the exactly-3-length XYZ
    triples used throughout can't be expressed either). These become
    prompt-only guidance -- see build_system_prompt -- instead of
    schema-enforced constraints."""

    def walk(node):
        if isinstance(node, dict):
            return {
                key: walk(value)
                for key, value in node.items()
                if key not in _UNSUPPORTED_SCHEMA_KEYWORDS
            }
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(copy.deepcopy(schema))


def translate_appearance(
    character: str,
    description: str,
    client: LLMClient,
    *,
    vocab: AppearanceVocabulary | None = None,
    extra_context: str | None = None,
) -> AppearanceResult:
    vocab = vocab or load_default_vocabulary()
    schema = _strip_unsupported_keywords(
        _localize_open_maps(AppearanceResult.model_json_schema(), vocab)
    )
    system_prompt = build_system_prompt()
    base_user_prompt = build_user_prompt(character, description, vocab, extra_context=extra_context)

    errors: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        # LLMClient.complete_json is single-turn (no conversation history), so
        # a retry is folded into one user-turn prompt rather than the design
        # spec's literal system/user/assistant/user multi-turn shape.
        prompt = base_user_prompt if attempt == 0 else base_user_prompt + "\n\n" + build_retry_prompt(errors)
        raw = client.complete_json(system_prompt, prompt, schema)
        try:
            result = AppearanceResult.model_validate(_pairs_to_dicts(raw))
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

    def critique_pose(self, description: str, image_bytes: bytes, media_type: str = "image/png") -> PoseCritique:
        """Two-call implementation: a vision call produces a free-text
        critique of the render against the description, then a text-only
        complete_json call extracts a structured verdict from that critique.
        Kept as two calls rather than combining an image content block with
        output_config.format in one request -- whether Anthropic's
        structured-output mode supports image input in the same call as a
        json_schema constraint is unverified, while both calls made here
        individually use already-proven code paths."""
        import base64

        encoded = base64.standard_b64encode(image_bytes).decode("ascii")
        vision_response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": encoded},
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a rendered image of a character. Does the pose visually "
                            f"match this description: {description!r}? Critique the pose "
                            "specifically -- if a limb or posture doesn't match, say concretely "
                            "what's wrong and where it should be instead."
                        ),
                    },
                ],
            }],
        )
        critique_text = next(
            (block.text for block in vision_response.content if block.type == "text"), ""
        )
        verdict_raw = self.complete_json(
            "You extract a structured verdict from a pose critique. Respond with a single "
            "JSON object matching the given schema.",
            f"Critique:\n{critique_text}\n\n"
            "Does this critique conclude the pose matches the description well enough to "
            "accept, or does it call out a mismatch that should be corrected?",
            PoseCritique.model_json_schema(),
        )
        return PoseCritique.model_validate(verdict_raw)
