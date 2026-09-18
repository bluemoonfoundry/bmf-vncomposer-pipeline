"""Translate a natural-language appearance description into a validated
PosePayload/FACSExpression pair, via a provider-agnostic LLM client.

DAZ Studio is only ever used to source a character once (see
scripts/onboard_character.py); all posing and expression happen in
Blender (blender/worker.py's apply_pose/apply_expression) so they can be
driven by natural language. This module is the translation step
scarecrow_pipeline/scene_tags.py calls "a separate, out-of-scope LLM
step" -- see
docs/superpowers/specs/2026-09-16-nl-appearance-translator-design.md. Note:
that doc's description of the pose contract (raw-float ik_targets/pole_targets
sent to the LLM) is superseded by docs/gemini_ikplan_a.md.

Arm IK goals are LLM-facing as a discrete anchor + small offset
(LimbGoal/PoseIntent), not raw world-space floats -- see
docs/gemini_ikplan_a.md and scarecrow_pipeline/anchors.py.
resolve_pose_intent() converts that into the concrete PosePayload
(ik_targets/pole_targets as world-space points) blender/worker.py already
knows how to apply; that contract is unchanged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from scarecrow_pipeline import anchors
from scarecrow_pipeline.schemas import FACSExpression, PosePayload, Vector3

MAX_ATTEMPTS = 2

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSABLE_BONES_PATH = REPO_ROOT / "docs" / "posable_bones.json"
DEFAULT_FACS_CONTROLS_PATH = REPO_ROOT / "docs" / "facs_controls.json"


class AppearanceTranslationError(RuntimeError):
    """Raised when the LLM's output cannot be validated after MAX_ATTEMPTS tries."""


class LLMClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> dict:
        """Return a dict parsed from the model's structured-output response,
        constrained to json_schema. Raises on a provider/network failure;
        does not itself validate the dict against the schema."""
        ...


class LimbGoal(BaseModel):
    """One arm's IK goal, expressed as a semantic anchor + a small local
    nudge -- never a raw world-space guess. See docs/gemini_ikplan_a.md
    Phase 1/2, adapted to this project's real rig in
    scarecrow_pipeline/anchors.py."""

    model_config = ConfigDict(extra="forbid")
    target_anchor: str = Field(..., description="One of anchors.ANCHOR_NAMES.")
    character_local_offset: Vector3 = Field(
        default=[0.0, 0.0, 0.0],
        description=(
            "[left(+)/right(-), front(+)/back(-), up(+)/down(-)] meters from "
            "target_anchor. Keep each component within [-0.15, 0.15]."
        ),
    )
    layer_depth: Literal["inner", "outer", "neutral"] = Field(
        default="neutral",
        description=(
            "For crossed arms: exactly one arm must be 'outer' (pushed toward "
            "the front to clear the other) and the other 'inner' (tucked "
            "toward the back). 'neutral' for a single-arm gesture."
        ),
    )
    elbow_strategy: Literal["DOWN_FORWARD", "OUTWARD", "DOWN_PINNED", "UP_FLUID"] = "DOWN_FORWARD"

    @field_validator("target_anchor")
    @classmethod
    def _known_anchor(cls, value: str) -> str:
        if value not in anchors.ANCHOR_NAMES:
            raise ValueError(f"target_anchor {value!r} is not in anchors.ANCHOR_NAMES")
        return value

    @field_validator("character_local_offset")
    @classmethod
    def _clamp_offset(cls, value: Vector3) -> Vector3:
        # Match the [-0.15, 0.15] range documented above and in
        # build_system_prompt -- anchors.clamp_offset's own 0.20 default is
        # for the wider critique-delta path (LimbDelta/apply_critique_delta),
        # not the LLM's initial placement.
        return anchors.clamp_offset(list(value), limit=0.15)


class PoseIntent(BaseModel):
    """LLM-facing pose wire shape -- resolved into a concrete PosePayload
    by resolve_pose_intent() before anything is rendered."""

    model_config = ConfigDict(extra="forbid")
    bone_rotations: dict[str, Vector3] = Field(default_factory=dict)
    root_location: Vector3 | None = None
    weight_stance: Literal["neutral", "shift_left", "shift_right"] = "neutral"
    left_arm: LimbGoal | None = None
    right_arm: LimbGoal | None = None
    look_at_target: Vector3 | None = None


class AppearanceIntent(BaseModel):
    """LLM-facing wire shape for one translate_pose_intent() call."""

    model_config = ConfigDict(extra="forbid")
    pose: PoseIntent | None = None
    expression: FACSExpression = Field(default_factory=FACSExpression)


class AppearanceResult(BaseModel):
    """One description's RESOLVED appearance -- assign both fields straight
    onto a CharacterPlacement/RenderRequest (scarecrow_pipeline/schemas.py),
    which carry pose and expression as the same two side-by-side fields.
    Unchanged by the anchor-based rewrite: blender/worker.py, sprite_set.py,
    and RenderRequest/CharacterPlacement all still consume PosePayload
    directly."""

    model_config = ConfigDict(extra="forbid")
    pose: PosePayload | None = None
    expression: FACSExpression = Field(default_factory=FACSExpression)


class LimbDelta(BaseModel):
    """One numeric correction to a previous attempt's LimbGoal -- see
    docs/gemini_ikplan_a.md Phase 4."""

    model_config = ConfigDict(extra="forbid")
    limb: Literal["left_arm", "right_arm"]
    delta_meters: Vector3 = Field(
        description=(
            "[left(+)/right(-), front(+)/back(-), up(+)/down(-)] meters to "
            "ADD to that limb's current character_local_offset."
        )
    )
    change_anchor: str | None = Field(
        default=None,
        description="Optional replacement target_anchor if the current one is fundamentally wrong.",
    )
    notes: str = ""

    @field_validator("change_anchor")
    @classmethod
    def _known_anchor(cls, value: str | None) -> str | None:
        if value is not None and value not in anchors.ANCHOR_NAMES:
            raise ValueError(f"change_anchor {value!r} is not in anchors.ANCHOR_NAMES")
        return value


class CritiqueDeltaFeedback(BaseModel):
    """Structured verdict from a vision-model critique of a rendered pose,
    expressed as numeric per-limb corrections rather than prose -- see
    docs/gemini_ikplan_a.md Phase 4."""

    model_config = ConfigDict(extra="forbid")
    pose_is_satisfactory: bool
    critique_summary: str
    adjustments: list[LimbDelta] = Field(default_factory=list)


def apply_critique_delta(intent: AppearanceIntent, feedback: CritiqueDeltaFeedback) -> AppearanceIntent:
    """Apply corrective vector deltas to a previous PoseIntent, in place of
    re-invoking the pose-generation LLM -- see docs/gemini_ikplan_a.md
    Phase 4's apply_critique_delta, adapted to this project's PoseIntent/
    LimbGoal shape."""
    new_intent = intent.model_copy(deep=True)
    if new_intent.pose is None:
        return new_intent
    for adjustment in feedback.adjustments:
        goal = getattr(new_intent.pose, adjustment.limb, None)
        if goal is None:
            continue
        if adjustment.change_anchor is not None:
            goal.target_anchor = adjustment.change_anchor
        offset = [goal.character_local_offset[i] + adjustment.delta_meters[i] for i in range(3)]
        goal.character_local_offset = anchors.clamp_offset(offset)
    return new_intent


class VisionCritiqueClient(Protocol):
    def critique_pose(
        self, description: str, image_bytes: bytes, current_intent: PoseIntent | None
    ) -> CritiqueDeltaFeedback:
        """Return a structured critique of a rendered PNG against its
        natural-language description and the pose actually attempted
        (current_intent, so the critique can reference specific limbs/
        anchors). Raises on a provider/network failure."""
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

    @property
    def bone_landmarks(self) -> dict[str, dict]:
        return {bone["name"]: bone for bone in self.bones}


def load_default_vocabulary(
    posable_bones_path: Path = DEFAULT_POSABLE_BONES_PATH,
    facs_controls_path: Path = DEFAULT_FACS_CONTROLS_PATH,
) -> AppearanceVocabulary:
    bones = json.loads(Path(posable_bones_path).read_text(encoding="utf-8"))["bones"]
    facs_controls = json.loads(Path(facs_controls_path).read_text(encoding="utf-8"))["controls"]
    return AppearanceVocabulary(bones=bones, facs_controls=facs_controls)


def _available_anchor_names(vocab: AppearanceVocabulary) -> list[str]:
    """anchors.ANCHOR_NAMES filtered down to anchors whose landmark bone is
    actually present in the given vocab -- so the LLM (via the system
    prompt's anchor list and the schema's target_anchor enum) is never
    offered an anchor that resolve_pose_intent can't resolve for this
    vocabulary. Mirrors build_user_prompt's existing anchor_positions
    filter."""
    bone_landmarks = vocab.bone_landmarks
    return [name for name in anchors.ANCHOR_NAMES if anchors.ANCHOR_LANDMARKS[name][0] in bone_landmarks]


def build_system_prompt(vocab: AppearanceVocabulary) -> str:
    available_anchors = _available_anchor_names(vocab)
    anchor_lines = "\n".join(
        f"    - {name}: {anchors.ANCHOR_DESCRIPTIONS[name]}" for name in available_anchors
    )
    return (
        "You translate a natural-language character appearance description "
        "into precise pose and facial-expression control values for a "
        "Diffeomorphic-imported DAZ character rig in Blender.\n\n"
        "Respond with a single JSON object matching the given schema:\n"
        "- pose.bone_rotations: a list of {name, value} entries, one per "
        "bone you're setting (spine/neck/torso/twist bones, or any bone "
        "with no clear spatial target), where name is the bone name and "
        "value is an [x, y, z] Euler rotation IN RADIANS, applied directly "
        "as that bone's rotation_euler.x/.y/.z -- value[0] is always the "
        "X-axis angle, value[1] always Y, value[2] always Z, regardless of "
        "that bone's own rotation_mode (e.g. \"YZX\") in the provided "
        "vocabulary -- rotation_mode only tells you the ORDER Blender "
        "composes those three axis rotations, not which array index holds "
        "which axis. NEVER set bone_rotations for l_shoulder, r_shoulder, "
        "l_upperarm, r_upperarm, l_forearm, or r_forearm -- use left_arm/"
        "right_arm below for any arm gesture with a spatial goal instead; "
        "the two mechanisms fight each other if both touch the same arm.\n"
        "- pose.left_arm / pose.right_arm: set one or both when the "
        "description calls for a specific arm gesture (crossed arms, hand "
        "on hip/chest, reaching). Each is a LimbGoal:\n"
        "  - target_anchor: pick the semantic landmark closest to where "
        "that hand/forearm should end up, from this fixed list (each is "
        "resolved to a world-space point in the anchor_positions JSON "
        "below):\n"
        f"{anchor_lines}\n"
        "  - character_local_offset: a SMALL [left(+)/right(-), "
        "front(+)/back(-), up(+)/down(-)] nudge in meters away from "
        "target_anchor, each component within [-0.15, 0.15]. Do not use "
        "this to reach a distant landmark -- pick the closer anchor "
        "instead.\n"
        "  - layer_depth: for crossed arms, exactly one of left_arm/"
        "right_arm must be \"outer\" (the arm resting on top, pushed toward "
        "the front to clear the other) and the other \"inner\" (tucked "
        "toward the back, resting against the chest); use \"neutral\" for "
        "any single-arm gesture that doesn't involve the other arm.\n"
        "  - elbow_strategy: \"DOWN_FORWARD\" for a relaxed arm bending "
        "across the body (crossed arms, resting on stomach) -- almost "
        "always correct; \"OUTWARD\" for a T-pose/fighting stance; "
        "\"DOWN_PINNED\" for a hand resting straight down at the side; "
        "\"UP_FLUID\" for touching the head or leaning on something.\n"
        "- pose.weight_stance: prefer \"shift_left\" or \"shift_right\" for "
        "a natural asymmetric standing pose over \"neutral\" -- bodies "
        "rarely balance perfectly on both legs. This automatically shifts "
        "the pelvis and counter-tilts the lower spine, so do not also add "
        "a bone_rotations or root_location entry to do the same thing.\n"
        "- pose.look_at_target is optional -- omit it entirely unless the "
        "description specifically calls for a gaze direction. It is a "
        "single [x, y, z] world-space point, not a list of pairs.\n"
        "- expression.weights: a list of {name, value} entries, one per "
        "FACS/morph control you're setting, where value is a weight in "
        "[0.0, 1.0].\n"
        "- Only use bone names from the provided vocabulary and anchor "
        "names from the fixed list above. Never invent a name.\n"
        "- Omit any field you have no information for rather than guessing "
        "a value. Do NOT add bone_rotations for finger bones for a relaxed "
        "or neutral pose -- leaving them unset keeps the hand naturally "
        "relaxed/open; only pose fingers when the description explicitly "
        "calls for a fist or grip shape."
    )


def build_user_prompt(
    character: str,
    description: str,
    vocab: AppearanceVocabulary,
    *,
    extra_context: str | None = None,
) -> str:
    bone_landmarks = vocab.bone_landmarks
    anchor_positions = {
        name: anchors.resolve_anchor_position(name, bone_landmarks)
        for name in anchors.ANCHOR_NAMES
        if anchors.ANCHOR_LANDMARKS[name][0] in bone_landmarks
    }
    prompt = (
        f"Character: {character}\n"
        f"Description: {description}\n\n"
        f"Available pose bones (JSON):\n{json.dumps(vocab.bones)}\n\n"
        f"Available expression controls (JSON):\n{json.dumps(vocab.facs_controls)}\n\n"
        f"Anchor world-space positions for this character (JSON):\n{json.dumps(anchor_positions)}"
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


def _vocabulary_errors(intent: AppearanceIntent, vocab: AppearanceVocabulary) -> list[str]:
    errors = []
    bone_names = vocab.bone_names
    facs_control_names = vocab.facs_control_names
    if intent.pose is not None:
        for bone_name in intent.pose.bone_rotations:
            if bone_name not in bone_names:
                errors.append(f"bone_name {bone_name!r} is not in the vocabulary")
    for control_name in intent.expression.weights:
        if control_name not in facs_control_names:
            errors.append(f"expression control {control_name!r} is not in the vocabulary")
    return errors


_OPEN_MAP_VOCAB_TITLES = {
    "Bone Rotations": "bone_names",
    "Weights": "facs_control_names",
}


def _localize_open_maps(schema: dict, vocab: AppearanceVocabulary) -> dict:
    """Rewrite bone_rotations/weights from an open-ended
    ``additionalProperties``-keyed map into a JSON array of {name, value}
    objects, with "name" constrained to an enum of the vocabulary, AND
    constrain every LimbGoal.target_anchor property to the anchors whose
    landmark bone is present in vocab (see _available_anchor_names) -- an
    anchor resolve_pose_intent can't resolve for this vocab is never
    offered to the LLM in the first place.

    Providers' native structured-output modes (e.g. Anthropic's
    output_config.format) require additionalProperties: false on every
    object schema and have no way to express "any key, this value shape".
    The obvious fix -- emit one named property per vocabulary entry -- was
    tried first, but at this project's real vocabulary size (143 bones,
    446 FACS controls) it produces ~700+ duplicated property schemas and
    Anthropic rejects the compiled result as "too large". A single enum
    listed once, reused by one item schema, stays small regardless of
    vocabulary size. The response is converted back into dict form by
    _pairs_to_dicts before AppearanceIntent validation.
    """
    field_vocab = {
        title: getattr(vocab, attr) for title, attr in _OPEN_MAP_VOCAB_TITLES.items()
    }
    available_anchors = _available_anchor_names(vocab)

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
            node = {key: walk(value) for key, value in node.items()}
            properties = node.get("properties")
            if isinstance(properties, dict) and "target_anchor" in properties:
                properties["target_anchor"] = {
                    **properties["target_anchor"],
                    "enum": available_anchors,
                }
            return node
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(copy.deepcopy(schema))


def _pairs_to_dicts(raw: dict) -> dict:
    """Reverse _localize_open_maps's array-of-pairs transform for
    bone_rotations/weights, converting the LLM's actual response back into
    the dict shape PoseIntent/FACSExpression expect."""
    raw = copy.deepcopy(raw)

    def pairs_to_dict(pairs):
        return {pair["name"]: pair["value"] for pair in pairs}

    pose = raw.get("pose")
    if isinstance(pose, dict) and isinstance(pose.get("bone_rotations"), list):
        pose["bone_rotations"] = pairs_to_dict(pose["bone_rotations"])
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


def resolve_pose_intent(intent: PoseIntent | None, vocab: AppearanceVocabulary) -> PosePayload | None:
    """Convert an LLM-facing PoseIntent into the concrete PosePayload
    blender/worker.py already knows how to apply -- see
    docs/gemini_ikplan_a.md Phase 1/3, adapted: clavicle protraction and
    weight-stance resolve to plain bone_rotations/root_location numbers
    (no bpy needed); only the arm IK targets need the anchor table."""
    if intent is None:
        return None

    bone_landmarks = vocab.bone_landmarks
    bone_rotations: dict[str, list[float]] = {}
    root_location = intent.root_location

    if intent.weight_stance != "neutral":
        shift_root, shift_rotations = anchors.resolve_weight_stance(intent.weight_stance)
        bone_rotations.update(shift_rotations)
        if root_location is None:
            root_location = shift_root

    both_arms_layered = (
        intent.left_arm is not None
        and intent.right_arm is not None
        and intent.left_arm.layer_depth in ("inner", "outer")
        and intent.right_arm.layer_depth in ("inner", "outer")
    )
    if both_arms_layered:
        bone_rotations.update(anchors.resolve_clavicle_protraction())

    bone_rotations.update(intent.bone_rotations)  # explicit LLM values win

    ik_targets: dict[str, list[float]] = {}
    pole_targets: dict[str, list[float]] = {}
    for side, goal in (("L", intent.left_arm), ("R", intent.right_arm)):
        if goal is None:
            continue
        limb_bones = anchors.LIMB_BONES[side]
        ik_target_bone = limb_bones["ik_target_bone"]
        shoulder_bone = limb_bones["shoulder_bone"]
        ik_position = anchors.resolve_limb_target(
            goal.target_anchor, goal.character_local_offset, goal.layer_depth, bone_landmarks
        )
        shoulder_position = bone_landmarks[shoulder_bone]["rest_head_world"]
        pole_position = anchors.resolve_pole_target(goal.elbow_strategy, side, ik_position, shoulder_position)
        ik_targets[ik_target_bone] = ik_position
        pole_targets[ik_target_bone] = pole_position

    return PosePayload(
        bone_rotations=bone_rotations,
        root_location=root_location,
        ik_targets=ik_targets,
        pole_targets=pole_targets,
        look_at_target=intent.look_at_target,
    )


def translate_pose_intent(
    character: str,
    description: str,
    client: LLMClient,
    *,
    vocab: AppearanceVocabulary | None = None,
    extra_context: str | None = None,
) -> AppearanceIntent:
    vocab = vocab or load_default_vocabulary()
    schema = _strip_unsupported_keywords(_localize_open_maps(AppearanceIntent.model_json_schema(), vocab))
    system_prompt = build_system_prompt(vocab)
    base_user_prompt = build_user_prompt(character, description, vocab, extra_context=extra_context)

    errors: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        # LLMClient.complete_json is single-turn (no conversation history), so
        # a retry is folded into one user-turn prompt.
        prompt = base_user_prompt if attempt == 0 else base_user_prompt + "\n\n" + build_retry_prompt(errors)
        raw = client.complete_json(system_prompt, prompt, schema)
        try:
            intent = AppearanceIntent.model_validate(_pairs_to_dicts(raw))
        except ValidationError as exc:
            errors = [str(exc)]
            continue
        errors = _vocabulary_errors(intent, vocab)
        if not errors:
            return intent

    raise AppearanceTranslationError(
        f"Could not produce a valid appearance for {character!r} after "
        f"{MAX_ATTEMPTS} attempt(s): " + "; ".join(errors)
    )


def translate_appearance(
    character: str,
    description: str,
    client: LLMClient,
    *,
    vocab: AppearanceVocabulary | None = None,
    extra_context: str | None = None,
) -> AppearanceResult:
    vocab = vocab or load_default_vocabulary()
    intent = translate_pose_intent(character, description, client, vocab=vocab, extra_context=extra_context)
    return AppearanceResult(pose=resolve_pose_intent(intent.pose, vocab), expression=intent.expression)


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

    def critique_pose(
        self, description: str, image_bytes: bytes, current_intent: PoseIntent | None, media_type: str = "image/png"
    ) -> CritiqueDeltaFeedback:
        """Two-call implementation: a vision call produces a free-text
        critique of the render against the description and the pose
        actually attempted, then a text-only complete_json call extracts
        structured per-limb deltas from that critique -- see
        docs/gemini_ikplan_a.md Phase 4."""
        import base64

        current_state = (
            f"left_arm: target_anchor={current_intent.left_arm.target_anchor!r}, "
            f"character_local_offset={current_intent.left_arm.character_local_offset}\n"
            if current_intent and current_intent.left_arm
            else "left_arm: not set\n"
        ) + (
            f"right_arm: target_anchor={current_intent.right_arm.target_anchor!r}, "
            f"character_local_offset={current_intent.right_arm.character_local_offset}"
            if current_intent and current_intent.right_arm
            else "right_arm: not set"
        )

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
                            f"match this description: {description!r}? The current arm state "
                            f"is:\n{current_state}\n\n"
                            "If a limb is wrong, say concretely which limb (left_arm/right_arm), "
                            "which direction it should move (left/right, front/back, up/down), "
                            "and roughly how far in centimeters."
                        ),
                    },
                ],
            }],
        )
        critique_text = next(
            (block.text for block in vision_response.content if block.type == "text"), ""
        )
        verdict_raw = self.complete_json(
            "You extract structured numeric pose corrections from a critique. Respond with a "
            "single JSON object matching the given schema. delta_meters is "
            "[left(+)/right(-), front(+)/back(-), up(+)/down(-)] meters to ADD to that limb's "
            "current character_local_offset -- a small nudge (typically 0.01-0.06), not a "
            "full re-placement, unless the critique says the anchor itself is wrong (use "
            "change_anchor for that instead).",
            f"Critique:\n{critique_text}\n\n"
            f"Current arm state:\n{current_state}\n\n"
            "Does this critique conclude the pose matches the description well enough to "
            "accept, or does it call out a mismatch that should be corrected?",
            _strip_unsupported_keywords(CritiqueDeltaFeedback.model_json_schema()),
        )
        return CritiqueDeltaFeedback.model_validate(verdict_raw)
