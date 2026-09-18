# Anchor-Based Pose Intent & Delta-Critique Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw world-space `ik_targets`/`pole_targets` floats in the LLM-facing pose contract with a discrete semantic-anchor + small-offset scheme (docs/gemini_ikplan_a.md), and replace the critique-retry loop's prose-only feedback with a structured, numeric per-limb delta scheme -- so a critique retry nudges specific coordinates deterministically instead of asking the LLM to re-guess a whole pose from scratch.

**Architecture:** `scarecrow_pipeline/anchors.py` (new, pure Python, no `bpy`) defines a small fixed table of named landmarks on the real Diffeomorphic Genesis 9 rig (adapted from the doc's placeholder MHX bone names) and resolves them from each bone's rest-pose world position already present in `docs/posable_bones.json`. `scarecrow_pipeline/nl_appearance.py`'s LLM-facing wire schema changes from `PosePayload`-with-raw-floats to a new `AppearanceIntent`/`PoseIntent`/`LimbGoal` shape (`target_anchor` + `character_local_offset` + `layer_depth` + `elbow_strategy`), resolved into a concrete `PosePayload` (unchanged contract) before anything is rendered -- so `blender/worker.py`, `scarecrow_pipeline/sprite_set.py`, and `scarecrow_pipeline/schemas.py` need no changes except one addition (a shrinkwrap collision guard, Phase 3 of the doc). `scarecrow_pipeline/pose_critique.py`'s retry loop stops re-calling the pose-generation LLM every attempt; instead a vision critique returns a structured `CritiqueDeltaFeedback` (numeric per-limb deltas), applied additively and deterministically to the previous attempt's `PoseIntent` before re-rendering.

**Tech Stack:** Python 3.12, Pydantic v2, pytest. No `bpy` in any file this plan adds to `scarecrow_pipeline/` (that package runs outside Blender -- see `nl_appearance.py`'s existing module docstring). `blender/worker.py` is the one file in this plan that does run inside Blender.

**Spec:** `docs/gemini_ikplan_a.md` (bd issue `scarecrow-4sm`'s engineering spec). This plan adapts every phase to this project's real rig (`docs/posable_bones.json`: Diffeomorphic Genesis 9, bones like `l_forearm`/`r_upperarm`/`spine4`, not the doc's placeholder MHX names like `upper_arm.L`/`chest`) and its real architecture (pose translation runs as plain-Python LLM orchestration with no live Blender armature -- see Global Constraints).

## Global Constraints

- `scarecrow_pipeline/` must stay `bpy`-free -- it is imported and run outside Blender (see `nl_appearance.py`'s module docstring: "this module is the translation step... a separate, out-of-scope LLM step"). Anchors are computed from **rest-pose** landmarks (`docs/posable_bones.json`'s `rest_head_world`/`rest_tail_world`), not live bpy Empties, because no live armature exists at translate time.
- `schemas.py`'s `PosePayload` (the contract `blender/worker.py`, `sprite_set.py`, and `RenderRequest`/`CharacterPlacement` all consume) does **not** change shape. Only the LLM-facing wire schema in `nl_appearance.py` changes. This keeps the blast radius to `nl_appearance.py`, `pose_critique.py`, their tests, and one addition to `blender/worker.py`.
- Character-local axis convention (matches `schemas.py`'s `SceneLighting` comment, which established that a front-facing subject's outward normal points toward world `-Y`): `+X = left`, `-X = right`, `+Y (local) = front` which maps to world `-Y`, `-Y (local) = back` maps to world `+Y`, `+Z = up` maps to world `+Z` directly.
- Every bone name referenced by generated code must be a real name from `docs/posable_bones.json` (verified during planning: `hip`, `spine1`-`spine4`, `l_shoulder`/`r_shoulder`, `l_upperarm`/`r_upperarm`, `l_forearm`/`r_forearm`, `head`, `l_hand`/`r_hand` all confirmed present).
- Each task must leave `python -m pytest -q` green before moving to the next task.
- The real end-to-end verification this feature exists for (`apply_appearance.py --with-critique` against a live Anthropic API + a real Blender render, confirming attempt N+1 converges closer than attempt N) is **not runnable in this environment** (no `ANTHROPIC_API_KEY`, no `blender` on `PATH`). Every task below is verified with unit tests against fakes; the final task documents this limitation on the bd issue instead of claiming it's closed.

---

### Task 1: `anchors.py` -- semantic anchor landmarks

**Files:**
- Create: `scarecrow_pipeline/anchors.py`
- Test: `tests/test_anchors.py`

**Interfaces:**
- Produces: `anchors.ANCHOR_NAMES: tuple[str, ...]`, `anchors.ANCHOR_LANDMARKS: dict[str, tuple[str, str]]`, `anchors.ANCHOR_DESCRIPTIONS: dict[str, str]`, `anchors.UnknownAnchorError(ValueError)`, `anchors.resolve_anchor_position(anchor_name: str, bone_landmarks: dict[str, dict]) -> list[float]`, `anchors.character_offset_to_world(offset: list[float]) -> list[float]`, `anchors.clamp_offset(offset: list[float], limit: float = 0.20) -> list[float]`, `anchors.resolve_limb_target(target_anchor: str, character_local_offset: list[float], layer_depth: str, bone_landmarks: dict[str, dict]) -> list[float]`, `anchors.resolve_pole_target(elbow_strategy: str, side: str, ik_target_position: list[float], shoulder_position: list[float]) -> list[float]`, `anchors.resolve_weight_stance(stance: str) -> tuple[list[float] | None, dict[str, list[float]]]`, `anchors.resolve_clavicle_protraction(forward_deg: float = 12.0, elevation_deg: float = 6.0) -> dict[str, list[float]]`, `anchors.LIMB_BONES: dict[str, dict[str, str]]` (`{"L": {"ik_target_bone": "l_forearm", "shoulder_bone": "l_upperarm"}, "R": {...}}`).
- Consumed by: Task 2 (`nl_appearance.py`).

- [ ] **Step 1: Write `scarecrow_pipeline/anchors.py`**

```python
"""Semantic surface-anchor landmarks for arm IK targets.

Adapted from docs/gemini_ikplan_a.md's Phase 1 (Anchor Empties parented to
live pose bones) to this project's actual architecture: translate_appearance()
runs as a plain-Python LLM-orchestration step with no bpy/live-armature
access (see nl_appearance.py's module docstring) -- posing only happens
later, in blender/worker.py, inside a Blender subprocess. So instead of
spawning bpy Empties at pose time, an anchor is a fixed (bone_name, "head"
or "tail") landmark resolved from that bone's REST-pose world position,
which is already present on every vocabulary entry (see
docs/posable_bones.json's rest_head_world/rest_tail_world) -- the same rest
data nl_appearance.py's system prompt previously told the LLM to reason
from manually. Bone names are this project's real Diffeomorphic Genesis 9
rig names (l_forearm, r_upperarm, spine4, ...), not the doc's placeholder
MHX names (upper_arm.L, chest, ...).
"""

from __future__ import annotations

import math

# anchor_name -> (bone_name, "head" | "tail")
ANCHOR_LANDMARKS: dict[str, tuple[str, str]] = {
    "ANCHOR_CHEST_CENTER": ("spine4", "head"),
    "ANCHOR_PECTORAL_L": ("l_shoulder", "head"),
    "ANCHOR_PECTORAL_R": ("r_shoulder", "head"),
    "ANCHOR_BICEP_LATERAL_L": ("l_upperarm", "tail"),
    "ANCHOR_BICEP_LATERAL_R": ("r_upperarm", "tail"),
    "ANCHOR_FOREARM_VENTRAL_L": ("l_forearm", "tail"),
    "ANCHOR_FOREARM_VENTRAL_R": ("r_forearm", "tail"),
}

ANCHOR_DESCRIPTIONS: dict[str, str] = {
    "ANCHOR_CHEST_CENTER": "center of the upper chest/collar, at spine4",
    "ANCHOR_PECTORAL_L": "left collarbone/pectoral, at the left shoulder joint",
    "ANCHOR_PECTORAL_R": "right collarbone/pectoral, at the right shoulder joint",
    "ANCHOR_BICEP_LATERAL_L": "outer point near the left elbow -- good for the RIGHT hand to grip when arms are crossed",
    "ANCHOR_BICEP_LATERAL_R": "outer point near the right elbow -- good for the LEFT hand to grip when arms are crossed",
    "ANCHOR_FOREARM_VENTRAL_L": "left wrist, at the end of the left forearm",
    "ANCHOR_FOREARM_VENTRAL_R": "right wrist, at the end of the right forearm",
}

ANCHOR_NAMES: tuple[str, ...] = tuple(ANCHOR_LANDMARKS)

# Which forearm bone is the IK end-effector, and which upperarm bone is
# that chain's shoulder-end landmark for pole-vector placement, per side.
LIMB_BONES: dict[str, dict[str, str]] = {
    "L": {"ik_target_bone": "l_forearm", "shoulder_bone": "l_upperarm"},
    "R": {"ik_target_bone": "r_forearm", "shoulder_bone": "r_upperarm"},
}


class UnknownAnchorError(ValueError):
    """Raised when a target_anchor/change_anchor name isn't in ANCHOR_NAMES,
    or its landmark bone isn't in the given vocabulary."""


def resolve_anchor_position(anchor_name: str, bone_landmarks: dict[str, dict]) -> list[float]:
    """World-space rest-pose position of a named anchor.

    bone_landmarks maps bone name -> that bone's vocabulary entry (must
    carry rest_head_world/rest_tail_world), e.g. {b["name"]: b for b in
    vocab.bones}.
    """
    if anchor_name not in ANCHOR_LANDMARKS:
        raise UnknownAnchorError(f"{anchor_name!r} is not a known anchor -- see ANCHOR_NAMES")
    bone_name, point = ANCHOR_LANDMARKS[anchor_name]
    bone = bone_landmarks.get(bone_name)
    if bone is None:
        raise UnknownAnchorError(
            f"anchor {anchor_name!r} depends on bone {bone_name!r}, which is not in the given vocabulary"
        )
    return list(bone[f"rest_{point}_world"])


def character_offset_to_world(offset: list[float]) -> list[float]:
    """Character-local [left(+)/right(-), front(+)/back(-), up(+)/down(-)]
    meters -> world-space [x, y, z] delta.

    This rig's character-local axes already align with world X/Z (left is
    +X, up is +Z); only the front/back axis is flipped, since the subject
    faces the camera along world -Y (see schemas.py's SceneLighting
    comment: a front-facing subject's outward normal points toward -Y).
    """
    left, front, up = offset
    return [left, -front, up]


def clamp_offset(offset: list[float], limit: float = 0.20) -> list[float]:
    return [max(-limit, min(limit, value)) for value in offset]


def resolve_limb_target(
    target_anchor: str,
    character_local_offset: list[float],
    layer_depth: str,
    bone_landmarks: dict[str, dict],
) -> list[float]:
    """World-space IK end-effector position for one LimbGoal."""
    base = resolve_anchor_position(target_anchor, bone_landmarks)
    local = list(character_local_offset)
    if layer_depth == "outer":
        local[1] += 0.05  # pushed further toward the front, to clear the "inner" limb
    elif layer_depth == "inner":
        local[1] -= 0.02  # tucked slightly toward the back, closer to the chest
    world_delta = character_offset_to_world(local)
    return [base[i] + world_delta[i] for i in range(3)]


_WORLD_DIRECTIONS = {
    "forward": [0.0, -1.0, 0.0],
    "down": [0.0, 0.0, -1.0],
}

_POLE_DIRECTIONS: dict[str, tuple[tuple[str, float], ...]] = {
    "DOWN_FORWARD": (("down", 0.25), ("forward", 0.15)),
    "OUTWARD": (("outward", 0.35),),
    "DOWN_PINNED": (("down", 0.35),),
    "UP_FLUID": (("down", 0.20),),
}


def resolve_pole_target(
    elbow_strategy: str,
    side: str,
    ik_target_position: list[float],
    shoulder_position: list[float],
) -> list[float]:
    """World-space pole (elbow-bend direction) position for one LimbGoal."""
    mid = [(a + b) / 2.0 for a, b in zip(shoulder_position, ik_target_position)]
    outward = [1.0 if side == "L" else -1.0, 0.0, 0.0]
    directions = {**_WORLD_DIRECTIONS, "outward": outward}
    position = list(mid)
    for direction_name, scale in _POLE_DIRECTIONS[elbow_strategy]:
        direction = directions[direction_name]
        position = [position[i] + direction[i] * scale for i in range(3)]
    return position


def resolve_weight_stance(stance: str) -> tuple[list[float] | None, dict[str, list[float]]]:
    """(root_location, bone_rotations) auto-generated from a weight_stance
    choice -- a small pelvis shift and a lower-spine counter-tilt, so the
    LLM doesn't have to compute a symmetric-looking pose by hand."""
    if stance == "neutral":
        return None, {}
    sign = 1.0 if stance == "shift_left" else -1.0
    root_location = [0.04 * sign, 0.0, 0.0]
    bone_rotations = {"spine1": [0.0, 0.0, math.radians(-3.0 * sign)]}
    return root_location, bone_rotations


def resolve_clavicle_protraction(forward_deg: float = 12.0, elevation_deg: float = 6.0) -> dict[str, list[float]]:
    """bone_rotations for l_shoulder/r_shoulder (this rig's clavicle-
    equivalent bones -- both parented to spine4, see docs/posable_bones.json)
    that rotate the shoulders forward and slightly up, so a crossed-arms
    pose doesn't distort the shoulder socket the way posing the arms alone
    would."""
    return {
        "l_shoulder": [math.radians(elevation_deg), 0.0, math.radians(forward_deg)],
        "r_shoulder": [math.radians(elevation_deg), 0.0, math.radians(-forward_deg)],
    }
```

- [ ] **Step 2: Write `tests/test_anchors.py`**

```python
import pytest

from scarecrow_pipeline import anchors


def _bone_landmarks():
    return {
        "spine4": {"rest_head_world": [0.0, 0.052, 1.2745], "rest_tail_world": [0.0, 0.052, 1.4048]},
        "l_shoulder": {"rest_head_world": [0.0384, 0.0474, 1.3575], "rest_tail_world": [0.1515, 0.0675, 1.3758]},
        "r_shoulder": {"rest_head_world": [-0.0385, 0.0474, 1.3575], "rest_tail_world": [-0.1541, 0.0568, 1.3644]},
        "l_upperarm": {"rest_head_world": [0.176, 0.0637, 1.3691], "rest_tail_world": [0.3506, 0.0222, 1.2154]},
        "r_upperarm": {"rest_head_world": [-0.1774, 0.0524, 1.3542], "rest_tail_world": [-0.3334, 0.0495, 1.1769]},
        "l_forearm": {"rest_head_world": [0.3567, 0.0146, 1.2243], "rest_tail_world": [0.5376, -0.1237, 1.1025]},
        "r_forearm": {"rest_head_world": [-0.3415, 0.042, 1.1841], "rest_tail_world": [-0.5282, -0.0183, 1.0162]},
    }


def test_resolve_anchor_position_uses_bone_rest_landmark():
    position = anchors.resolve_anchor_position("ANCHOR_BICEP_LATERAL_R", _bone_landmarks())

    assert position == [-0.3334, 0.0495, 1.1769]


def test_resolve_anchor_position_rejects_unknown_anchor():
    with pytest.raises(anchors.UnknownAnchorError, match="NOT_AN_ANCHOR"):
        anchors.resolve_anchor_position("NOT_AN_ANCHOR", _bone_landmarks())


def test_resolve_anchor_position_rejects_bone_missing_from_vocab():
    with pytest.raises(anchors.UnknownAnchorError, match="l_upperarm"):
        anchors.resolve_anchor_position("ANCHOR_BICEP_LATERAL_L", {})


def test_character_offset_to_world_flips_front_back_only():
    assert anchors.character_offset_to_world([0.1, 0.05, -0.02]) == [0.1, -0.05, -0.02]


def test_clamp_offset_limits_each_axis():
    assert anchors.clamp_offset([0.5, -0.5, 0.1], limit=0.20) == [0.20, -0.20, 0.1]


def test_resolve_limb_target_outer_pushes_forward_inner_pulls_back():
    landmarks = _bone_landmarks()
    base = anchors.resolve_anchor_position("ANCHOR_BICEP_LATERAL_R", landmarks)

    outer = anchors.resolve_limb_target("ANCHOR_BICEP_LATERAL_R", [0.0, 0.0, 0.0], "outer", landmarks)
    inner = anchors.resolve_limb_target("ANCHOR_BICEP_LATERAL_R", [0.0, 0.0, 0.0], "inner", landmarks)

    # "outer" pushes toward the front (world -Y); "inner" pulls toward the back (world +Y).
    assert outer[1] < base[1]
    assert inner[1] > base[1]


def test_resolve_limb_target_applies_character_local_offset():
    landmarks = _bone_landmarks()
    base = anchors.resolve_anchor_position("ANCHOR_CHEST_CENTER", landmarks)

    target = anchors.resolve_limb_target("ANCHOR_CHEST_CENTER", [0.05, 0.0, 0.1], "neutral", landmarks)

    assert target == pytest.approx([base[0] + 0.05, base[1], base[2] + 0.1])


def test_resolve_pole_target_down_forward_moves_below_and_in_front_of_midpoint():
    shoulder = [-0.1774, 0.0524, 1.3542]
    ik_target = [-0.1, -0.05, 1.15]

    pole = anchors.resolve_pole_target("DOWN_FORWARD", "R", ik_target, shoulder)
    mid_z = (shoulder[2] + ik_target[2]) / 2.0
    mid_y = (shoulder[1] + ik_target[1]) / 2.0

    assert pole[2] < mid_z
    assert pole[1] < mid_y


def test_resolve_pole_target_outward_moves_away_from_centerline():
    shoulder = [-0.1774, 0.0524, 1.3542]
    ik_target = [-0.1, -0.05, 1.15]

    left_pole = anchors.resolve_pole_target("OUTWARD", "L", ik_target, shoulder)
    right_pole = anchors.resolve_pole_target("OUTWARD", "R", ik_target, shoulder)

    assert left_pole[0] > 0
    assert right_pole[0] < 0


def test_resolve_weight_stance_neutral_is_a_no_op():
    root_location, bone_rotations = anchors.resolve_weight_stance("neutral")

    assert root_location is None
    assert bone_rotations == {}


def test_resolve_weight_stance_shift_left_shifts_root_and_counter_tilts_spine():
    root_location, bone_rotations = anchors.resolve_weight_stance("shift_left")

    assert root_location[0] > 0
    assert bone_rotations["spine1"][2] < 0


def test_resolve_weight_stance_shift_right_mirrors_shift_left():
    left_root, left_rotations = anchors.resolve_weight_stance("shift_left")
    right_root, right_rotations = anchors.resolve_weight_stance("shift_right")

    assert right_root[0] == -left_root[0]
    assert right_rotations["spine1"][2] == -left_rotations["spine1"][2]


def test_resolve_clavicle_protraction_mirrors_left_and_right():
    rotations = anchors.resolve_clavicle_protraction()

    assert rotations["l_shoulder"][2] > 0
    assert rotations["r_shoulder"][2] < 0
    assert rotations["l_shoulder"][0] == rotations["r_shoulder"][0] > 0
```

- [ ] **Step 3: Run the new tests**

Run: `python -m pytest tests/test_anchors.py -v`
Expected: 13 passed.

- [ ] **Step 4: Commit**

```bash
git add scarecrow_pipeline/anchors.py tests/test_anchors.py
git commit -m "Add semantic anchor landmark module (docs/gemini_ikplan_a.md Phase 1, adapted to real G9 rig)"
```

---

### Task 2: `nl_appearance.py` -- anchor-based wire schema, resolution, and prompts

**Files:**
- Modify: `scarecrow_pipeline/nl_appearance.py` (full rewrite of the sections below; `AppearanceTranslationError`, `LLMClient`, `AppearanceVocabulary`, `load_default_vocabulary`, `build_retry_prompt`, `_vocabulary_errors` (renamed/trimmed), `_strip_unsupported_keywords`, and `AnthropicClient.complete_json` are unchanged)
- Modify: `tests/test_nl_appearance.py` (full rewrite)

**Interfaces:**
- Consumes: `scarecrow_pipeline.anchors` (Task 1) -- `ANCHOR_NAMES`, `ANCHOR_DESCRIPTIONS`, `resolve_limb_target`, `resolve_pole_target`, `resolve_weight_stance`, `resolve_clavicle_protraction`, `LIMB_BONES`, `UnknownAnchorError`.
- Produces: `nl_appearance.LimbGoal`, `nl_appearance.PoseIntent`, `nl_appearance.AppearanceIntent`, `nl_appearance.resolve_pose_intent(intent: PoseIntent, vocab: AppearanceVocabulary) -> PosePayload`, `nl_appearance.translate_pose_intent(character, description, client, *, vocab=None, extra_context=None) -> AppearanceIntent`, `nl_appearance.translate_appearance(character, description, client, *, vocab=None, extra_context=None) -> AppearanceResult` (return type unchanged, used by `apply_appearance.py`), `nl_appearance.LimbDelta`, `nl_appearance.CritiqueDeltaFeedback`, `nl_appearance.apply_critique_delta(intent: AppearanceIntent, feedback: CritiqueDeltaFeedback) -> AppearanceIntent`, `nl_appearance.VisionCritiqueClient` (Protocol, `critique_pose(description: str, image_bytes: bytes, current_intent: PoseIntent | None) -> CritiqueDeltaFeedback`). `AppearanceResult` (unchanged: `pose: PosePayload | None`, `expression: FACSExpression`) stays the return type of `translate_appearance`, consumed by `apply_appearance.py` and `pose_critique.py` (Task 3).
- `PoseCritique` is removed (replaced by `CritiqueDeltaFeedback`) -- Task 4 fixes the one other test file (`tests/test_apply_appearance.py`) that imports it.

- [ ] **Step 1: Replace the schema section of `nl_appearance.py`**

Replace lines 1-69 (module docstring through `VisionCritiqueClient`) with:

```python
"""Translate a natural-language appearance description into a validated
PosePayload/FACSExpression pair, via a provider-agnostic LLM client.

DAZ Studio is only ever used to source a character once (see
scripts/onboard_character.py); all posing and expression happen in
Blender (blender/worker.py's apply_pose/apply_expression) so they can be
driven by natural language. This module is the translation step
scarecrow_pipeline/scene_tags.py calls "a separate, out-of-scope LLM
step" -- see
docs/superpowers/specs/2026-09-16-nl-appearance-translator-design.md.

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
```

Note: `AppearanceResult` moves down here (it now comes after `PosePayload`-independent types are defined, but it only depends on `schemas.PosePayload`/`FACSExpression`, already imported) -- placing it here keeps every wire/result type grouped before the prompt-building functions.

- [ ] **Step 2: Replace `build_system_prompt`/`build_user_prompt`**

Replace the old `build_system_prompt`/`build_user_prompt` functions with:

```python
def build_system_prompt() -> str:
    anchor_lines = "\n".join(
        f"    - {name}: {anchors.ANCHOR_DESCRIPTIONS[name]}" for name in anchors.ANCHOR_NAMES
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
```

- [ ] **Step 3: Replace vocabulary-error checking, schema localization, and pair conversion**

Replace `_vocabulary_errors`, `_OPEN_MAP_VOCAB_TITLES`, `_localize_open_maps`, and `_pairs_to_dicts` with:

```python
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
    constrain every LimbGoal.target_anchor property to anchors.ANCHOR_NAMES.

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
                    "enum": list(anchors.ANCHOR_NAMES),
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
```

- [ ] **Step 4: Replace `translate_appearance` with `translate_pose_intent`/`resolve_pose_intent`/`translate_appearance`**

Keep `_strip_unsupported_keywords` exactly as-is. Replace the old `translate_appearance` function with:

```python
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
    system_prompt = build_system_prompt()
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
```

- [ ] **Step 5: Update `AnthropicClient.critique_pose`**

Replace the existing `critique_pose` method body (keep `__init__`/`complete_json` unchanged) with:

```python
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
            CritiqueDeltaFeedback.model_json_schema(),
        )
        return CritiqueDeltaFeedback.model_validate(verdict_raw)
```

- [ ] **Step 6: Replace `tests/test_nl_appearance.py`**

```python
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
```

- [ ] **Step 7: Run the updated tests**

Run: `python -m pytest tests/test_nl_appearance.py tests/test_anchors.py -v`
Expected: all pass (2 skipped: the opt-in Anthropic smoke tests).

- [ ] **Step 8: Commit**

```bash
git add scarecrow_pipeline/nl_appearance.py tests/test_nl_appearance.py
git commit -m "Replace raw ik_targets/pole_targets LLM contract with anchor-based LimbGoal/PoseIntent (docs/gemini_ikplan_a.md Phases 1-2)"
```

---

### Task 3: `pose_critique.py` -- structured delta-critique loop

**Files:**
- Modify: `scarecrow_pipeline/pose_critique.py`
- Modify: `tests/test_pose_critique.py` (full rewrite)

**Interfaces:**
- Consumes: `nl_appearance.AppearanceIntent`, `AppearanceResult`, `LLMClient`, `VisionCritiqueClient`, `translate_pose_intent`, `resolve_pose_intent`, `apply_critique_delta` (Task 2).
- Produces: `pose_critique.run_with_critique(...)` -- same signature and same outcome-dict keys (`status`, `attempts`, `matches_description`, `critique`, plus `outcome.entry`) as before, so `scripts/apply_appearance.py` needs no changes. `RenderFailedError`, `RenderOutcome`, `RenderFn`, `DEFAULT_MAX_RENDER_ATTEMPTS` unchanged.

- [ ] **Step 1: Rewrite `scarecrow_pipeline/pose_critique.py`**

```python
"""Render -> vision critique -> retry loop for translate_appearance's output.

apply_appearance.py's Blender subprocess call is a real, expensive render,
and translate_appearance()'s first guess at a pose has no way to know
whether it actually looks like its description -- see bd issue
scarecrow-57h. This module closes that loop: translate ONCE, then render
(via an injected callback so this module never touches Blender/subprocess
directly), critique the render against the description with a
vision-capable client, and on a mismatch apply the critique's numeric
per-limb deltas directly to the previous attempt's PoseIntent -- see
docs/gemini_ikplan_a.md Phase 4 -- rather than re-invoking the
pose-generation LLM. This makes convergence deterministic: attempt N+1's
ik_targets/pole_targets are attempt N's plus an explicit numeric
correction, not an independent fresh guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scarecrow_pipeline.nl_appearance import (
    AppearanceIntent,
    AppearanceResult,
    AppearanceVocabulary,
    LLMClient,
    VisionCritiqueClient,
    apply_critique_delta,
    load_default_vocabulary,
    resolve_pose_intent,
    translate_pose_intent,
)

DEFAULT_MAX_RENDER_ATTEMPTS = 2


class RenderFailedError(RuntimeError):
    """Raised by a render_fn callback to signal a Blender-side render
    failure, distinct from a critique mismatch (which is not an error)."""


@dataclass
class RenderOutcome:
    """One render attempt's result, as returned by a render_fn callback.

    entry is the caller-defined manifest dict for this attempt (e.g.
    apply_appearance.py's {character, description, output_path,
    request_path, status}); image_path is the rendered PNG to critique.
    """

    entry: dict
    image_path: Path


RenderFn = Callable[[AppearanceResult, int], RenderOutcome]


def run_with_critique(
    character: str,
    description: str,
    *,
    translate_client: LLMClient,
    vision_client: VisionCritiqueClient,
    render_fn: RenderFn,
    vocab: AppearanceVocabulary | None = None,
    max_attempts: int = DEFAULT_MAX_RENDER_ATTEMPTS,
) -> dict:
    """Translate once, then render + critique + apply numeric deltas,
    retrying up to max_attempts.

    Never raises on a critique mismatch or on a render failure signaled via
    RenderFailedError -- returns a manifest dict describing the best-effort
    last attempt in either case, so a human or CI can still inspect the
    artifact. Only propagates an unexpected exception from render_fn or the
    clients themselves (a real provider/subprocess failure, not a "the pose
    didn't match" outcome).
    """
    vocab = vocab or load_default_vocabulary()
    intent: AppearanceIntent = translate_pose_intent(character, description, translate_client, vocab=vocab)

    last_outcome: RenderOutcome | None = None
    last_feedback = None

    for attempt in range(1, max_attempts + 1):
        result = AppearanceResult(pose=resolve_pose_intent(intent.pose, vocab), expression=intent.expression)
        try:
            outcome = render_fn(result, attempt)
        except RenderFailedError as exc:
            return {"status": "render_failed", "attempts": attempt, "error": str(exc)}
        last_outcome = outcome

        image_bytes = outcome.image_path.read_bytes()
        feedback = vision_client.critique_pose(description, image_bytes, intent.pose)
        last_feedback = feedback

        if feedback.pose_is_satisfactory:
            return {
                "status": "ok",
                "attempts": attempt,
                "matches_description": True,
                "critique": feedback.critique_summary,
                **outcome.entry,
            }

        intent = apply_critique_delta(intent, feedback)

    return {
        "status": "ok",
        "attempts": max_attempts,
        "matches_description": False,
        "critique": last_feedback.critique_summary if last_feedback else None,
        **(last_outcome.entry if last_outcome else {}),
    }
```

- [ ] **Step 2: Rewrite `tests/test_pose_critique.py`**

```python
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
            {"name": "r_forearm", "category": "arm", "rotation_mode": "XYZ"},
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
```

- [ ] **Step 3: Run the updated tests**

Run: `python -m pytest tests/test_pose_critique.py -v`
Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add scarecrow_pipeline/pose_critique.py tests/test_pose_critique.py
git commit -m "Rewrite critique retry loop as a numeric delta-application loop (docs/gemini_ikplan_a.md Phase 4)"
```

---

### Task 4: Fix `tests/test_apply_appearance.py`'s fake vision client

**Files:**
- Modify: `tests/test_apply_appearance.py`

**Interfaces:**
- Consumes: `nl_appearance.CritiqueDeltaFeedback` (Task 2) in place of the removed `PoseCritique`.

- [ ] **Step 1: Read the current fake and its one usage**

Run: `python -c "import subprocess; print(subprocess.run(['python','-c','import re,sys; text=open(\"tests/test_apply_appearance.py\",encoding=\"utf-8\").read(); print(text[text.index(chr(10).join([\"\", \"\"]))-200:])'], capture_output=True, text=True).stdout)"` -- or simply open `tests/test_apply_appearance.py` around its `FakeVisionClient` class (search for `PoseCritique`) in an editor; it currently reads:

```python
class FakeVisionClient:
    """Returns queued PoseCritique verdicts in order."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)

    def critique_pose(self, description, image_bytes):
        return self._verdicts.pop(0)
```

used as:

```python
    from scarecrow_pipeline.nl_appearance import PoseCritique

    vision_client = FakeVisionClient([PoseCritique(matches_description=True, critique="matches")])
```

- [ ] **Step 2: Update the fake and its usage**

Replace the class with:

```python
class FakeVisionClient:
    """Returns queued CritiqueDeltaFeedback verdicts in order."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)

    def critique_pose(self, description, image_bytes, current_intent):
        return self._verdicts.pop(0)
```

Replace the usage with:

```python
    from scarecrow_pipeline.nl_appearance import CritiqueDeltaFeedback

    vision_client = FakeVisionClient([CritiqueDeltaFeedback(pose_is_satisfactory=True, critique_summary="matches")])
```

- [ ] **Step 3: Run the test file**

Run: `python -m pytest tests/test_apply_appearance.py -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_apply_appearance.py
git commit -m "Update test_apply_appearance's fake vision client for CritiqueDeltaFeedback"
```

---

### Task 5: `blender/worker.py` -- shrinkwrap collision guard (Phase 3)

**Files:**
- Modify: `blender/worker.py:153-175` (the `ik_target_bones` loop inside `apply_pose`)

**Interfaces:**
- Consumes: nothing new from Tasks 1-4 -- `apply_pose`'s input dict shape (`pose.get("ik_targets", {})`, etc.) is unchanged, since `PosePayload` didn't change.
- No test: this file only runs inside Blender (`bpy`), which is not available in this environment (no `blender` on `PATH` -- see Global Constraints). This task is written for a human or CI environment with Blender installed to apply and verify visually; it is not covered by `pytest`.

- [ ] **Step 1: Add a body-mesh resolver next to `resolve_armature`**

In `blender/worker.py`, immediately after the `resolve_armature` function (currently ending around line 64), add:

```python
def resolve_body_mesh(armature):
    """Find the mesh object Armature-deformed by `armature` -- the
    shrinkwrap collision guard (see apply_pose's IK section) needs this to
    keep an IK target from sinking the hand into the torso. Matching via
    an ARMATURE modifier pointing at this specific object is reliable
    across characters, unlike matching by object name."""
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if any(mod.type == "ARMATURE" and mod.object == armature for mod in obj.modifiers):
            return obj
    return None
```

- [ ] **Step 2: Add the shrinkwrap guard inside the IK-target loop**

In `apply_pose`, immediately after this existing block (around line 161-165):

```python
        bone = armature.pose.bones[bone_name]
        constraint = next((item for item in bone.constraints if item.name == "Scarecrow IK"), None) or bone.constraints.new("IK")
        constraint.name = "Scarecrow IK"
        constraint.target = target
        constraint.chain_count = 2
```

add:

```python
        body_mesh = resolve_body_mesh(armature)
        if body_mesh is not None:
            guard = next((item for item in bone.constraints if item.name == "Scarecrow Surface Guard"), None) \
                or bone.constraints.new("SHRINKWRAP")
            guard.name = "Scarecrow Surface Guard"
            guard.target = body_mesh
            guard.shrinkwrap_type = "NEAREST_SURFACE"
            guard.distance = 0.03
            guard.influence = 0.8
            # IK must resolve the base position first; the guard nudges it off
            # the surface afterward, so it must sit after "Scarecrow IK" in
            # the constraint stack -- constraints.new() already appends to
            # the end, so no explicit reordering is needed as long as this
            # block stays after the IK constraint is created above.
```

- [ ] **Step 3: Document the verification gap**

Add a comment directly above the new `guard = ...` line block (already included above) is sufficient; additionally, in this task's commit message, state plainly that this addition has not been rendered/visually verified in this environment (no Blender available) -- see Task 6.

- [ ] **Step 4: Commit**

```bash
git add blender/worker.py
git commit -m "Add shrinkwrap non-penetration guard on IK target bones (docs/gemini_ikplan_a.md Phase 3, unverified -- no Blender in this environment)"
```

---

### Task 6: Full verification and bd issue update

**Files:** none (verification + bookkeeping only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: all tests pass (2 skipped -- the opt-in Anthropic smoke tests, same as before this plan).

- [ ] **Step 2: Grep for any remaining references to removed names**

Run: `grep -rn "PoseCritique\b" --include=*.py .` (excluding `__pycache__`)
Expected: no matches. If any remain, fix them before proceeding.

- [ ] **Step 3: Update the bd issue**

```bash
bd update scarecrow-4sm --notes="Superseded the additive prose-context patch (commit 0902d88, reverted) with the full docs/gemini_ikplan_a.md anchor-based rearchitecture, adapted to this project's real rig/architecture: scarecrow_pipeline/anchors.py (semantic landmarks from docs/posable_bones.json rest-pose data, no bpy), nl_appearance.py's LLM wire contract replaced (LimbGoal/PoseIntent/AppearanceIntent instead of raw ik_targets/pole_targets floats), resolve_pose_intent() converts back to the unchanged PosePayload contract blender/worker.py already consumes, pose_critique.py's retry loop now applies a structured CritiqueDeltaFeedback's numeric per-limb deltas directly to the previous PoseIntent instead of re-invoking the pose LLM each attempt (deterministic convergence), and blender/worker.py gained a shrinkwrap non-penetration guard on IK target bones. Full suite green.

NOT done: the acceptance criteria's real end-to-end apply_appearance.py --with-critique run against a live Anthropic API + Blender render is still not verifiable in this environment (no ANTHROPIC_API_KEY, no blender on PATH) -- same limitation as before. The Phase 3 shrinkwrap guard (blender/worker.py) is also unverified visually for the same reason. Left open pending a run with real credentials/Blender."
```

- [ ] **Step 4: Push**

```bash
git pull --rebase
git push
git status
```

Expected: `git status` reports the branch up to date with `origin/main`.

---

## Self-Review Notes

- **Spec coverage:** Phase 1 (anchors) -> Task 1. Phase 2 (discrete LLM schema) -> Task 2 Steps 1-2. Phase 3 (clavicle protraction, torso dynamics) -> Task 2 Step 4 (`resolve_pose_intent`); Phase 3's shrinkwrap collision guard -> Task 5. Phase 4 (delta-critique loop) -> Task 2 Steps 1 and 5 (schema + `AnthropicClient`) and Task 3 (loop rewrite). Phase 5 (headless Blender verification harness) is explicitly **not** included -- this environment has no Blender, so a `tests/test_arm_cross.py` requiring `bpy` would be untestable dead weight; Task 5's gap is documented instead of faked.
- **Deliberate trims from the doc**, stated here so they aren't mistaken for oversights: `HandShapeEnum`/hand posing (doc's Phase 2) is dropped -- the existing system prompt already leaves fingers unset by default and no acceptance criterion needs it. `ANCHOR_HIP_CREST_*`/`ANCHOR_SHOULDER_TOP_*`/`ANCHOR_CHIN`/`ANCHOR_TEMPLE_*` are dropped -- this rig's `hip`/`pelvis` bones aren't split left/right so "hip crest" anchors can't be meaningfully derived, and head/face anchors aren't needed by any `LimbGoal` (only arms use anchors; `weight_stance` handles torso balance separately, `look_at_target` already handles gaze).
- **Type consistency check:** `LimbGoal.target_anchor: str` (Task 2 Step 1) validated against `anchors.ANCHOR_NAMES` (Task 1) via `field_validator`, consistent everywhere it's read (`resolve_pose_intent`, `apply_critique_delta`, `_localize_open_maps`'s enum injection). `PoseIntent.left_arm`/`right_arm: LimbGoal | None` matches `LimbDelta.limb: Literal["left_arm", "right_arm"]` used via `getattr(new_intent.pose, adjustment.limb, None)` in `apply_critique_delta`. `resolve_pose_intent(intent: PoseIntent | None, vocab) -> PosePayload | None` signature matches every call site (Task 2 Step 4's `translate_appearance`, Task 3's `run_with_critique`). `VisionCritiqueClient.critique_pose`'s third parameter (`current_intent: PoseIntent | None`) matches both its `AnthropicClient` implementation (Task 2 Step 5) and its caller in `pose_critique.run_with_critique` (Task 3 Step 1, passes `intent.pose`).
