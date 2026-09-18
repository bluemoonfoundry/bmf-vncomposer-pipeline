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
