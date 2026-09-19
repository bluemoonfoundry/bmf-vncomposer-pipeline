"""Strict contracts shared by LLM orchestration, vision extraction, and Blender."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Weight = Annotated[float, Field(ge=0.0, le=1.0)]
Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class FACSExpression(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weights: dict[str, Weight] = Field(default_factory=dict)


class LimbGoalPayload(BaseModel):
    """One arm's IK goal, expressed as a semantic anchor + small offset --
    NOT a pre-resolved world-space point. blender/worker.py resolves this
    LIVE, from the character's currently-posed bone matrices (after this
    same PosePayload's bone_rotations/root_location are applied and the
    depsgraph is updated), not from static rest-pose data.

    This exists because resolving the anchor/pole world position ahead of
    time (in plain Python, from docs/posable_bones.json's REST bone
    positions -- the original approach) goes stale the moment a torso/spine/
    clavicle bone_rotation is applied in the same pose: the shoulder is no
    longer where the rest-pose data says it is, so a world-space IK target
    computed from that stale position drifts away from the actual (now
    rotated) shoulder/chest, producing an asymmetric, not-actually-crossed
    arm pose -- see scarecrow-5mn."""

    model_config = ConfigDict(extra="forbid")
    target_anchor: str = Field(..., description="One of scarecrow_pipeline.anchors.ANCHOR_NAMES.")
    character_local_offset: Vector3 = Field(default=[0.0, 0.0, 0.0])
    layer_depth: Literal["inner", "outer", "neutral"] = "neutral"
    elbow_strategy: Literal["DOWN_FORWARD", "OUTWARD", "DOWN_PINNED", "UP_FLUID"] = "DOWN_FORWARD"


class PosePayload(BaseModel):
    """Body pose, expressed as direct Blender bone control.

    DAZ Studio is only ever used to source the base character once; all
    posing happens in Blender so it can be driven by natural language. Bone
    names must match the imported rig's posable (non-"(drv)") pose bone
    names, e.g. "hip", "r_upperarm", "l_forearm" -- see
    docs/posable_bones.json for the full generated list.
    """

    model_config = ConfigDict(extra="forbid")
    bone_rotations: dict[str, Vector3] = Field(
        default_factory=dict,
        description="Bone name -> XYZ Euler rotation in radians, applied directly to pose.bones[name].rotation_euler.",
    )
    root_location: Vector3 | None = Field(
        default=None,
        description="Optional world-space-ish offset applied to the root ('hip') bone's location, for crouching/leaning/repositioning.",
    )
    ik_targets: dict[str, Vector3] = Field(
        default_factory=dict,
        description=(
            "Bone name -> world-space IK end-effector point, applied as-is with "
            "no further resolution. Superseded by limb_goals for the same bone "
            "name -- prefer limb_goals for anything anchor-based (arm gestures), "
            "since a raw world-space point here is computed before this pose's "
            "own bone_rotations are applied and can't account for them."
        ),
    )
    pole_targets: dict[str, Vector3] = Field(
        default_factory=dict,
        description=(
            "Bone name -> world-space pole (elbow/knee direction) point, for the "
            "2-bone IK chain whose end-effector target is ik_targets[same bone "
            "name]. A pole_targets entry with no matching ik_targets entry is "
            "meaningless and will be rejected. Ignored for a bone also present "
            "in limb_goals (its pole is resolved live instead)."
        ),
    )
    limb_goals: dict[str, LimbGoalPayload] = Field(
        default_factory=dict,
        description=(
            "IK target bone name (e.g. 'l_forearm') -> LimbGoalPayload, resolved "
            "LIVE in blender/worker.py from the character's currently-posed bone "
            "matrices, not from static rest-pose data -- see LimbGoalPayload's "
            "docstring and scarecrow-5mn. Takes priority over ik_targets/"
            "pole_targets for the same bone name."
        ),
    )
    look_at_target: Vector3 | None = Field(
        default=None,
        description=(
            "World-space point for the head and both eyes to look at, via a "
            "DAMPED_TRACK constraint on each. Bypasses those bones' imported "
            "anatomical LIMIT_ROTATION range (the track constraint evaluates "
            "after it in the constraint stack), so keep targets within a "
            "plausible forward-facing cone. Omit/None to clear any existing look-at."
        ),
    )

    @field_validator("bone_rotations", "ik_targets", "pole_targets")
    @classmethod
    def finite_vectors(cls, value: dict[str, list[float]]) -> dict[str, list[float]]:
        if any(not all(isinstance(x, (int, float)) for x in vector) for vector in value.values()):
            raise ValueError("Vectors must contain only numbers")
        return value


class SceneLighting(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # direction is the direction light TRAVELS. The default camera looks
    # from (0, -8, 1.6) toward the origin, so a subject facing the camera
    # has an outward normal pointing toward -Y -- a light must travel
    # toward +Y to hit that face rather than the subject's back. intensity
    # is a Cycles Sun's irradiance in W/m^2; 15.0 was picked empirically
    # (blender/worker.py's configure_lighting) to produce a visibly lit
    # render, since Blender's own new-sun default of 1.0 is nearly black
    # once combined with this scene's 0.35-strength ambient world light.
    direction: Vector3 = [0.0, 1.0, -1.0]
    color_rgb: tuple[Weight, Weight, Weight] = (1.0, 1.0, 1.0)
    ambient_rgb: tuple[Weight, Weight, Weight] = (0.5, 0.5, 0.5)
    intensity: float = Field(default=15.0, ge=0.0)
    confidence: Weight = 0.0


class CameraMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    projection: Literal["PERSP", "ORTHO"] = "PERSP"
    focal_length_mm: float = Field(default=50.0, gt=0.0, le=300.0)
    location: Vector3 = [0.0, -8.0, 1.6]
    rotation_euler: Vector3 = [1.5708, 0.0, 0.0]


class RenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    character: str = Field(min_length=1)
    master_blend: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    background_path: str | None = None
    outfit: str | None = None
    hair: str | None = None
    emotion: str | None = None
    pose: PosePayload | None = None
    expression: FACSExpression = Field(default_factory=FACSExpression)
    lighting: SceneLighting = Field(default_factory=SceneLighting)
    camera: CameraMetadata = Field(default_factory=CameraMetadata)
    render_samples: int = Field(default=64, ge=1, le=4096)
    transparent: bool = True


class CharacterPlacement(BaseModel):
    """One character's identity, look, and pose within a multi-character scene."""

    model_config = ConfigDict(extra="forbid")
    character: str = Field(min_length=1)
    master_blend: str = Field(min_length=1)
    outfit: str | None = None
    hair: str | None = None
    pose: PosePayload | None = None
    expression: FACSExpression = Field(default_factory=FACSExpression)
    location: Vector3 = Field(
        default=[0.0, 0.0, 0.0],
        description="World-space placement for this character's root, distinct from PosePayload.root_location (a pose-relative crouch/lean offset).",
    )
    rotation_euler: Vector3 = [0.0, 0.0, 0.0]
    scale: float = Field(default=1.0, gt=0.0)


class ImagePlateBackground(BaseModel):
    """2D background: compositor.py's AlphaOver composites characters over a still plate (or transparent)."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["image_plate"] = "image_plate"
    background_path: str | None = None
    transparent: bool = True


class Environment3DBackground(BaseModel):
    """3D background: characters are placed directly into a pre-imported Phase 4 environment collection; no 2D compositing."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["3d_environment"] = "3d_environment"
    environment_collection: str = Field(
        min_length=1,
        description="Name of the Environment_<name> collection produced by blender/import_environment_asset.py.",
    )


BackgroundSpec = Annotated[
    ImagePlateBackground | Environment3DBackground, Field(discriminator="mode")
]


class SceneRequest(BaseModel):
    """Multi-character CG scene composition contract consumed by blender/worker.py:render_scene()."""

    model_config = ConfigDict(extra="forbid")
    characters: list[CharacterPlacement] = Field(min_length=1)
    camera: CameraMetadata = Field(default_factory=CameraMetadata)
    lighting: SceneLighting = Field(default_factory=SceneLighting)
    background: BackgroundSpec = Field(default_factory=ImagePlateBackground)
    output_path: str = Field(min_length=1)
    render_samples: int = Field(default=64, ge=1, le=4096)
