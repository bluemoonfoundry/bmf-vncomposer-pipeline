# Engineering Spec: Affordance Anchors & Delta Critique Loop for Blender/G9
**Tracking Issue:** `scarecrow-4sm`  
**Target Rig:** DAZ Genesis 9 (Diffeomorphic Blender Import / MHX or Rigify conversion)  
**Objective:** Replace ungrounded 3D coordinate estimation with parametric rig anchors, collision constraints, and localized vector-delta feedback loops.

---

## Directory Structure & New Modules

Ensure your pipeline module has the following layout for the posing subsystem:

```text
bmf-vncomposer-pipeline/
├── posing/
│   ├── __init__.py
│   ├── anchors.py             # Phase 1: Semantic surface anchors & local coordinate frames
│   ├── schemas.py             # Phase 2: Pydantic schemas for discrete anchor selection
│   ├── ik_runtime.py          # Phase 3: IK positioning, shrinkwrap, and clavicle protraction
│   ├── critique_optimizer.py  # Phase 4: Local delta aggregator & state machine
│   └── prompts/
│       ├── pose_planner.jinja2
│       └── pose_critique.jinja2
└── tests/
    └── test_arm_cross.py      # Automated headless Blender verification script
```

---

## Phase 1: Semantic Surface Anchors & Coordinate Transform Module

Create `posing/anchors.py`. This module handles:
1. Detecting or injecting parented "Anchor Empties" on the Diffeomorphic G9 skeleton.
2. Converting between **Character-Local Space** and **Blender World Space**.

```python
# posing/anchors.py
import bpy
from mathutils import Vector, Matrix
from typing import Dict, Tuple

# Pre-defined relative offsets in bone-local space (Rest Pose)
# Format: {Anchor_Name: (parent_bone_name, local_offset_vector)}
DEFAULT_ANCHOR_DEFINITIONS: Dict[str, Tuple[str, Vector]] = {
    # Torso
    "ANCHOR_CHEST_CENTER": ("chest", Vector((0.0, 0.12, 0.05))),
    "ANCHOR_PECTORAL_L": ("chest", Vector((0.08, 0.11, 0.05))),
    "ANCHOR_PECTORAL_R": ("chest", Vector((-0.08, 0.11, 0.05))),
    "ANCHOR_RIBCAGE_MID": ("spine", Vector((0.0, 0.10, 0.0))),
    "ANCHOR_HIP_CREST_L": ("pelvis", Vector((0.16, 0.0, -0.02))),
    "ANCHOR_HIP_CREST_R": ("pelvis", Vector((-0.16, 0.0, -0.02))),
    # Arms / Shoulders
    "ANCHOR_SHOULDER_TOP_L": ("upper_arm.L", Vector((0.0, 0.0, 0.05))),
    "ANCHOR_SHOULDER_TOP_R": ("upper_arm.R", Vector((0.0, 0.0, 0.05))),
    "ANCHOR_BICEP_LATERAL_L": ("upper_arm.L", Vector((0.06, 0.0, -0.12))),
    "ANCHOR_BICEP_LATERAL_R": ("upper_arm.R", Vector((-0.06, 0.0, -0.12))),
    "ANCHOR_FOREARM_VENTRAL_L": ("forearm.L", Vector((0.0, 0.04, -0.10))),
    "ANCHOR_FOREARM_VENTRAL_R": ("forearm.R", Vector((0.0, 0.04, -0.10))),
    # Head / Face
    "ANCHOR_CHIN": ("head", Vector((0.0, 0.09, -0.05))),
    "ANCHOR_TEMPLE_R": ("head", Vector((-0.09, 0.02, 0.04))),
    "ANCHOR_TEMPLE_L": ("head", Vector((0.09, 0.02, 0.04))),
}

def ensure_semantic_anchors(armature_obj: bpy.types.Object) -> Dict[str, bpy.types.Object]:
    """
    Spawns or verifies parented Empty objects for each landmark anchor.
    Anchors move automatically when parent bones animate/deform.
    """
    created_anchors = {}
    
    # Ensure a dedicated collection exists
    anchor_col = bpy.data.collections.get("Rig_Anchors")
    if not anchor_col:
        anchor_col = bpy.data.collections.new("Rig_Anchors")
        bpy.context.scene.collection.children.link(anchor_col)

    for anchor_name, (bone_name, offset) in DEFAULT_ANCHOR_DEFINITIONS.items():
        if bone_name not in armature_obj.pose.bones:
            continue

        empty_name = f"ANCHOR_{armature_obj.name}_{anchor_name}"
        empty_obj = bpy.data.objects.get(empty_name)

        if not empty_obj:
            empty_obj = bpy.data.objects.new(empty_name, None)
            empty_obj.empty_display_type = 'SPHERE'
            empty_obj.empty_display_size = 0.02
            anchor_col.objects.link(empty_obj)

            # Parent to the specific bone
            empty_obj.parent = armature_obj
            empty_obj.parent_type = 'BONE'
            empty_obj.parent_bone = bone_name
            empty_obj.location = offset
        
        created_anchors[anchor_name] = empty_obj

    return created_anchors

def get_character_local_matrix(armature_obj: bpy.types.Object) -> Matrix:
    """
    Returns the transformation matrix defining the Character Coordinate Frame:
      +X : Character's Left
      -X : Character's Right
      +Y : Character's Front (Sagittal Anterior)
      -Y : Character's Back (Sagittal Posterior)
      +Z : Character's Up (Superior)
    """
    root_bone = armature_obj.pose.bones.get("root") or armature_obj.pose.bones.get("pelvis")
    if root_bone:
        return armature_obj.matrix_world @ root_bone.matrix
    return armature_obj.matrix_world

def local_offset_to_world(armature_obj: bpy.types.Object, offset_vec: Vector) -> Vector:
    """Transforms a delta offset from Character-Local space to Blender World space."""
    char_matrix = get_character_local_matrix(armature_obj)
    rotation_scale = char_matrix.to_3x3()
    return rotation_scale @ offset_vec
```

---

## Phase 2: LLM JSON Schema & Translation Layer

Create `posing/schemas.py`. Disallow arbitrary world $[x, y, z]$ floats in the output. The LLM must choose a discrete semantic anchor, a layer depth (outer vs. inner for arm clearance), and a character-local micro-offset.

```python
# posing/schemas.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

AnchorEnum = Literal[
    "ANCHOR_CHEST_CENTER",
    "ANCHOR_PECTORAL_L",
    "ANCHOR_PECTORAL_R",
    "ANCHOR_RIBCAGE_MID",
    "ANCHOR_HIP_CREST_L",
    "ANCHOR_HIP_CREST_R",
    "ANCHOR_BICEP_LATERAL_L",
    "ANCHOR_BICEP_LATERAL_R",
    "ANCHOR_FOREARM_VENTRAL_L",
    "ANCHOR_FOREARM_VENTRAL_R",
    "ANCHOR_CHIN",
    "ANCHOR_TEMPLE_R",
    "ANCHOR_TEMPLE_L"
]

ElbowPoleStrategy = Literal[
    "DOWN_FORWARD",  # Standard crossed arms / resting on stomach
    "OUTWARD",       # T-pose, fighting stance
    "DOWN_PINNED",   # Hands at sides
    "UP_FLUID"       # Touching head / leaning
]

HandShapeEnum = Literal[
    "RELAXED_CUP",
    "FIST_SOFT",
    "OPEN_FLAT_TOUCH",
    "POINTING_INDEX",
    "GRASP_BICEP"
]

class LimbGoal(BaseModel):
    target_anchor: AnchorEnum = Field(..., description="Target anatomical surface anchor.")
    character_local_offset: List[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="Offset [dx, dy, dz] in meters in character space (+X=Left, +Y=Front, +Z=Up). Keep values between -0.15 and 0.15."
    )
    layer_depth: Literal["inner", "outer", "neutral"] = Field(
        default="neutral",
        description="For crossed arms/legs. 'outer' limb is pushed +4cm forward automatically to clear the other limb."
    )
    elbow_strategy: ElbowPoleStrategy = Field(default="DOWN_FORWARD")
    hand_shape: HandShapeEnum = Field(default="RELAXED_CUP")

class FullBodyPoseIntent(BaseModel):
    pose_name: str
    weight_stance: Literal["neutral", "shift_left", "shift_right"] = Field(default="neutral")
    spine_twist_degrees: float = Field(default=0.0, description="Rotation of upper torso relative to hips.")
    left_arm: Optional[LimbGoal] = None
    right_arm: Optional[LimbGoal] = None
```

Prompt Template: `posing/prompts/pose_planner.jinja2`:
```jinja2
You are a 3D Cinematic Director & Kinematics Engine.
Generate an intentional body pose for a character matching this prompt:
"{{ user_prompt }}"

Rules:
1. NEVER guess raw world coordinates. Pick a semantic `target_anchor` from the schema.
2. For "crossed arms":
   - One arm MUST be `layer_depth: "outer"` and the other `layer_depth: "inner"`.
   - The outer arm hand typically targets the opposite bicep (`ANCHOR_BICEP_LATERAL_*`).
   - The inner arm hand typically targets under the opposite forearm or ribcage.
   - Set `elbow_strategy: "DOWN_FORWARD"`.
   - Set `hand_shape: "GRASP_BICEP"` or "RELAXED_CUP" (never clenched fists unless angry).
3. Weight shift: Natural poses are rarely symmetric. Shift weight to left or right leg.

Emit strictly conforming JSON adhering to the FullBodyPoseIntent schema.
```

---

## Phase 3: IK Runtime, Surface Snapping & Clavicle Logic

Create `posing/ik_runtime.py`. This executes the resolved targets in Blender and uses Blender constraints to prevent torso penetration and auto-protract clavicles.

```python
# posing/ik_runtime.py
import bpy
from mathutils import Vector, Euler
from math import radians
from .schemas import FullBodyPoseIntent, LimbGoal
from .anchors import ensure_semantic_anchors, local_offset_to_world

def apply_pose_to_g9(
    armature_obj: bpy.types.Object,
    body_mesh_obj: bpy.types.Object,
    pose_intent: FullBodyPoseIntent
):
    anchors = ensure_semantic_anchors(armature_obj)
    
    # 1. Weight shift and spine line of action
    _apply_torso_dynamics(armature_obj, pose_intent)

    # 2. Clavicle Protraction (Crucial for crossing arms without socket distortion)
    if pose_intent.left_arm and pose_intent.right_arm:
        if pose_intent.left_arm.layer_depth in ["inner", "outer"] and \
           pose_intent.right_arm.layer_depth in ["inner", "outer"]:
            _protract_clavicles(armature_obj, forward_deg=12.0, elevation_deg=6.0)

    # 3. Apply Left Arm
    if pose_intent.left_arm:
        _apply_limb_ik(
            armature_obj=armature_obj,
            body_mesh_obj=body_mesh_obj,
            limb="L",
            goal=pose_intent.left_arm,
            anchors=anchors
        )

    # 4. Apply Right Arm
    if pose_intent.right_arm:
        _apply_limb_ik(
            armature_obj=armature_obj,
            body_mesh_obj=body_mesh_obj,
            limb="R",
            goal=pose_intent.right_arm,
            anchors=anchors
        )
        
    bpy.context.view_layer.update()

def _apply_limb_ik(
    armature_obj: bpy.types.Object,
    body_mesh_obj: bpy.types.Object,
    limb: str,
    goal: LimbGoal,
    anchors: dict
):
    # Locate Diffeomorphic IK target and Pole bone controls
    # Default Diffeomorphic / MHX rig bone naming:
    ik_target_bone = armature_obj.pose.bones.get(f"hand.ik.{limb}") or armature_obj.pose.bones.get(f"IK_hand.{limb}")
    pole_bone = armature_obj.pose.bones.get(f"elbow_pole.{limb}") or armature_obj.pose.bones.get(f"pole_elbow.{limb}")

    if not ik_target_bone or goal.target_anchor not in anchors:
        return

    anchor_obj = anchors[goal.target_anchor]
    
    # Base target position from semantic anchor
    target_pos = anchor_obj.matrix_world.translation.copy()

    # Apply character-local offsets
    local_offset = Vector(goal.character_local_offset)
    
    # Auto-separate forearm collision via layer depth
    if goal.layer_depth == "outer":
        local_offset.y += 0.05  # Push forward 5cm along character anterior axis
    elif goal.layer_depth == "inner":
        local_offset.y -= 0.02  # Tuck closer to chest

    world_offset = local_offset_to_world(armature_obj, local_offset)
    final_ik_pos = target_pos + world_offset

    # Position the IK control
    ik_target_bone.matrix.translation = final_ik_pos

    # Attach/Update Shrinkwrap Constraint on IK bone to prevent mesh penetration
    _ensure_surface_limit(armature_obj, ik_target_bone, body_mesh_obj)

    # Resolve Elbow Pole Vector
    if pole_bone:
        _position_pole(armature_obj, pole_bone, ik_target_bone, limb, goal.elbow_strategy)

def _ensure_surface_limit(armature_obj: bpy.types.Object, pose_bone: bpy.types.PoseBone, body_mesh: bpy.types.Object):
    """Adds a non-penetration Limit Distance constraint against the deformed body surface."""
    constraint_name = "Surface_Collision_Guard"
    con = pose_bone.constraints.get(constraint_name)
    if not con:
        con = pose_bone.constraints.new(type='SHRINKWRAP')
        con.name = constraint_name
        con.target = body_mesh
        con.shrinkwrap_type = 'NEAREST_SURFACE'
        con.distance = 0.03  # Maintain 3cm distance outside the skin
        con.influence = 0.8  # Soft blend to avoid sudden popping

def _position_pole(armature_obj, pole_bone, ik_bone, limb: str, strategy: str):
    shoulder = armature_obj.pose.bones.get(f"upper_arm.{limb}")
    if not shoulder:
        return
    
    s_pos = shoulder.matrix.translation
    h_pos = ik_bone.matrix.translation
    mid = (s_pos + h_pos) * 0.5

    char_matrix = armature_obj.matrix_world
    forward = (char_matrix.to_3x3() @ Vector((0, 1, 0))).normalized()
    down = (char_matrix.to_3x3() @ Vector((0, 0, -1))).normalized()
    outward = (char_matrix.to_3x3() @ Vector((1 if limb == 'L' else -1, 0, 0))).normalized()

    if strategy == "DOWN_FORWARD":
        pole_bone.matrix.translation = mid + (down * 0.25) + (forward * 0.15)
    elif strategy == "OUTWARD":
        pole_bone.matrix.translation = mid + (outward * 0.35)
    elif strategy == "DOWN_PINNED":
        pole_bone.matrix.translation = mid + (down * 0.35)
    else:
        pole_bone.matrix.translation = mid + (down * 0.20)

def _protract_clavicles(armature_obj: bpy.types.Object, forward_deg: float, elevation_deg: float):
    for limb, sign in [("L", 1), ("R", -1)]:
        clavicle = armature_obj.pose.bones.get(f"clavicle.{limb}")
        if clavicle:
            clavicle.rotation_mode = 'XYZ'
            clavicle.rotation_euler = Euler((
                radians(elevation_deg),
                0.0,
                radians(forward_deg * sign)
            ), 'XYZ')

def _apply_torso_dynamics(armature_obj: bpy.types.Object, intent: FullBodyPoseIntent):
    pelvis = armature_obj.pose.bones.get("pelvis")
    if pelvis and intent.weight_stance != "neutral":
        shift_amount = 0.04 if intent.weight_stance == "shift_left" else -0.04
        pelvis.location.x += shift_amount
        # Counter-tilt spine
        spine = armature_obj.pose.bones.get("spine")
        if spine:
            spine.rotation_mode = 'XYZ'
            spine.rotation_euler.z = radians(-3.0 if intent.weight_stance == "shift_left" else 3.0)
```

---

## Phase 4: Local Delta-Critique Loop

Create `posing/critique_optimizer.py`. This state machine retains the current pose state and sends explicit local coordinate delta instructions to the Vision LLM.

```python
# posing/critique_optimizer.py
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from .schemas import FullBodyPoseIntent

class LimbDelta(BaseModel):
    limb: str = Field(..., description="'left_arm' or 'right_arm'")
    delta_meters: List[float] = Field(
        ...,
        description="Relative adjustment in meters [dx, dy, dz] in Character Space: +X=Left, +Y=Front, +Z=Up."
    )
    change_anchor: Optional[str] = Field(None, description="Optional new Anchor Enum if target is completely off.")
    notes: str

class CritiqueDeltaFeedback(BaseModel):
    pose_is_satisfactory: bool
    critique_summary: str
    adjustments: List[LimbDelta]

def apply_critique_delta(current_pose: FullBodyPoseIntent, feedback: CritiqueDeltaFeedback) -> FullBodyPoseIntent:
    """
    Applies corrective vector deltas iteratively without re-deriving coordinates from scratch.
    """
    new_pose = current_pose.copy(deep=True)

    for adj in feedback.adjustments:
        limb_goal = getattr(new_pose, adj.limb, None)
        if not limb_goal:
            continue

        if adj.change_anchor:
            limb_goal.target_anchor = adj.change_anchor

        # Apply simple vector addition in local character space
        limb_goal.character_local_offset[0] += adj.delta_meters[0]
        limb_goal.character_local_offset[1] += adj.delta_meters[1]
        limb_goal.character_local_offset[2] += adj.delta_meters[2]

        # Clamp offsets to safe ranges (-20cm to +20cm from landmark)
        for i in range(3):
            limb_goal.character_local_offset[i] = max(-0.20, min(0.20, limb_goal.character_local_offset[i]))

    return new_pose
```

Prompt Template: `posing/prompts/pose_critique.jinja2`:
```jinja2
You are an expert 3D Pose Supervisor. Analyze the rendered character image against this intent:
"{{ user_prompt }}"

Current Kinematic Configuration:
- Left Arm Target: {{ pose_state.left_arm.target_anchor }} | Local Offset: {{ pose_state.left_arm.character_local_offset }}
- Right Arm Target: {{ pose_state.right_arm.target_anchor }} | Local Offset: {{ pose_state.right_arm.character_local_offset }}

Coordinate Direction Reference (Character-Centric):
- +X: Move toward Character's Left   | -X: Move toward Character's Right
- +Y: Move Forward (away from chest) | -Y: Move Backward (closer to chest)
- +Z: Move Upward                   | -Z: Move Downward

Task:
If limbs are not positioned correctly (e.g. arms not crossing, hand clipping into body, forearm in wrong depth):
1. Specify corrective deltas in METERS (e.g., [0.0, 0.03, -0.04] to nudge 3cm forward and 4cm down).
2. Do NOT hallucinate raw world coordinates. Only produce deltas ($\Delta X, \Delta Y, \Delta Z$).
3. Output strictly adhering to the `CritiqueDeltaFeedback` JSON schema.
```

---

## Phase 5: Verification & Headless Blender Test Harness

Create `tests/test_arm_cross.py` to allow Claude Code to test the pipeline headlessly without UI interaction:

```python
# tests/test_arm_cross.py
import bpy
import sys
from posing.schemas import FullBodyPoseIntent, LimbGoal
from posing.ik_runtime import apply_pose_to_g9
from posing.critique_optimizer import CritiqueDeltaFeedback, LimbDelta, apply_critique_delta

def run_test():
    # 1. Locate character in test scene
    armature = bpy.data.objects.get("Genesis 9") or bpy.data.objects.get("Armature")
    body_mesh = bpy.data.objects.get("Genesis 9 Mesh") or bpy.data.objects.get("Body")

    assert armature is not None, "Genesis 9 armature not found in scene."
    assert body_mesh is not None, "Genesis 9 body mesh not found in scene."

    # 2. Instantiate intent for Crossed Arms
    intent = FullBodyPoseIntent(
        pose_name="arms_crossed_test",
        weight_stance="shift_right",
        left_arm=LimbGoal(
            target_anchor="ANCHOR_BICEP_LATERAL_R",
            character_local_offset=[0.0, 0.02, 0.0],
            layer_depth="outer",
            elbow_strategy="DOWN_FORWARD"
        ),
        right_arm=LimbGoal(
            target_anchor="ANCHOR_BICEP_LATERAL_L",
            character_local_offset=[0.0, -0.01, -0.02],
            layer_depth="inner",
            elbow_strategy="DOWN_FORWARD"
        )
    )

    # 3. Apply base pose
    print("[TEST] Applying Base Arm-Cross Pose...")
    apply_pose_to_g9(armature, body_mesh, intent)

    # 4. Simulate a Delta Critique step (e.g., Vision model says right arm is too low)
    mock_feedback = CritiqueDeltaFeedback(
        pose_is_satisfactory=False,
        critique_summary="Raise right arm 3cm and move 2cm inward.",
        adjustments=[
            LimbDelta(
                limb="right_arm",
                delta_meters=[0.02, 0.0, 0.03],
                notes="Shift up and in"
            )
        ]
    )

    print("[TEST] Applying Delta Critique Update...")
    updated_intent = apply_critique_delta(intent, mock_feedback)
    apply_pose_to_g9(armature, body_mesh, updated_intent)

    # 5. Check convergence distance between hands and anchors
    left_hand = armature.pose.bones.get("hand.ik.L") or armature.pose.bones.get("IK_hand.L")
    assert left_hand is not None, "IK bone not found."
    print(f"[TEST] SUCCESS. Final Left Hand Position: {left_hand.matrix.translation}")

if __name__ == "__main__":
    run_test()
```

Run test via command line:
```bash
blender -b path/to/your_test_scene.blend --python tests/test_arm_cross.py
```

---

## Action Plan for Claude Code

When prompted to execute this task, Claude Code should perform the following steps sequentially:

1. **Inspect Blender Rig**: Open the target `.blend` file or inspecting script to verify bone naming convention (`hand.ik.L` vs `IK_hand.L`, `chest` vs `spine_02`). Adjust `DEFAULT_ANCHOR_DEFINITIONS` in `posing/anchors.py` accordingly.
2. **Implement Files**: Create `posing/anchors.py`, `posing/schemas.py`, `posing/ik_runtime.py`, and `posing/critique_optimizer.py`.
3. **Refactor Pipeline Orchestrator**: Update `bmf-vncomposer-pipeline`'s main controller:
   - Call the LLM with `pose_planner.jinja2` + structured JSON mode.
   - Run `apply_pose_to_g9()` in Blender.
   - Render the critique frame.
   - Pass image to Vision Model using `pose_critique.jinja2`.
   - If `pose_is_satisfactory` is false, run `apply_critique_delta()` and re-render (limit max retries to 3).
4. **Run Headless Verification**: Execute `tests/test_arm_cross.py` using Blender CLI to confirm absence of script crashes and valid transform updates.