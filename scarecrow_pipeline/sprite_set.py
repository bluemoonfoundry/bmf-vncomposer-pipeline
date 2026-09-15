"""Sprite-set batch generation: expand a set of character variants into RenderRequests.

Phase 1 of the sprite-set/CG-compositing plan. Pure orchestration over the
existing single-character render path -- reuses RenderRequest/PosePayload/
FACSExpression from scarecrow_pipeline.schemas UNCHANGED.

Combo bone names and FACS control names (for PosePayload.bone_rotations and
FACSExpression.weights) must come from the controlled vocabularies in
docs/posable_bones.json and docs/facs_controls.json -- this module does not
hardcode or validate their contents, it only forwards whatever the caller
supplies straight into PosePayload/FACSExpression.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from scarecrow_pipeline.registry import Registry
from scarecrow_pipeline.schemas import (
    CameraMetadata,
    FACSExpression,
    PosePayload,
    RenderRequest,
    SceneLighting,
)

DEFAULT_NAMING_TEMPLATE = "{character}_{outfit}_{emotion}_{pose}.png"


class SpriteCombo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outfit: str | None = None
    hair: str | None = None
    emotion_label: str | None = None
    expression: FACSExpression | None = None
    pose: PosePayload | None = None


class SpriteSetSpec(BaseModel):
    """A batch of sprite renders for one character across outfit/hair/pose/expression variants.

    If master_blend is omitted, it (and the character's collection name) is
    resolved via scarecrow_pipeline.registry.Registry.load() by character
    name -- the registry is the preferred source of truth over requiring
    both explicitly.
    """

    model_config = ConfigDict(extra="forbid")
    character: str = Field(min_length=1)
    master_blend: str | None = None
    base_camera: CameraMetadata = Field(default_factory=CameraMetadata)
    base_lighting: SceneLighting = Field(default_factory=SceneLighting)
    combos: list[SpriteCombo] = Field(default_factory=list)
    output_dir: str = Field(min_length=1)
    naming_template: str = DEFAULT_NAMING_TEMPLATE
    transparent: bool = True

    _collection: str = PrivateAttr()

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> "SpriteSetSpec":
        registry = Registry.load()
        record = registry.characters.get(self.character)

        # RenderRequest.character is actually the master_blend collection name
        # to append (see CharacterRecord.collection's docstring) -- it need not
        # match the spec's human-facing character name (e.g. "JasonCross" vs.
        # the onboarded collection "Jasoncross_v1").
        self._collection = record.collection if record is not None else self.character

        if self.master_blend is None:
            if record is None:
                raise ValueError(
                    f"master_blend not given and character {self.character!r} "
                    "is not in the registry -- either pass master_blend explicitly "
                    "or onboard the character first"
                )
            self.master_blend = record.master_blend

        if record is not None and (record.outfits or record.hair):
            for combo in self.combos:
                if combo.outfit is not None and combo.outfit not in record.outfits:
                    raise ValueError(
                        f"outfit {combo.outfit!r} is not registered for character "
                        f"{self.character!r} (known: {record.outfits})"
                    )
                if combo.hair is not None and combo.hair not in record.hair:
                    raise ValueError(
                        f"hair {combo.hair!r} is not registered for character "
                        f"{self.character!r} (known: {record.hair})"
                    )

        return self

    def _filename_for(self, combo: SpriteCombo) -> str:
        return self.naming_template.format(
            character=self.character,
            outfit=combo.outfit or "base",
            hair=combo.hair or "base",
            emotion=combo.emotion_label or "neutral",
            pose="posed" if combo.pose is not None else "base",
        )

    def expand(self) -> list[tuple[str, RenderRequest]]:
        """Expand each combo into a (output_filename, RenderRequest) pair.

        Filenames are deduplicated: if the naming template produces the same
        name for two distinct combos (e.g. two combos differing only in the
        pose payload's contents, which the template cannot represent as
        text), a numeric suffix is appended to keep filenames unique.
        """
        results: list[tuple[str, RenderRequest]] = []
        seen: dict[str, int] = {}

        for combo in self.combos:
            filename = self._filename_for(combo)
            if filename in seen:
                seen[filename] += 1
                stem, ext = os.path.splitext(filename)
                filename = f"{stem}_{seen[filename]}{ext}"
            else:
                seen[filename] = 1

            request = RenderRequest(
                character=self._collection,
                master_blend=self.master_blend,
                output_path=os.path.join(self.output_dir, filename),
                outfit=combo.outfit,
                hair=combo.hair,
                emotion=combo.emotion_label,
                pose=combo.pose,
                expression=combo.expression or FACSExpression(),
                lighting=self.base_lighting,
                camera=self.base_camera,
                transparent=self.transparent,
            )
            results.append((filename, request))

        return results
