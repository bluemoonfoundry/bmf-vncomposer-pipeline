"""Character/outfit/environment registry: what's been onboarded and where it lives.

Formalizes what artifacts/JasonCross_imported.blend currently is only
implicitly -- a reusable master cache. Downstream tools (sprite-set batch
generation, outfit variant authoring, multi-character scene assembly) use
this to validate a request references a character/outfit/environment that
actually exists before spending render time on it, the same way
docs/posable_bones.json and docs/facs_controls.json let a caller validate a
bone/control name before driving it.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "registry.json"


class CharacterRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    master_blend: str = Field(min_length=1)
    collection: str = Field(min_length=1, description="Collection name in master_blend passed as RenderRequest.character")
    outfits: list[str] = Field(default_factory=list, description="Available 'Outfit_<name>' collection names")
    hair: list[str] = Field(default_factory=list, description="Available 'Hair_<name>' collection names")


class EnvironmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_type: str = Field(min_length=1, description="e.g. 'daz', 'fbx', 'gltf', 'usd'")
    blend_path: str = Field(min_length=1, description="Normalized .blend containing the Environment_<name> collection")
    collection: str = Field(min_length=1)


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    characters: dict[str, CharacterRecord] = Field(default_factory=dict)
    environments: dict[str, EnvironmentRecord] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_REGISTRY_PATH) -> "Registry":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path | str = DEFAULT_REGISTRY_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2) + "\n", encoding="utf-8")

    def upsert_character(self, record: CharacterRecord) -> None:
        self.characters[record.name] = record

    def upsert_environment(self, record: EnvironmentRecord) -> None:
        self.environments[record.name] = record
