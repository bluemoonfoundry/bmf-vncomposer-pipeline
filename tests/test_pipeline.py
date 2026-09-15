import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from scarecrow_pipeline.registry import CharacterRecord, Registry
from scarecrow_pipeline.schemas import RenderRequest
from scarecrow_pipeline.sprite_set import SpriteCombo, SpriteSetSpec
from scarecrow_pipeline.vision import estimate_lighting

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "render_request.schema.json"


def test_render_request_rejects_out_of_range_weights():
    request = {
        "character": "Alice",
        "master_blend": "alice.blend",
        "output_path": "out.png",
        "expression": {"weights": {"jawOpen": 0.4}},
    }
    assert RenderRequest.model_validate(request).expression.weights["jawOpen"] == 0.4


def test_lighting_estimator_returns_normalized_metadata():
    image = BytesIO()
    Image.new("RGB", (12, 12), (220, 180, 160)).save(image, format="PNG")
    image.seek(0)
    lighting = estimate_lighting(image)
    assert len(lighting.direction) == 3
    assert all(0 <= channel <= 1 for channel in lighting.color_rgb)


def test_json_schema_file_matches_current_model():
    """Catches drift like the old base_pose field that outlived its removal
    from PosePayload -- regenerate with scripts/export_json_schema.py."""
    checked_in = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    current = RenderRequest.model_json_schema()
    assert checked_in == current, (
        "schemas/render_request.schema.json is stale; "
        "run `python scripts/export_json_schema.py` to regenerate it."
    )


def test_registry_round_trips_through_disk(tmp_path):
    path = tmp_path / "registry.json"
    registry = Registry.load(path)
    assert registry.characters == {}

    registry.upsert_character(CharacterRecord(
        name="JasonCross",
        master_blend="artifacts/JasonCross_imported.blend",
        collection="Jasoncross_v1",
    ))
    registry.save(path)

    reloaded = Registry.load(path)
    assert reloaded.characters["JasonCross"].collection == "Jasoncross_v1"


def _registry_with_jason(outfits=None, hair=None):
    registry = Registry()
    registry.upsert_character(CharacterRecord(
        name="JasonCross",
        master_blend="artifacts/JasonCross_imported.blend",
        collection="Jasoncross_v1",
        outfits=outfits or [],
        hair=hair or [],
    ))
    return registry


def test_sprite_set_expand_count_and_validation_succeeds(monkeypatch):
    monkeypatch.setattr(
        "scarecrow_pipeline.sprite_set.Registry.load",
        lambda *a, **k: _registry_with_jason(outfits=["formal"]),
    )
    spec = SpriteSetSpec(
        character="JasonCross",
        output_dir="out/sprites",
        combos=[
            SpriteCombo(outfit="formal", emotion_label="happy"),
            SpriteCombo(outfit="formal", emotion_label="sad"),
            SpriteCombo(emotion_label="happy"),
        ],
    )
    pairs = spec.expand()
    assert len(pairs) == 3
    for filename, request in pairs:
        assert RenderRequest.model_validate(request.model_dump()) == request
        assert request.output_path.endswith(filename)


def test_sprite_set_filenames_unique_for_emotion_and_outfit_variants(monkeypatch):
    monkeypatch.setattr(
        "scarecrow_pipeline.sprite_set.Registry.load",
        lambda *a, **k: _registry_with_jason(outfits=["formal", "casual"]),
    )
    spec = SpriteSetSpec(
        character="JasonCross",
        output_dir="out",
        combos=[
            SpriteCombo(outfit="formal", emotion_label="happy"),
            SpriteCombo(outfit="formal", emotion_label="sad"),
            SpriteCombo(outfit="casual", emotion_label="happy"),
        ],
    )
    filenames = [filename for filename, _ in spec.expand()]
    assert len(filenames) == len(set(filenames))


def test_sprite_set_rejects_unregistered_outfit(monkeypatch):
    monkeypatch.setattr(
        "scarecrow_pipeline.sprite_set.Registry.load",
        lambda *a, **k: _registry_with_jason(outfits=["formal"]),
    )
    with pytest.raises(Exception):
        SpriteSetSpec(
            character="JasonCross",
            output_dir="out",
            combos=[SpriteCombo(outfit="nonexistent")],
        )
