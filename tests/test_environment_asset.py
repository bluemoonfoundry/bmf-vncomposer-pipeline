import pytest

from scarecrow_pipeline.environment_asset import compute_bounding_box, resolve_handler


@pytest.mark.parametrize("source_path,expected", [
    ("set.dbz", "daz"),
    ("prop.duf", "daz"),
    ("scene.FBX", "fbx"),
    ("scene.gltf", "gltf"),
    ("scene.glb", "gltf"),
    ("mesh.obj", "obj"),
    ("stage.usd", "usd"),
    ("stage.usda", "usd"),
    ("stage.usdc", "usd"),
    ("stage.usdz", "usd"),
])
def test_resolve_handler_maps_known_extensions(source_path, expected):
    assert resolve_handler(source_path) == expected


def test_resolve_handler_rejects_unknown_extension():
    with pytest.raises(ValueError, match="Unsupported environment asset extension"):
        resolve_handler("model.blend")


def test_compute_bounding_box_over_multiple_points():
    corners = [(-1.0, 0.0, 2.0), (3.0, -5.0, 2.0), (0.0, 4.0, -1.0)]
    bbox_min, bbox_max = compute_bounding_box(corners)
    assert bbox_min == [-1.0, -5.0, -1.0]
    assert bbox_max == [3.0, 4.0, 2.0]


def test_compute_bounding_box_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one point"):
        compute_bounding_box([])
