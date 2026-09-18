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
    position = anchors.resolve_anchor_position("ANCHOR_PECTORAL_R", _bone_landmarks())

    assert position == [-0.0385, 0.0474, 1.3575]


def test_resolve_anchor_position_blends_bicep_lateral_toward_shoulder():
    landmarks = _bone_landmarks()
    head = landmarks["r_upperarm"]["rest_head_world"]
    tail = landmarks["r_upperarm"]["rest_tail_world"]

    position = anchors.resolve_anchor_position("ANCHOR_BICEP_LATERAL_R", landmarks)

    expected = [head[i] + (tail[i] - head[i]) * 0.35 for i in range(3)]
    assert position == pytest.approx(expected)
    # blended point sits much closer to the shoulder (head) than the old
    # elbow (tail) landmark did -- that's the whole fix for scarecrow-5mn.
    assert abs(position[0]) < abs(tail[0])


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
