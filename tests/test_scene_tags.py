import pytest

from scarecrow_pipeline.scene_tags import extract_scene_tags


def test_extracts_single_line_tag_immediately_before_image_statement():
    text = (
        '# scarecrow: eileen smiling warmly in the garden\n'
        'image eileen happy garden = "eileen_happy_garden.png"\n'
    )
    entries = extract_scene_tags(text, source_file="script.rpy")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.renpy_tag == "eileen happy garden"
    assert entry.description == "eileen smiling warmly in the garden"
    assert entry.source_file == "script.rpy"
    assert entry.line == 2


def test_concatenates_multi_line_description_block():
    text = (
        '# scarecrow: eileen standing at the door,\n'
        '# scarecrow: hand raised as if knocking\n'
        'image eileen knocking = "eileen_knocking.png"\n'
    )
    entries = extract_scene_tags(text, source_file="script.rpy")
    assert len(entries) == 1
    assert entries[0].description == "eileen standing at the door, hand raised as if knocking"
    assert entries[0].line == 3


def test_extracts_tag_from_scene_statement():
    text = (
        '# scarecrow: wide shot of the moonlit garden, empty\n'
        'scene bg garden night\n'
    )
    entries = extract_scene_tags(text, source_file="script.rpy")
    assert len(entries) == 1
    assert entries[0].renpy_tag == "bg garden night"
    assert entries[0].description == "wide shot of the moonlit garden, empty"
    assert entries[0].line == 2


def test_multiple_tags_have_correct_independent_line_numbers():
    text = (
        'label start:\n'
        '\n'
        '    # scarecrow: eileen happy in the garden\n'
        '    image eileen happy garden = "a.png"\n'
        '\n'
        '    # scarecrow: eileen sad in the kitchen\n'
        '    image eileen sad kitchen = "b.png"\n'
    )
    entries = extract_scene_tags(text, source_file="script.rpy")
    assert [(e.renpy_tag, e.line) for e in entries] == [
        ("eileen happy garden", 4),
        ("eileen sad kitchen", 7),
    ]


def test_ignores_comment_block_not_immediately_followed_by_image_or_scene():
    text = (
        '# scarecrow: this description is orphaned\n'
        '\n'
        'image eileen happy = "a.png"\n'
    )
    entries = extract_scene_tags(text, source_file="script.rpy")
    assert entries == []
