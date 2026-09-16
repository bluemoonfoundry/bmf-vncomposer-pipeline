import json
import sys

from scripts.extract_scene_tags import main


SAMPLE_RPY = '''label start:

    # scarecrow: eileen smiling warmly in the garden,
    # scarecrow: sunlight on her face
    image eileen happy garden = "eileen_happy_garden.png"

    # scarecrow: wide shot of the moonlit garden, empty
    scene bg garden night

    "Some dialogue line with no tag above it."
    image untagged sprite = "untagged.png"
'''


def test_cli_writes_manifest_matching_sample_rpy_end_to_end(tmp_path, monkeypatch, capsys):
    project_root = tmp_path / "project"
    project_root.mkdir()
    rpy_path = project_root / "script.rpy"
    rpy_path.write_text(SAMPLE_RPY, encoding="utf-8")

    out_path = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys, "argv", ["extract_scene_tags.py", str(project_root), "--out", str(out_path)]
    )

    main()

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert manifest == [
        {
            "renpy_tag": "eileen happy garden",
            "description": "eileen smiling warmly in the garden, sunlight on her face",
            "source_file": "script.rpy",
            "line": 5,
        },
        {
            "renpy_tag": "bg garden night",
            "description": "wide shot of the moonlit garden, empty",
            "source_file": "script.rpy",
            "line": 8,
        },
    ]

    captured = capsys.readouterr()
    assert captured.out == ""


def test_cli_prints_manifest_to_stdout_when_no_out_given(tmp_path, monkeypatch, capsys):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "script.rpy").write_text(SAMPLE_RPY, encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["extract_scene_tags.py", str(project_root)])

    main()

    captured = capsys.readouterr()
    manifest = json.loads(captured.out)
    assert [entry["renpy_tag"] for entry in manifest] == [
        "eileen happy garden",
        "bg garden night",
    ]
