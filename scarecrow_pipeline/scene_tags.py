"""Scan Ren'Py .rpy source for the '# scarecrow: <description>' tag convention.

A comment block of one or more consecutive '# scarecrow: ...' lines,
immediately followed (no blank line) by an 'image <tag> = ...' or
'scene <tag>' statement, describes the desired render for that tag. This
module only extracts {tag, description, source_file, line} manifest
entries; translating the description into a RenderRequest/SceneRequest is
a separate, out-of-scope LLM step.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

_COMMENT_RE = re.compile(r"^#\s*scarecrow:\s*(.*)$")
_IMAGE_RE = re.compile(r"^image\s+(.+?)\s*=.*$")
_SCENE_RE = re.compile(r"^scene\s+(.+?)\s*$")


class SceneTagEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    renpy_tag: str
    description: str
    source_file: str
    line: int


def extract_scene_tags(text: str, source_file: str) -> list[SceneTagEntry]:
    entries: list[SceneTagEntry] = []
    pending_description: list[str] = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()

        comment_match = _COMMENT_RE.match(stripped)
        if comment_match:
            pending_description.append(comment_match.group(1).strip())
            continue

        if pending_description:
            image_match = _IMAGE_RE.match(stripped)
            scene_match = _SCENE_RE.match(stripped) if not image_match else None
            tag_match = image_match or scene_match
            if tag_match:
                entries.append(
                    SceneTagEntry(
                        renpy_tag=tag_match.group(1).strip(),
                        description=" ".join(pending_description),
                        source_file=source_file,
                        line=lineno,
                    )
                )
            pending_description = []

    return entries
