"""Render -> vision critique -> retry loop for translate_appearance's output.

apply_appearance.py's Blender subprocess call is a real, expensive render,
and translate_appearance()'s first guess at a pose has no way to know
whether it actually looks like its description -- see bd issue
scarecrow-57h. This module closes that loop: translate, render (via an
injected callback so this module never touches Blender/subprocess directly),
critique the render against the description with a vision-capable client,
and retry with the critique folded back into the next translate_appearance()
call on a mismatch, up to a bounded number of attempts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scarecrow_pipeline.nl_appearance import (
    AppearanceResult,
    AppearanceVocabulary,
    LLMClient,
    PoseCritique,
    VisionCritiqueClient,
    translate_appearance,
)

DEFAULT_MAX_RENDER_ATTEMPTS = 2


class RenderFailedError(RuntimeError):
    """Raised by a render_fn callback to signal a Blender-side render
    failure, distinct from a critique mismatch (which is not an error)."""


@dataclass
class RenderOutcome:
    """One render attempt's result, as returned by a render_fn callback.

    entry is the caller-defined manifest dict for this attempt (e.g.
    apply_appearance.py's {character, description, output_path,
    request_path, status}); image_path is the rendered PNG to critique.
    """

    entry: dict
    image_path: Path


RenderFn = Callable[[AppearanceResult, int], RenderOutcome]


def run_with_critique(
    character: str,
    description: str,
    *,
    translate_client: LLMClient,
    vision_client: VisionCritiqueClient,
    render_fn: RenderFn,
    vocab: AppearanceVocabulary | None = None,
    max_attempts: int = DEFAULT_MAX_RENDER_ATTEMPTS,
) -> dict:
    """Translate + render + critique, retrying with corrective context on a
    mismatch, up to max_attempts.

    Never raises on a critique mismatch or on a render failure signaled via
    RenderFailedError -- returns a manifest dict describing the best-effort
    last attempt in either case, so a human or CI can still inspect the
    artifact. Only propagates an unexpected exception from render_fn or the
    clients themselves (a real provider/subprocess failure, not a "the pose
    didn't match" outcome).
    """
    extra_context: str | None = None
    last_outcome: RenderOutcome | None = None
    last_critique: PoseCritique | None = None

    for attempt in range(1, max_attempts + 1):
        result = translate_appearance(
            character, description, translate_client, vocab=vocab, extra_context=extra_context
        )
        try:
            outcome = render_fn(result, attempt)
        except RenderFailedError as exc:
            return {"status": "render_failed", "attempts": attempt, "error": str(exc)}
        last_outcome = outcome

        image_bytes = outcome.image_path.read_bytes()
        critique = vision_client.critique_pose(description, image_bytes)
        last_critique = critique

        if critique.matches_description:
            return {
                "status": "ok",
                "attempts": attempt,
                "matches_description": True,
                "critique": critique.critique,
                **outcome.entry,
            }

        extra_context = (
            "A previous attempt at this description was rendered and critiqued as NOT "
            f"matching it: {critique.critique}\nAdjust the pose to address this critique."
        )

    return {
        "status": "ok",
        "attempts": max_attempts,
        "matches_description": False,
        "critique": last_critique.critique if last_critique else None,
        **(last_outcome.entry if last_outcome else {}),
    }
