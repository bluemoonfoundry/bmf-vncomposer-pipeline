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

import json
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


def _describe_previous_targets(result: AppearanceResult) -> str:
    """Render the previous attempt's numeric ik_targets/pole_targets as
    corrective context for the next translate_appearance() call.

    Without this, a critique retry only sees prose (e.g. "arms are not
    crossed") and re-derives ik_targets/pole_targets from scratch with no
    memory of where it placed them last time -- so each attempt is an
    independent guess rather than a correction, and there's no guarantee
    attempt N+1 converges any closer than attempt N. Handing back the
    actual [x, y, z] world-space points lets the LLM nudge specific
    numbers instead of re-guessing blind.
    """
    if result.pose is None:
        return ""
    parts = []
    if result.pose.ik_targets:
        parts.append(f"ik_targets: {json.dumps(result.pose.ik_targets)}")
    if result.pose.pole_targets:
        parts.append(f"pole_targets: {json.dumps(result.pose.pole_targets)}")
    if not parts:
        return ""
    return (
        "\nThe previous attempt's pose used these numeric world-space targets "
        "(end-effector/pole bone name -> [x, y, z]):\n"
        + "\n".join(parts)
        + "\nAdjust these specific numeric values to address the critique above "
        "-- nudge them toward the correct position rather than discarding them "
        "and guessing brand-new coordinates from scratch, unless the critique "
        "implies a fundamentally different approach is needed."
    )


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
            + _describe_previous_targets(result)
        )

    return {
        "status": "ok",
        "attempts": max_attempts,
        "matches_description": False,
        "critique": last_critique.critique if last_critique else None,
        **(last_outcome.entry if last_outcome else {}),
    }
