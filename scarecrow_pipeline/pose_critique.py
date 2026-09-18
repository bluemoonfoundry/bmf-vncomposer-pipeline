"""Render -> vision critique -> retry loop for translate_appearance's output.

apply_appearance.py's Blender subprocess call is a real, expensive render,
and translate_appearance()'s first guess at a pose has no way to know
whether it actually looks like its description -- see bd issue
scarecrow-57h. This module closes that loop: translate ONCE, then render
(via an injected callback so this module never touches Blender/subprocess
directly), critique the render against the description with a
vision-capable client, and on a mismatch apply the critique's numeric
per-limb deltas directly to the previous attempt's PoseIntent -- see
docs/gemini_ikplan_a.md Phase 4 -- rather than re-invoking the
pose-generation LLM. This makes convergence deterministic: attempt N+1's
ik_targets/pole_targets are attempt N's plus an explicit numeric
correction, not an independent fresh guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scarecrow_pipeline.nl_appearance import (
    AppearanceIntent,
    AppearanceResult,
    AppearanceVocabulary,
    LLMClient,
    VisionCritiqueClient,
    apply_critique_delta,
    load_default_vocabulary,
    resolve_pose_intent,
    translate_pose_intent,
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
    """Translate once, then render + critique + apply numeric deltas,
    retrying up to max_attempts.

    Never raises on a critique mismatch or on a render failure signaled via
    RenderFailedError -- returns a manifest dict describing the best-effort
    last attempt in either case, so a human or CI can still inspect the
    artifact. Only propagates an unexpected exception from render_fn or the
    clients themselves (a real provider/subprocess failure, not a "the pose
    didn't match" outcome).
    """
    vocab = vocab or load_default_vocabulary()
    intent: AppearanceIntent = translate_pose_intent(character, description, translate_client, vocab=vocab)

    last_outcome: RenderOutcome | None = None
    last_feedback = None

    for attempt in range(1, max_attempts + 1):
        result = AppearanceResult(pose=resolve_pose_intent(intent.pose, vocab), expression=intent.expression)
        try:
            outcome = render_fn(result, attempt)
        except RenderFailedError as exc:
            return {"status": "render_failed", "attempts": attempt, "error": str(exc)}
        last_outcome = outcome

        image_bytes = outcome.image_path.read_bytes()
        feedback = vision_client.critique_pose(description, image_bytes, intent.pose)
        last_feedback = feedback

        if feedback.pose_is_satisfactory:
            return {
                "status": "ok",
                "attempts": attempt,
                "matches_description": True,
                "critique": feedback.critique_summary,
                **outcome.entry,
            }

        intent = apply_critique_delta(intent, feedback)

    return {
        "status": "ok",
        "attempts": max_attempts,
        "matches_description": False,
        "critique": last_feedback.critique_summary if last_feedback else None,
        **(last_outcome.entry if last_outcome else {}),
    }
