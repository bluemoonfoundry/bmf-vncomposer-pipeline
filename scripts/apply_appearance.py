"""CLI: apply a natural-language appearance description to an onboarded
character in Blender.

Chains scarecrow_pipeline/nl_appearance.py's translate_appearance() with the
existing single-character render path (blender/worker.py), the same way
scripts/generate_sprite_set.py already invokes it: one Blender subprocess,
built from a RenderRequest. translate_appearance()'s only new contribution is
filling in RenderRequest.pose/.expression from natural-language text instead
of hand-written values -- everything else (registry lookup, subprocess
invocation, --python-exit-code) follows generate_sprite_set.py's pattern.

Usage:
  python scripts/apply_appearance.py --character JasonCross `
    --description "standing at ease, arms crossed, jaw slightly open" `
    --output artifacts/JasonCross_appearance_test.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scarecrow_pipeline.nl_appearance import (
    AppearanceResult,
    AppearanceVocabulary,
    LLMClient,
    VisionCritiqueClient,
    translate_appearance,
)
from scarecrow_pipeline.pose_critique import (
    DEFAULT_MAX_RENDER_ATTEMPTS,
    RenderFailedError,
    RenderOutcome,
    run_with_critique,
)
from scarecrow_pipeline.registry import Registry
from scarecrow_pipeline.schemas import RenderRequest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLCHAIN_PATH = REPO_ROOT / "config" / "toolchain.json"
WORKER_SCRIPT = REPO_ROOT / "blender" / "worker.py"


def resolve_blender_exe(explicit: str | None, toolchain_path: Path = DEFAULT_TOOLCHAIN_PATH) -> str:
    if explicit:
        return explicit
    toolchain = json.loads(toolchain_path.read_text(encoding="utf-8"))
    return toolchain["blender"]["executable"]


def _resolve_absolute(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _render_once(
    character: str,
    description: str,
    result: AppearanceResult,
    output_path_abs: Path,
    record,
    blender_exe: str,
    render_samples: int,
    run,
) -> dict:
    """Build a RenderRequest from one translate_appearance() result and run
    the single Blender subprocess that applies and renders it. Returns a
    manifest-style dict {status, output_path, request_path, stderr?} --
    never raises on a Blender-side failure, mirroring
    generate_sprite_set.py's render_combo()."""
    # RenderRequest.character is the master_blend collection name to append
    # (see CharacterRecord.collection's docstring), not the registry's
    # human-facing key -- same convention scarecrow_pipeline/sprite_set.py
    # already follows.
    request = RenderRequest(
        character=record.collection,
        master_blend=str(_resolve_absolute(record.master_blend)),
        output_path=str(output_path_abs),
        pose=result.pose,
        expression=result.expression,
        render_samples=render_samples,
    )

    requests_dir = output_path_abs.parent / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    request_json_path = (requests_dir / f"{output_path_abs.stem}.request.json").resolve()
    request_json_path.write_text(json.dumps(request.model_dump(), indent=2), encoding="utf-8")

    command = [
        blender_exe,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(WORKER_SCRIPT.resolve()),
        "--",
        "--request",
        str(request_json_path),
    ]

    entry = {
        "character": character,
        "description": description,
        "output_path": str(output_path_abs),
        "request_path": str(request_json_path),
    }
    try:
        completed = run(command, capture_output=True, text=True)
    except OSError as exc:
        entry["status"] = "failed"
        entry["stderr"] = str(exc)
        return entry

    if completed.returncode == 0:
        entry["status"] = "ok"
    else:
        entry["status"] = "failed"
        entry["stderr"] = (completed.stderr or "")[-4000:]
    return entry


def apply_appearance(
    character: str,
    description: str,
    output_path: str,
    *,
    client: LLMClient,
    blender_exe: str,
    registry_path: Path | str | None = None,
    vocab: AppearanceVocabulary | None = None,
    render_samples: int = 64,
    run=subprocess.run,
    vision_client: VisionCritiqueClient | None = None,
    max_render_attempts: int = DEFAULT_MAX_RENDER_ATTEMPTS,
) -> dict:
    """Translate description into pose/expression and apply it to `character`
    in its onboarded master_blend via one Blender subprocess.

    When vision_client is None (the default), behaves exactly as before:
    one translate_appearance() call, one render, returns
    {status, output_path, request_path, stderr?}. When vision_client is
    given, renders are critiqued against `description` and retried (with the
    critique fed back into the next translate_appearance() call) up to
    max_render_attempts -- see scarecrow_pipeline/pose_critique.py. Raises
    ValueError up front if `character` isn't in the registry, since there is
    no master_blend to apply the appearance to.
    """
    registry = Registry.load(registry_path) if registry_path else Registry.load()
    record = registry.characters.get(character)
    if record is None:
        raise ValueError(
            f"character {character!r} is not in the registry -- onboard it "
            "first (see scripts/onboard_character.py)"
        )

    output_path_abs = _resolve_absolute(output_path)

    if vision_client is None:
        result = translate_appearance(character, description, client, vocab=vocab)
        return _render_once(
            character, description, result, output_path_abs, record, blender_exe, render_samples, run
        )

    def render_fn(result: AppearanceResult, attempt: int) -> RenderOutcome:
        entry = _render_once(
            character, description, result, output_path_abs, record, blender_exe, render_samples, run
        )
        if entry["status"] != "ok":
            raise RenderFailedError(entry.get("stderr", "render failed"))
        return RenderOutcome(entry=entry, image_path=output_path_abs)

    return run_with_critique(
        character,
        description,
        translate_client=client,
        vision_client=vision_client,
        render_fn=render_fn,
        vocab=vocab,
        max_attempts=max_render_attempts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blender-exe")
    parser.add_argument("--registry-path")
    parser.add_argument("--render-samples", type=int, default=64)
    parser.add_argument(
        "--with-critique",
        action="store_true",
        help="Render, critique the result against --description with a vision-capable "
             "model, and retry (up to --max-render-attempts) if it doesn't match.",
    )
    parser.add_argument("--max-render-attempts", type=int, default=DEFAULT_MAX_RENDER_ATTEMPTS)
    args = parser.parse_args(argv)

    from scarecrow_pipeline.nl_appearance import AnthropicClient

    client = AnthropicClient()
    blender_exe = resolve_blender_exe(args.blender_exe)

    entry = apply_appearance(
        args.character,
        args.description,
        args.output,
        client=client,
        blender_exe=blender_exe,
        registry_path=args.registry_path,
        render_samples=args.render_samples,
        vision_client=client if args.with_critique else None,
        max_render_attempts=args.max_render_attempts,
    )

    if entry["status"] != "ok":
        print(f"FAILED: {entry['output_path']}: {entry.get('stderr', '')[:500]}", file=sys.stderr)
        return 1
    if entry.get("matches_description") is False:
        print(
            f"WARNING: {entry['output_path']} did not pass critique after "
            f"{entry.get('attempts')} attempt(s): {entry.get('critique')}",
            file=sys.stderr,
        )
    print(f"OK: {entry['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
