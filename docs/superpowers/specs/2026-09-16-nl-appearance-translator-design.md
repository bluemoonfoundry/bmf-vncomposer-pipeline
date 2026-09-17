# NL appearance translator design

## Context

`scarecrow_pipeline/schemas.py` already defines the render-time contract for
pose and expression (`PosePayload`, `FACSExpression`), and `blender/worker.py`
already applies that contract directly to an onboarded character's rig
(`apply_pose`/`apply_expression`) -- posing was deliberately moved out of DAZ
Studio and into Blender specifically so it could be driven by natural
language later (see `scarecrow_pipeline/schemas.py`'s `PosePayload` docstring
and `blender/worker.py`'s `apply_pose` docstring). `scarecrow_pipeline/scene_tags.py`
extracts `# scarecrow: <description>` comments from Ren'Py scripts and
explicitly calls translating that description into a `RenderRequest` "a
separate, out-of-scope LLM step."

This spec covers that step: a standalone module that takes a character name
and a natural-language description and returns a validated
`(PosePayload | None, FACSExpression)` pair, ready to drop into a
`RenderRequest`/`CharacterPlacement`. Wiring it into `scene_tags.py`'s
extracted entries is out of scope for this spec -- a follow-up issue.

The character-to-Blender onboarding workflow this builds on (DAZ Studio GUI +
DazScriptServer export -> Blender import -> registry) is already implemented
and unchanged by this work; see `docs/outfit-onboarding-workflow.md` and
`scripts/onboard_character.py`.

## Non-goals (this spec)

- Wiring into `scene_tags.py` / Ren'Py scene tags (follow-up).
- Incremental/conversational pose adjustment ("raise it a bit more" relative
  to a current pose). This translator is stateless: one description in, one
  complete pose+expression out, from scratch every time.
- Per-character vocabulary. `docs/posable_bones.json` and
  `docs/facs_controls.json` are treated as a single shared vocabulary across
  all onboarded characters (same Genesis-based rig/FACS set). A character
  with a genuinely different rig is a future problem.
- A second LLM provider implementation. The interface must not hard-code
  Anthropic, but only an Anthropic client ships now.

## Module: `scarecrow_pipeline/nl_appearance.py`

### Public entry point

```python
def translate_appearance(
    character: str,
    description: str,
    client: LLMClient,
    *,
    vocab: AppearanceVocabulary | None = None,  # defaults to loading the checked-in docs
) -> AppearanceResult:
    ...
```

`AppearanceResult` is a small `BaseModel` wrapping `pose: PosePayload | None`
and `expression: FACSExpression` -- mirrors the two fields `CharacterPlacement`
and `RenderRequest` already carry side by side, so callers can assign both
straight onto either.

One combined call per description, not two: a description like "furious,
fists clenched" implies pose and expression together, and splitting it into
two independent LLM calls would need the pose call to somehow know about
tone and vice versa for no benefit.

### `LLMClient` protocol (provider-agnostic)

```python
class LLMClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> dict:
        """Return a dict parsed from the model's structured-output response,
        constrained to json_schema. Raises on a provider/network failure;
        does not itself validate the dict against the schema."""
```

One narrow method. `nl_appearance.py` only imports this protocol, never a
concrete provider SDK -- swapping or adding a provider means writing a new
class that satisfies it, not touching the translator or its tests.

### `AnthropicClient`

Lives in `scarecrow_pipeline/nl_appearance.py` (or a sibling
`scarecrow_pipeline/llm_clients.py` if the file grows) behind an optional
dependency, e.g.:

```toml
[project.optional-dependencies]
llm = ["anthropic>=0.40"]
```

mirroring the existing `[vision]` optional-dependency pattern in
`pyproject.toml`. It calls the Messages API with structured output (tool-use
or the equivalent constrained-output mechanism) using the JSON Schema it's
given, and returns the parsed dict. API key comes from the standard
`ANTHROPIC_API_KEY` environment variable -- no new config file.

### JSON Schema construction

Reuse the existing pattern from `scripts/export_json_schema.py`
(`Model.model_json_schema()`), applied to `AppearanceResult` rather than
`RenderRequest`, so the schema handed to the LLM can never hand-drift from
the Pydantic contract. No new schema-generation code path.

### Vocabulary loading

```python
class AppearanceVocabulary(BaseModel):
    bones: list[dict]        # docs/posable_bones.json "bones" entries, as-is
    facs_controls: list[dict]  # docs/facs_controls.json "controls" entries, as-is

def load_default_vocabulary() -> AppearanceVocabulary:
    """Reads docs/posable_bones.json and docs/facs_controls.json."""
```

Both files are already lean (name/category/rotation_mode/limits for bones;
name/category/drives_shape_key_directly for FACS controls) -- passed to the
LLM verbatim in the prompt rather than summarized or filtered, so the model
always sees the real, current name list.

### Prompt construction

System prompt: explains the task (translate a natural-language appearance
description into bone rotations in radians and FACS control weights in
[0,1]), states the axis-order caveat from `docs/posable_bones.json`'s note
(don't assume XYZ order), and states that `look_at_target`/`ik_targets` are
optional and should be omitted unless the description calls for them.

User prompt: the character name, the raw description, and both vocabulary
lists serialized as JSON.

### Validation and one bounded retry

JSON Schema constrains shape; it cannot constrain *which* bone/control names
exist, since that's a dynamic vocabulary. After `complete_json` returns:

1. `AppearanceResult.model_validate(raw)` -- catches shape errors (wrong
   vector length, out-of-range weight, unknown field, since all the models
   use `extra="forbid"`).
2. Cross-check every key in `pose.bone_rotations` / `pose.ik_targets` against
   `vocab.bones` names, and every key in `expression.weights` against
   `vocab.facs_controls` names -- catches hallucinated names Pydantic can't
   see.
3. On either failure, build one retry: re-send the original system/user
   prompt plus an assistant turn (the invalid response) and a new user turn
   listing the specific error(s) (e.g. `"bone_name 'l_uparm' is not in the
   vocabulary"`). Re-validate the second response the same way.
4. If the second attempt still fails, raise `AppearanceTranslationError`
   with the validation errors attached. No further retries -- this mirrors
   the project's existing "no unbounded retry, no silent infinite loop"
   posture from the DAZ Studio headless-export decision
   (`docs/outfit-onboarding-workflow.md`).

`blender/worker.py`'s existing silent-skip behavior for unknown bone/control
names (`apply_pose`/`apply_expression`'s `WARNING:` prints) is unchanged and
stays as the last-mile safety net for anything that still slips through --
this translator's job is to make that net rarely needed, not to replace it.

### Testing

`tests/test_nl_appearance.py`, following the existing fake-callable pattern
(`_fake_run_factory` / `_fake_export_fn` in `tests/test_onboard_character.py`):

- `FakeLLMClient` takes a queue of canned dicts (or dict-or-exception) and
  returns them in order from `complete_json`, recording calls made (so a
  test can assert the retry prompt actually contained the error message).
- Happy path: one valid response -> correct `AppearanceResult`.
- Retry path: one invalid response (bad bone name) followed by one valid
  response -> succeeds, and the second call's user_prompt mentions the bad
  name.
- Exhausted-retry path: two invalid responses in a row ->
  `AppearanceTranslationError`.
- Vocabulary cross-check: a response with a schema-valid but
  vocabulary-invalid bone name is rejected even though Pydantic alone would
  accept it.
- No test instantiates `AnthropicClient` or makes a network call; it's
  exercised only by a manual/opt-in smoke test, not the suite that runs by
  default.

## Open follow-up (not this spec)

- Wiring `scene_tags.py`'s extracted `SceneTagEntry` list through
  `translate_appearance` into full `RenderRequest`s (needs deciding
  character/outfit/camera/lighting defaults, which are outside this spec).
- Per-character vocabulary lookup, if/when a character with a different rig
  is onboarded.
