# NL Appearance Translator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scarecrow_pipeline/nl_appearance.py`, a standalone module that translates a natural-language character-appearance description into a validated `(PosePayload | None, FACSExpression)` pair, via a provider-agnostic LLM client with one bounded retry on invalid output.

**Architecture:** A pure-Python translation layer sitting on top of the existing `PosePayload`/`FACSExpression` schemas (`scarecrow_pipeline/schemas.py`) and the existing generated vocabulary docs (`docs/posable_bones.json`, `docs/facs_controls.json`). A narrow `LLMClient` protocol decouples the translator from any specific provider SDK; `AnthropicClient` is the one concrete implementation, gated behind an optional dependency so the core module and its tests never require a network call or the `anthropic` package.

**Tech Stack:** Python 3.10+, Pydantic 2.x (matching `scarecrow_pipeline/schemas.py`'s `ConfigDict(extra="forbid")` convention), pytest, `anthropic` SDK as an optional extra.

**Spec:** `docs/superpowers/specs/2026-09-16-nl-appearance-translator-design.md`

## Global Constraints

- No unbounded retry: exactly `MAX_ATTEMPTS = 2` total LLM calls per `translate_appearance()` invocation, then raise.
- Every new Pydantic model uses `model_config = ConfigDict(extra="forbid")`, matching `scarecrow_pipeline/schemas.py`.
- `nl_appearance.py` never imports `anthropic` (or any provider SDK) at module scope — only inside `AnthropicClient.__init__`/`complete_json`, so the module and its default test suite never require the optional dependency.
- Vocabulary docs (`docs/posable_bones.json`, `docs/facs_controls.json`) are passed to the LLM verbatim (as parsed JSON), never summarized or filtered.
- No test instantiates `AnthropicClient` with a real API key or makes a network call.
- This plan does not touch `scarecrow_pipeline/scene_tags.py` or any Ren'Py wiring — explicitly out of scope per the spec.

---

## Task 1: Vocabulary loading

**Files:**
- Create: `scarecrow_pipeline/nl_appearance.py`
- Test: `tests/test_nl_appearance.py`

**Interfaces:**
- Consumes: `docs/posable_bones.json` (`{"bones": [{"name": ..., ...}, ...]}`), `docs/facs_controls.json` (`{"controls": [{"name": ..., ...}, ...]}`) — both already exist in the repo.
- Produces: `AppearanceVocabulary` (fields `bones: list[dict]`, `facs_controls: list[dict]`, properties `bone_names: set[str]`, `facs_control_names: set[str]`), `load_default_vocabulary(posable_bones_path=..., facs_controls_path=...) -> AppearanceVocabulary`, module constants `REPO_ROOT`, `DEFAULT_POSABLE_BONES_PATH`, `DEFAULT_FACS_CONTROLS_PATH`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nl_appearance.py`:

```python
from pathlib import Path

from scarecrow_pipeline.nl_appearance import (
    AppearanceVocabulary,
    load_default_vocabulary,
)


def test_load_default_vocabulary_reads_real_docs():
    vocab = load_default_vocabulary()

    assert "hip" in vocab.bone_names
    assert "l_upperarm" in vocab.bone_names
    assert any(name.startswith("facs_") for name in vocab.facs_control_names)


def test_load_default_vocabulary_accepts_explicit_paths(tmp_path):
    bones_path = tmp_path / "bones.json"
    bones_path.write_text('{"bones": [{"name": "hip", "category": "spine"}]}', encoding="utf-8")
    controls_path = tmp_path / "controls.json"
    controls_path.write_text('{"controls": [{"name": "facs_bs_JawOpenWide", "category": "jaw"}]}', encoding="utf-8")

    vocab = load_default_vocabulary(posable_bones_path=bones_path, facs_controls_path=controls_path)

    assert vocab.bone_names == {"hip"}
    assert vocab.facs_control_names == {"facs_bs_JawOpenWide"}


def test_appearance_vocabulary_rejects_unknown_field():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AppearanceVocabulary(bones=[], facs_controls=[], extra_field="nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nl_appearance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scarecrow_pipeline.nl_appearance'`

- [ ] **Step 3: Write the implementation**

Create `scarecrow_pipeline/nl_appearance.py`:

```python
"""Translate a natural-language appearance description into a validated
PosePayload/FACSExpression pair, via a provider-agnostic LLM client.

DAZ Studio is only ever used to source a character once (see
scripts/onboard_character.py); all posing and expression happen in
Blender (blender/worker.py's apply_pose/apply_expression) so they can be
driven by natural language. This module is the translation step
scarecrow_pipeline/scene_tags.py calls "a separate, out-of-scope LLM
step" -- see
docs/superpowers/specs/2026-09-16-nl-appearance-translator-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSABLE_BONES_PATH = REPO_ROOT / "docs" / "posable_bones.json"
DEFAULT_FACS_CONTROLS_PATH = REPO_ROOT / "docs" / "facs_controls.json"


class AppearanceVocabulary(BaseModel):
    """The bone/FACS-control name vocabulary the LLM is constrained to.

    Shared across all onboarded characters (same Genesis-based rig/FACS
    set) -- see docs/posable_bones.json and docs/facs_controls.json,
    regenerated per-character via blender/dump_posable_bones.py and
    blender/dump_facs_controls.py.
    """

    model_config = ConfigDict(extra="forbid")
    bones: list[dict]
    facs_controls: list[dict]

    @property
    def bone_names(self) -> set[str]:
        return {bone["name"] for bone in self.bones}

    @property
    def facs_control_names(self) -> set[str]:
        return {control["name"] for control in self.facs_controls}


def load_default_vocabulary(
    posable_bones_path: Path = DEFAULT_POSABLE_BONES_PATH,
    facs_controls_path: Path = DEFAULT_FACS_CONTROLS_PATH,
) -> AppearanceVocabulary:
    bones = json.loads(Path(posable_bones_path).read_text(encoding="utf-8"))["bones"]
    facs_controls = json.loads(Path(facs_controls_path).read_text(encoding="utf-8"))["controls"]
    return AppearanceVocabulary(bones=bones, facs_controls=facs_controls)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_appearance.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scarecrow_pipeline/nl_appearance.py tests/test_nl_appearance.py
git commit -m "Add AppearanceVocabulary loader for NL appearance translator"
```

---

## Task 2: Result schema, error type, and LLMClient protocol

**Files:**
- Modify: `scarecrow_pipeline/nl_appearance.py`
- Test: `tests/test_nl_appearance.py`

**Interfaces:**
- Consumes: `scarecrow_pipeline.schemas.PosePayload`, `scarecrow_pipeline.schemas.FACSExpression` (existing, unchanged).
- Produces: `AppearanceResult` (fields `pose: PosePayload | None = None`, `expression: FACSExpression` default-factory `FACSExpression`), `AppearanceTranslationError(RuntimeError)`, `LLMClient` protocol with `complete_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> dict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nl_appearance.py`:

```python
from scarecrow_pipeline.nl_appearance import AppearanceResult, AppearanceTranslationError
from scarecrow_pipeline.schemas import FACSExpression, PosePayload


def test_appearance_result_defaults_to_no_pose_and_empty_expression():
    result = AppearanceResult()

    assert result.pose is None
    assert result.expression == FACSExpression(weights={})


def test_appearance_result_accepts_pose_and_expression():
    result = AppearanceResult(
        pose=PosePayload(bone_rotations={"hip": [0.0, 0.0, 0.0]}),
        expression=FACSExpression(weights={"facs_bs_JawOpenWide": 0.5}),
    )

    assert result.pose.bone_rotations == {"hip": [0.0, 0.0, 0.0]}
    assert result.expression.weights == {"facs_bs_JawOpenWide": 0.5}


def test_appearance_translation_error_is_a_runtime_error():
    assert issubclass(AppearanceTranslationError, RuntimeError)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nl_appearance.py -v`
Expected: FAIL with `ImportError: cannot import name 'AppearanceResult'`

- [ ] **Step 3: Write the implementation**

Add to `scarecrow_pipeline/nl_appearance.py` (after the imports, before `AppearanceVocabulary`):

```python
from typing import Protocol

from pydantic import Field

from scarecrow_pipeline.schemas import FACSExpression, PosePayload

MAX_ATTEMPTS = 2


class AppearanceTranslationError(RuntimeError):
    """Raised when the LLM's output cannot be validated after MAX_ATTEMPTS tries."""


class AppearanceResult(BaseModel):
    """One description's translated appearance.

    Assign both fields straight onto a CharacterPlacement/RenderRequest
    (scarecrow_pipeline/schemas.py), which carry pose and expression as
    the same two side-by-side fields.
    """

    model_config = ConfigDict(extra="forbid")
    pose: PosePayload | None = None
    expression: FACSExpression = Field(default_factory=FACSExpression)


class LLMClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> dict:
        """Return a dict parsed from the model's structured-output response,
        constrained to json_schema. Raises on a provider/network failure;
        does not itself validate the dict against the schema."""
        ...
```

(Add the `from typing import Protocol` and `from pydantic import Field` to the top-level import block rather than inline; the snippet above shows placement for clarity.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_appearance.py -v`
Expected: PASS (6 tests total)

- [ ] **Step 5: Commit**

```bash
git add scarecrow_pipeline/nl_appearance.py tests/test_nl_appearance.py
git commit -m "Add AppearanceResult, AppearanceTranslationError, LLMClient protocol"
```

---

## Task 3: Prompt builders

**Files:**
- Modify: `scarecrow_pipeline/nl_appearance.py`
- Test: `tests/test_nl_appearance.py`

**Interfaces:**
- Consumes: `AppearanceVocabulary` (Task 1).
- Produces: `build_system_prompt() -> str`, `build_user_prompt(character: str, description: str, vocab: AppearanceVocabulary) -> str`, `build_retry_prompt(errors: list[str]) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nl_appearance.py`:

```python
from scarecrow_pipeline.nl_appearance import (
    build_retry_prompt,
    build_system_prompt,
    build_user_prompt,
)


def test_build_system_prompt_mentions_radians_and_vocabulary_only():
    prompt = build_system_prompt()

    assert "radians" in prompt.lower()
    assert "vocabulary" in prompt.lower()


def test_build_user_prompt_includes_character_description_and_vocab():
    vocab = AppearanceVocabulary(
        bones=[{"name": "hip", "category": "spine"}],
        facs_controls=[{"name": "facs_bs_JawOpenWide", "category": "jaw"}],
    )

    prompt = build_user_prompt("JasonCross", "standing at ease, arms crossed", vocab)

    assert "JasonCross" in prompt
    assert "standing at ease, arms crossed" in prompt
    assert "hip" in prompt
    assert "facs_bs_JawOpenWide" in prompt


def test_build_retry_prompt_lists_each_error():
    prompt = build_retry_prompt(["bone_name 'l_uparm' is not in the vocabulary"])

    assert "l_uparm" in prompt
    assert "not in the vocabulary" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nl_appearance.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_system_prompt'`

- [ ] **Step 3: Write the implementation**

Add to `scarecrow_pipeline/nl_appearance.py` (after `load_default_vocabulary`):

```python
def build_system_prompt() -> str:
    return (
        "You translate a natural-language character appearance description "
        "into precise pose and facial-expression control values for a "
        "Diffeomorphic-imported DAZ character rig in Blender.\n\n"
        "Respond with a single JSON object matching the given schema:\n"
        "- pose.bone_rotations: a mapping of bone name to an [x, y, z] Euler "
        "rotation IN RADIANS. Each bone has its OWN native rotation axis "
        "order (see each bone's rotation_mode in the provided vocabulary) "
        "-- do not assume XYZ order; provide the three angles as if applied "
        "in that bone's own order.\n"
        "- pose.ik_targets and pose.look_at_target are optional -- omit "
        "them entirely unless the description specifically calls for an IK "
        "target or a gaze direction.\n"
        "- expression.weights: a mapping of FACS/morph control name to a "
        "weight in [0.0, 1.0].\n"
        "- Only use bone and control names that appear in the provided "
        "vocabulary, exactly as spelled there. Never invent a name.\n"
        "- Omit any field you have no information for rather than guessing "
        "a value."
    )


def build_user_prompt(character: str, description: str, vocab: AppearanceVocabulary) -> str:
    return (
        f"Character: {character}\n"
        f"Description: {description}\n\n"
        f"Available pose bones (JSON):\n{json.dumps(vocab.bones)}\n\n"
        f"Available expression controls (JSON):\n{json.dumps(vocab.facs_controls)}"
    )


def build_retry_prompt(errors: list[str]) -> str:
    joined = "\n".join(f"- {error}" for error in errors)
    return (
        "Your previous response was invalid:\n"
        f"{joined}\n\n"
        "Respond again with a corrected JSON object matching the same "
        "schema, using only bone/control names from the vocabulary already "
        "provided."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_appearance.py -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Commit**

```bash
git add scarecrow_pipeline/nl_appearance.py tests/test_nl_appearance.py
git commit -m "Add system/user/retry prompt builders for NL appearance translator"
```

---

## Task 4: `translate_appearance` orchestration (validation + bounded retry)

**Files:**
- Modify: `scarecrow_pipeline/nl_appearance.py`
- Test: `tests/test_nl_appearance.py`

**Interfaces:**
- Consumes: `AppearanceVocabulary`/`load_default_vocabulary` (Task 1), `AppearanceResult`/`AppearanceTranslationError`/`LLMClient`/`MAX_ATTEMPTS` (Task 2), `build_system_prompt`/`build_user_prompt`/`build_retry_prompt` (Task 3).
- Produces: `translate_appearance(character: str, description: str, client: LLMClient, *, vocab: AppearanceVocabulary | None = None) -> AppearanceResult`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nl_appearance.py`:

```python
from scarecrow_pipeline.nl_appearance import translate_appearance


class FakeLLMClient:
    """Returns queued dict responses (or raises a queued exception) in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, json_schema):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "json_schema": json_schema,
        })
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _small_vocab():
    return AppearanceVocabulary(
        bones=[{"name": "hip", "category": "spine", "rotation_mode": "XYZ"}],
        facs_controls=[{"name": "facs_bs_JawOpenWide", "category": "jaw"}],
    )


def test_translate_appearance_happy_path():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {
            "pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}},
            "expression": {"weights": {"facs_bs_JawOpenWide": 0.3}},
        }
    ])

    result = translate_appearance("JasonCross", "leaning forward slightly, mouth open", client, vocab=vocab)

    assert result.pose.bone_rotations == {"hip": [0.1, 0.0, 0.0]}
    assert result.expression.weights == {"facs_bs_JawOpenWide": 0.3}
    assert len(client.calls) == 1


def test_translate_appearance_retries_once_on_invalid_bone_name_then_succeeds():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {
            "pose": {"bone_rotations": {"l_uparm": [0.1, 0.0, 0.0]}},
            "expression": {"weights": {}},
        },
        {
            "pose": {"bone_rotations": {"hip": [0.1, 0.0, 0.0]}},
            "expression": {"weights": {}},
        },
    ])

    result = translate_appearance("JasonCross", "leaning forward", client, vocab=vocab)

    assert result.pose.bone_rotations == {"hip": [0.1, 0.0, 0.0]}
    assert len(client.calls) == 2
    assert "l_uparm" in client.calls[1]["user_prompt"]
    assert "not in the vocabulary" in client.calls[1]["user_prompt"]


def test_translate_appearance_raises_after_exhausted_retries():
    vocab = _small_vocab()
    client = FakeLLMClient([
        {"pose": {"bone_rotations": {"l_uparm": [0.0, 0.0, 0.0]}}, "expression": {"weights": {}}},
        {"pose": {"bone_rotations": {"l_uparm": [0.0, 0.0, 0.0]}}, "expression": {"weights": {}}},
    ])

    import pytest
    from scarecrow_pipeline.nl_appearance import AppearanceTranslationError

    with pytest.raises(AppearanceTranslationError, match="l_uparm"):
        translate_appearance("JasonCross", "leaning forward", client, vocab=vocab)

    assert len(client.calls) == 2


def test_translate_appearance_rejects_vocabulary_invalid_control_even_though_schema_valid():
    """A response can be perfectly valid PosePayload/FACSExpression JSON --
    Pydantic alone has no way to know 'facs_bs_MadeUpControl' isn't real."""
    vocab = _small_vocab()
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {"facs_bs_MadeUpControl": 1.0}}},
        {"pose": None, "expression": {"weights": {"facs_bs_MadeUpControl": 1.0}}},
    ])

    import pytest
    from scarecrow_pipeline.nl_appearance import AppearanceTranslationError

    with pytest.raises(AppearanceTranslationError, match="facs_bs_MadeUpControl"):
        translate_appearance("JasonCross", "jaw wide open", client, vocab=vocab)


def test_translate_appearance_retries_on_schema_invalid_response():
    """A response that fails Pydantic validation entirely (e.g. a weight out
    of [0,1]) also triggers the retry path, not just vocabulary mismatches."""
    vocab = _small_vocab()
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {"facs_bs_JawOpenWide": 2.5}}},
        {"pose": None, "expression": {"weights": {"facs_bs_JawOpenWide": 0.5}}},
    ])

    result = translate_appearance("JasonCross", "jaw open", client, vocab=vocab)

    assert result.expression.weights == {"facs_bs_JawOpenWide": 0.5}
    assert len(client.calls) == 2


def test_translate_appearance_uses_default_vocabulary_when_none_given():
    client = FakeLLMClient([
        {"pose": None, "expression": {"weights": {}}},
    ])

    result = translate_appearance("JasonCross", "neutral", client)

    assert result.expression.weights == {}
    # The default vocabulary (docs/posable_bones.json) is much larger than
    # the 1-bone fixture used elsewhere in this file -- a real bone name
    # proves load_default_vocabulary() was used, not an empty vocabulary.
    assert "l_upperarm" in client.calls[0]["user_prompt"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nl_appearance.py -v`
Expected: FAIL with `ImportError: cannot import name 'translate_appearance'`

- [ ] **Step 3: Write the implementation**

Add to `scarecrow_pipeline/nl_appearance.py` (at the end of the file). First add `from pydantic import ValidationError` to the top-level import block, then:

```python
def _vocabulary_errors(result: AppearanceResult, vocab: AppearanceVocabulary) -> list[str]:
    errors = []
    if result.pose is not None:
        for bone_name in result.pose.bone_rotations:
            if bone_name not in vocab.bone_names:
                errors.append(f"bone_name {bone_name!r} is not in the vocabulary")
        for bone_name in result.pose.ik_targets:
            if bone_name not in vocab.bone_names:
                errors.append(f"ik_targets bone_name {bone_name!r} is not in the vocabulary")
    for control_name in result.expression.weights:
        if control_name not in vocab.facs_control_names:
            errors.append(f"expression control {control_name!r} is not in the vocabulary")
    return errors


def translate_appearance(
    character: str,
    description: str,
    client: LLMClient,
    *,
    vocab: AppearanceVocabulary | None = None,
) -> AppearanceResult:
    vocab = vocab or load_default_vocabulary()
    schema = AppearanceResult.model_json_schema()
    system_prompt = build_system_prompt()
    base_user_prompt = build_user_prompt(character, description, vocab)

    errors: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        prompt = base_user_prompt if attempt == 0 else base_user_prompt + "\n\n" + build_retry_prompt(errors)
        raw = client.complete_json(system_prompt, prompt, schema)
        try:
            result = AppearanceResult.model_validate(raw)
        except ValidationError as exc:
            errors = [str(exc)]
            continue
        errors = _vocabulary_errors(result, vocab)
        if not errors:
            return result

    raise AppearanceTranslationError(
        f"Could not produce a valid appearance for {character!r} after "
        f"{MAX_ATTEMPTS} attempt(s): " + "; ".join(errors)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_appearance.py -v`
Expected: PASS (15 tests total)

- [ ] **Step 5: Commit**

```bash
git add scarecrow_pipeline/nl_appearance.py tests/test_nl_appearance.py
git commit -m "Add translate_appearance orchestration with bounded retry-on-invalid-output"
```

---

## Task 5: `AnthropicClient`, optional dependency, and README docs

**Files:**
- Modify: `scarecrow_pipeline/nl_appearance.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `tests/test_nl_appearance.py`

**Interfaces:**
- Consumes: `LLMClient` protocol, `AppearanceTranslationError` (Task 2).
- Produces: `AnthropicClient(model: str = "claude-sonnet-5", max_tokens: int = 4096)` implementing `complete_json`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nl_appearance.py`:

```python
def test_anthropic_client_raises_helpful_error_without_optional_dependency():
    """This repo's default install does not include the 'anthropic' package
    (it's an optional extra) -- constructing AnthropicClient without it
    installed must fail with a clear message, not a bare ModuleNotFoundError
    from deep inside the client."""
    import pytest

    from scarecrow_pipeline.nl_appearance import AnthropicClient

    try:
        import anthropic  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("anthropic is installed in this environment; nothing to verify here")

    with pytest.raises(ImportError, match="anthropic"):
        AnthropicClient()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_nl_appearance.py -v`
Expected: FAIL with `ImportError: cannot import name 'AnthropicClient'`

- [ ] **Step 3: Write the implementation**

Add to `scarecrow_pipeline/nl_appearance.py` (at the end of the file):

```python
class AnthropicClient:
    """LLMClient backed by the Anthropic Messages API's structured (tool-use)
    output.

    Requires the optional 'llm' dependency group
    (`pip install -e '.[llm]'`) and an ANTHROPIC_API_KEY environment
    variable. Never imported at module scope by nl_appearance.py or
    exercised by the default test suite -- see
    test_anthropic_client_raises_helpful_error_without_optional_dependency.
    """

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 4096):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicClient requires the 'anthropic' package -- install "
                "with `pip install -e '.[llm]'`"
            ) from exc
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def complete_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> dict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[{
                "name": "emit_appearance",
                "description": "Emit the translated pose and expression.",
                "input_schema": json_schema,
            }],
            tool_choice={"type": "tool", "name": "emit_appearance"},
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        raise AppearanceTranslationError("Anthropic response contained no tool_use block")
```

Edit `pyproject.toml`'s `[project.optional-dependencies]` table to add:

```toml
llm = ["anthropic>=0.40"]
```

(alongside the existing `vision = ["opencv-python>=4.8"]` entry, so the table reads `vision = [...]`, `llm = [...]`, `dev = [...]`).

Edit `README.md`'s `## Layout` bullet list to add, after the `scarecrow_pipeline/vision.py` line:

```markdown
- `scarecrow_pipeline/nl_appearance.py` — translates a natural-language appearance description into a validated `PosePayload`/`FACSExpression` pair via a provider-agnostic `LLMClient`; `AnthropicClient` (optional `pip install -e '.[llm]'`, `ANTHROPIC_API_KEY`) is the one concrete implementation.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nl_appearance.py -v`
Expected: PASS (16 tests total)

Then run the full suite to confirm nothing else broke:

Run: `python -m pytest -q`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add scarecrow_pipeline/nl_appearance.py pyproject.toml README.md tests/test_nl_appearance.py
git commit -m "Add AnthropicClient, optional [llm] dependency, and README docs"
```

---

## Self-Review Notes

- **Spec coverage:** `AppearanceResult`/module entry point (Task 4), `LLMClient` protocol + `AnthropicClient` (Tasks 2, 5), JSON Schema via `model_json_schema()` (Task 4), vocabulary loading verbatim (Task 1), prompt construction (Task 3), validation + one bounded retry (Task 4), `FakeLLMClient`-based tests including the vocabulary-cross-check case Pydantic alone can't catch (Task 4), optional dependency + README (Task 5). `scene_tags.py` wiring and per-character vocabulary are confirmed out of scope and untouched.
- **Placeholder scan:** no TBD/TODO; every step has concrete code.
- **Type consistency:** `translate_appearance` signature matches across Tasks 4 and the interface blocks in Tasks 2–4; `AppearanceVocabulary.bone_names`/`facs_control_names` used consistently in Tasks 1 and 4; `MAX_ATTEMPTS` defined once (Task 2), used only in Task 4.
