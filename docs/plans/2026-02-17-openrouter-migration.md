# OpenRouter Migration + Mistral Addition — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace 6 direct provider APIs with a single OpenRouter endpoint, add Mistral Large as 7th eval model + Mistral Small as 6th judge.

**Architecture:** All models accessed through OpenRouter's OpenAI-compatible `/v1/chat/completions` endpoint. Single `OpenAICompatibleProvider` with OpenRouter base URL. Mock provider unchanged for dry-run testing.

**Tech Stack:** Python, httpx, Pydantic, OpenRouter API (OpenAI-compatible format)

**Design doc:** `docs/plans/2026-02-17-openrouter-migration-design.md`

---

### Task 1: Rewrite Provider Registry for OpenRouter

**Files:**
- Modify: `src/providers/__init__.py` (full rewrite — currently 84 lines)
- Modify: `tests/test_providers/test_registry.py` (full rewrite — currently 61 lines)

**Step 1: Write the failing tests**

Replace `tests/test_providers/test_registry.py` entirely:

```python
import pytest
from src.providers import create_provider
from src.config import ModelConfig
from src.providers.base import BaseProvider
from src.mock import MockProvider
from src.providers.openai_compat import OpenAICompatibleProvider


class TestCreateProvider:
    def test_mock_provider(self):
        config = ModelConfig(provider="mock", family="mock", model_id="mock-1.0")
        p = create_provider("test", config, {})
        assert isinstance(p, MockProvider)

    def test_openrouter_provider(self):
        config = ModelConfig(
            provider="openrouter",
            family="openai",
            model_id="openai/gpt-5.1",
        )
        p = create_provider("test", config, {"OPENROUTER_API_KEY": "fake-key"})
        assert isinstance(p, OpenAICompatibleProvider)
        assert p.family == "openai"
        assert p.model_id == "openai/gpt-5.1"
        assert p._base_url == "https://openrouter.ai/api/v1"

    def test_openrouter_passes_family_from_config(self):
        config = ModelConfig(
            provider="openrouter",
            family="mistral",
            model_id="mistralai/mistral-large-2512",
        )
        p = create_provider("test", config, {"OPENROUTER_API_KEY": "fake-key"})
        assert p.family == "mistral"

    def test_unknown_provider_raises(self):
        config = ModelConfig(provider="unknown", family="test", model_id="test")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("test", config, {})

    def test_missing_api_key_raises(self):
        config = ModelConfig(
            provider="openrouter",
            family="openai",
            model_id="openai/gpt-5.1",
        )
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            create_provider("test", config, {})
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_providers/test_registry.py -v`
Expected: FAIL — old registry doesn't know about "openrouter"

**Step 3: Write the implementation**

Replace `src/providers/__init__.py` entirely:

```python
"""Provider registry — factory for creating provider instances from config."""

from __future__ import annotations

from src.config import ModelConfig
from src.providers.base import BaseProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_provider(
    name: str,
    model_config: ModelConfig,
    api_keys: dict[str, str],
) -> BaseProvider:
    """Create a provider instance from config.

    Args:
        name: Friendly name for logging
        model_config: ModelConfig from experiment.yaml
        api_keys: Dict of env var name -> API key value

    Returns:
        A BaseProvider instance ready to make API calls

    Raises:
        ValueError: If provider is unknown or API key is missing
    """
    provider_name = model_config.provider

    if provider_name == "mock":
        from src.mock import MockProvider
        return MockProvider(family=model_config.family, model_id=model_config.model_id)

    if provider_name == "openrouter":
        from src.providers.openai_compat import OpenAICompatibleProvider
        api_key = api_keys.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found for OpenRouter (expected OPENROUTER_API_KEY)"
            )
        return OpenAICompatibleProvider(
            family=model_config.family,
            model_id=model_config.model_id,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )

    raise ValueError(f"Unknown provider: {provider_name}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_providers/test_registry.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/providers/__init__.py tests/test_providers/test_registry.py
git commit -m "refactor: rewrite provider registry for OpenRouter-only"
```

---

### Task 2: Add OpenRouter Headers to OpenAICompatibleProvider

**Files:**
- Modify: `src/providers/openai_compat.py` (lines 48-55 and 100-107)
- Modify: `tests/test_providers/test_openai_compat.py` (full rewrite)

**Step 1: Write the failing test**

Replace `tests/test_providers/test_openai_compat.py` entirely:

```python
import pytest
from src.providers.openai_compat import OpenAICompatibleProvider


class TestOpenAICompatibleProviderInit:
    def test_default_base_url(self):
        p = OpenAICompatibleProvider(
            family="openai", model_id="openai/gpt-5.1", api_key="fake",
        )
        assert p._base_url == "https://api.openai.com/v1"

    def test_custom_base_url(self):
        p = OpenAICompatibleProvider(
            family="openai", model_id="openai/gpt-5.1", api_key="fake",
            base_url="https://openrouter.ai/api/v1",
        )
        assert p._base_url == "https://openrouter.ai/api/v1"

    def test_openrouter_headers_included(self):
        p = OpenAICompatibleProvider(
            family="openai", model_id="openai/gpt-5.1", api_key="fake",
            base_url="https://openrouter.ai/api/v1",
        )
        headers = p._build_headers()
        assert headers["Authorization"] == "Bearer fake"
        assert "HTTP-Referer" in headers
        assert "X-Title" in headers

    def test_non_openrouter_no_extra_headers(self):
        p = OpenAICompatibleProvider(
            family="openai", model_id="gpt-5", api_key="fake",
            base_url="https://api.openai.com/v1",
        )
        headers = p._build_headers()
        assert "HTTP-Referer" not in headers
        assert "X-Title" not in headers
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_providers/test_openai_compat.py -v`
Expected: FAIL — `_build_headers()` method doesn't exist yet

**Step 3: Write the implementation**

Modify `src/providers/openai_compat.py` — extract header building into `_build_headers()` and add OpenRouter-specific headers when the base URL is OpenRouter:

```python
"""OpenAI-compatible provider adapter.

Works with any API that follows the OpenAI chat completions format.
When base_url points to OpenRouter, includes recommended OpenRouter headers.
"""

from __future__ import annotations

import logging

import httpx

from src.providers.base import BaseProvider
from src.schemas import ProviderResponse

logger = logging.getLogger(__name__)

_OPENROUTER_HOST = "openrouter.ai"


class OpenAICompatibleProvider(BaseProvider):
    def __init__(
        self,
        family: str,
        model_id: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
    ):
        super().__init__(family=family, model_id=model_id)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if _OPENROUTER_HOST in self._base_url:
            headers["HTTP-Referer"] = "https://github.com/alexcsaky/syco-lingual-latest"
            headers["X-Title"] = "SycoLingual"
        return headers

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return ProviderResponse(
            text=choice["message"]["content"],
            model_version=data.get("model", self.model_id),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            finish_reason=choice.get("finish_reason", "unknown"),
            raw_response=data,
        )

    async def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        """Call with JSON response format. Uses json_schema response_format."""
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_output",
                    "schema": output_schema,
                    "strict": True,
                },
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return ProviderResponse(
            text=choice["message"]["content"],
            model_version=data.get("model", self.model_id),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            finish_reason=choice.get("finish_reason", "unknown"),
            raw_response=data,
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_providers/test_openai_compat.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/providers/openai_compat.py tests/test_providers/test_openai_compat.py
git commit -m "feat: add OpenRouter headers to OpenAI-compatible provider"
```

---

### Task 3: Delete Provider-Specific Adapters and Tests

**Files to delete:**
- `src/providers/anthropic.py`
- `src/providers/google.py`
- `src/providers/xai.py`
- `src/providers/moonshot.py`
- `src/providers/deepseek.py`
- `tests/test_providers/test_anthropic.py`
- `tests/test_providers/test_google.py`

**Step 1: Verify no remaining imports of deleted modules**

Search for imports of deleted modules across the codebase. The only consumers should be:
- `src/providers/__init__.py` — already rewritten in Task 1, no longer imports these
- `tests/test_providers/test_registry.py` — already rewritten in Task 1
- `tests/test_providers/test_openai_compat.py` — already rewritten in Task 2

Run: `grep -r "from src.providers.anthropic\|from src.providers.google\|from src.providers.xai\|from src.providers.moonshot\|from src.providers.deepseek" src/ tests/`
Expected: No matches (all imports were removed in Tasks 1-2)

**Step 2: Delete the files**

```bash
rm src/providers/anthropic.py src/providers/google.py src/providers/xai.py src/providers/moonshot.py src/providers/deepseek.py
rm tests/test_providers/test_anthropic.py tests/test_providers/test_google.py
```

**Step 3: Run full test suite to verify nothing breaks**

Run: `pytest -v`
Expected: All tests pass (74 existing minus the deleted tests + new tests from Tasks 1-2)

**Step 4: Commit**

```bash
git add -u  # stages deletions
git commit -m "refactor: remove provider-specific adapters replaced by OpenRouter"
```

---

### Task 4: Update Experiment Config for OpenRouter

**Files:**
- Modify: `config/experiment.yaml` (full rewrite)

**Step 1: Write the updated config**

Replace `config/experiment.yaml` entirely:

```yaml
run_id: "v2_main_001"
random_seed: 42

models:
  claude-sonnet-4-6:
    provider: openrouter
    family: anthropic
    model_id: "anthropic/claude-sonnet-4.6"
  gpt-5.1:
    provider: openrouter
    family: openai
    model_id: "openai/gpt-5.1"
  gemini-3-flash:
    provider: openrouter
    family: google
    model_id: "google/gemini-3-flash-preview"
  grok-4.1:
    provider: openrouter
    family: xai
    model_id: "x-ai/grok-4.1-fast"
  kimi-k2.5:
    provider: openrouter
    family: moonshot
    model_id: "moonshotai/kimi-k2.5"
  deepseek-v3.2:
    provider: openrouter
    family: deepseek
    model_id: "deepseek/deepseek-v3.2"
  mistral-large:
    provider: openrouter
    family: mistral
    model_id: "mistralai/mistral-large-2512"

judges:
  gpt-5.1-mini:
    provider: openrouter
    family: openai
    model_id: "openai/gpt-5.1-codex-mini"
  claude-haiku-4-5:
    provider: openrouter
    family: anthropic
    model_id: "anthropic/claude-haiku-4-5"
  gemini-3-flash:
    provider: openrouter
    family: google
    model_id: "google/gemini-3-flash-preview"
  grok-4.1-fast:
    provider: openrouter
    family: xai
    model_id: "x-ai/grok-4.1-fast"
  deepseek-v3.2:
    provider: openrouter
    family: deepseek
    model_id: "deepseek/deepseek-v3.2"
  mistral-small:
    provider: openrouter
    family: mistral
    model_id: "mistralai/mistral-small-2503"

evaluation:
  temperature: 0.0
  max_tokens: 1024
  concurrency_per_provider: 5
  max_retries: 5
  retry_initial_delay_seconds: 1.0

judging:
  temperature: 0.0
  max_tokens: 256
  concurrency_per_provider: 10
  max_retries: 3
  validation_subset_fraction: 0.25

paths:
  prompts: "data/prompts/translated_prompts.jsonl"
  responses: "data/responses/responses.jsonl"
  judgements: "data/judgements/judgements.jsonl"
  judgements_english: "data/judgements/judgements_english_validation.jsonl"
  judge_prompts_dir: "config/judge_prompts"
  fixtures_dir: "data/fixtures"

languages:
  en: "English"
  ja: "日本語"
  bn: "বাংলা"
  de: "Deutsch"
  es: "Español"
  fr: "Français"
  zh: "中文"
  ar: "العربية"
  sl: "Slovenščina"
  lv: "Latviešu"

cost_per_million_tokens: {}
```

**Step 2: Verify config loads**

Run: `python -c "from src.config import load_config; c = load_config('config/experiment.yaml'); print(f'{len(c.models)} models, {len(c.judges)} judges')"`
Expected: `7 models, 6 judges`

**Step 3: Commit**

```bash
git add config/experiment.yaml
git commit -m "config: update experiment.yaml for OpenRouter with 7 models and 6 judges"
```

---

### Task 5: Wire create_provider() into Runner and Judge

**Files:**
- Modify: `src/runner.py` (lines 65-76)
- Modify: `src/judge.py` (lines 95-105)

**Step 1: Write tests for non-dry-run provider creation**

No new test file needed — the existing mock-based tests still work (dry_run=True path unchanged). We add one test each to verify the create_provider wiring exists:

Add to `tests/test_runner.py`:

```python
    @pytest.mark.asyncio
    async def test_non_dry_run_requires_api_key(self, mock_config):
        """Non-dry-run should attempt to create real provider and fail without API key."""
        runner = EvaluationRunner(mock_config, dry_run=False)
        with pytest.raises(ValueError, match="Unknown provider|OPENROUTER_API_KEY"):
            await runner.run()
```

Add to `tests/test_judge.py` inside `TestJudgingModule`:

```python
    @pytest.mark.asyncio
    async def test_non_dry_run_requires_api_key(self, tmp_path):
        """Non-dry-run should attempt to create real provider and fail without API key."""
        responses_path = tmp_path / "responses.jsonl"
        r = _make_response("mirror_001_a", "mock-model")
        responses_path.write_text(r.model_dump_json() + "\n")

        judgements_path = tmp_path / "judgements.jsonl"
        judge_prompts_dir = tmp_path / "judge_prompts"
        judge_prompts_dir.mkdir()
        (judge_prompts_dir / "mirroring_en.txt").write_text("Judge.")

        config = self._make_config(tmp_path, responses_path, judgements_path, judge_prompts_dir)
        module = JudgingModule(config, dry_run=False)
        with pytest.raises(ValueError, match="Unknown provider|OPENROUTER_API_KEY"):
            await module.run()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py::TestEvaluationRunner::test_non_dry_run_requires_api_key tests/test_judge.py::TestJudgingModule::test_non_dry_run_requires_api_key -v`
Expected: FAIL — current code raises `NotImplementedError`, not `ValueError`

**Step 3: Implement the changes**

**In `src/runner.py`**, replace lines 63-76 (the provider creation block):

```python
# Before (lines 63-76):
            providers: dict[str, BaseProvider] = {}
            for model_name, model_config in models.items():
                if self.dry_run:
                    providers[model_name] = MockProvider(
                        family=model_config.family,
                        model_id=model_config.model_id,
                    )
                else:
                    # Real providers will be created here once implemented
                    raise NotImplementedError(
                        f"Real provider for {model_config.provider} not yet implemented"
                    )
```

```python
# After:
            providers: dict[str, BaseProvider] = {}
            for model_name, model_config in models.items():
                if self.dry_run:
                    providers[model_name] = MockProvider(
                        family=model_config.family,
                        model_id=model_config.model_id,
                    )
                else:
                    from src.providers import create_provider
                    api_keys = {
                        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
                    }
                    providers[model_name] = create_provider(
                        model_name, model_config, api_keys
                    )
```

Also add `import os` to the imports at the top of `src/runner.py`.

**In `src/judge.py`**, replace lines 95-105 (the judge provider creation block):

```python
# Before (lines 95-105):
            judges: dict[str, BaseProvider] = {}
            for judge_name, judge_config in self.config.judges.items():
                if self.dry_run:
                    judges[judge_name] = MockProvider(
                        family=judge_config.family,
                        model_id=judge_config.model_id,
                    )
                else:
                    raise NotImplementedError(
                        f"Real provider for {judge_config.provider} not yet implemented"
                    )
```

```python
# After:
            judges: dict[str, BaseProvider] = {}
            for judge_name, judge_config in self.config.judges.items():
                if self.dry_run:
                    judges[judge_name] = MockProvider(
                        family=judge_config.family,
                        model_id=judge_config.model_id,
                    )
                else:
                    from src.providers import create_provider
                    api_keys = {
                        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
                    }
                    judges[judge_name] = create_provider(
                        judge_name, judge_config, api_keys
                    )
```

Also add `import os` to the imports at the top of `src/judge.py`.

**Step 4: Run all tests**

Run: `pytest -v`
Expected: All tests PASS (both old and new)

**Step 5: Commit**

```bash
git add src/runner.py src/judge.py tests/test_runner.py tests/test_judge.py
git commit -m "feat: wire create_provider() into runner and judge for non-dry-run mode"
```

---

### Task 6: Add Mistral Judge Prompt Placeholders

**Files:**
- Create: `config/judge_prompts/mirroring_*.txt` (only if missing for new facets)
- Check: Existing 40 judge prompt files already cover 4 facets × 10 languages

**Step 1: Verify existing judge prompt files**

The judge prompt files are named `{facet}_{language}.txt`. The existing 40 files cover all 4 facets × 10 languages. These are shared by ALL judges (the same prompt file is used regardless of which judge model reads it), so no new files are needed for adding a Mistral judge.

Run: `ls config/judge_prompts/ | wc -l`
Expected: 40

No code changes needed. This task is verification only.

**Step 2: Commit (nothing to commit)**

Skip — no changes.

---

### Task 7: Update E2E Test Config for OpenRouter

**Files:**
- Modify: `tests/test_e2e.py` (update YAML config to use provider: mock, add Mistral to judges)

**Step 1: Update E2E test**

The E2E test's YAML config uses `provider: mock` — this still works. But we should verify the test reflects 6 judges (not 3) for a realistic test. Also update the expected count.

In `tests/test_e2e.py`, update the YAML config inside `test_full_pipeline_mock` to include a 4th mock judge and update assertions:

No structural changes to the E2E test are needed — it uses `provider: mock` throughout, which is still valid. The test currently has 3 judges and asserts 18 judge scores (6 responses × 3 judges). This is fine for testing the pipeline mechanics.

**Step 2: Run full test suite**

Run: `pytest -v`
Expected: All tests pass

**Step 3: Commit (if any changes)**

Skip if no changes needed.

---

### Task 8: Update Session Log and Final Verification

**Files:**
- Modify: `docs/claude-notes/master-session-log.md`

**Step 1: Run full test suite**

Run: `pytest -v`
Expected: All tests pass

**Step 2: Check no dead imports remain**

Run: `grep -r "from src.providers.anthropic\|from src.providers.google\|from src.providers.xai\|from src.providers.moonshot\|from src.providers.deepseek" src/ tests/`
Expected: No output

**Step 3: Update session log**

Append to `docs/claude-notes/master-session-log.md`:

```markdown
---

## Session: 2026-02-17 — OpenRouter Migration + Mistral Addition

### Changes Made
- Replaced 6 direct provider APIs with single OpenRouter endpoint
- Deleted: anthropic.py, google.py, xai.py, moonshot.py, deepseek.py provider adapters
- Added OpenRouter headers (HTTP-Referer, X-Title) to OpenAICompatibleProvider
- Rewrote provider registry: 2 entries (mock + openrouter) instead of 7
- Added Mistral Large as 7th evaluated model
- Added Mistral Small as 6th judge
- Updated all model IDs to latest OpenRouter slugs
- Wired create_provider() into runner.py and judge.py for non-dry-run mode
- Updated experiment.yaml for 7 models + 6 judges

### What's Needed Before Production
- OPENROUTER_API_KEY in .env
- Real translated prompts from Tanzim
- Real judge system prompts from Tanzim
- Verify OpenRouter model IDs are current (check openrouter.ai/models)
```

**Step 4: Commit**

```bash
git add docs/claude-notes/master-session-log.md
git commit -m "docs: update session log with OpenRouter migration"
```
