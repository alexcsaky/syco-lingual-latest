# OpenRouter Migration + Mistral Addition — Design Document

**Date:** 2026-02-17
**Status:** Approved via brainstorm session

## Summary

Migrate from 6 direct provider APIs to a single OpenRouter endpoint, and add Mistral Large as a 7th evaluated model with a corresponding Mistral judge.

## Motivation

1. **Simplicity**: One API key, one base URL, one provider adapter instead of six.
2. **Mistral addition**: OpenRouter provides access to Mistral models without needing a separate Mistral API integration.
3. **Operational**: No need to manage 6+ API keys across different provider dashboards. Single billing, single rate limit surface.
4. **Production readiness**: The runner and judge currently raise `NotImplementedError` for non-dry-run mode — this migration wires up real provider creation via `create_provider()`.

## Architecture Change

### Before
```
runner.py / judge.py
    └── create_provider()
        ├── OpenAICompatibleProvider (api.openai.com)
        ├── AnthropicProvider (api.anthropic.com — Messages API, tool use)
        ├── GoogleProvider (generativelanguage.googleapis.com — Gemini API)
        ├── XAIProvider (api.x.ai — OpenAI-compat)
        ├── MoonshotProvider (api.moonshot.cn — OpenAI-compat)
        └── DeepSeekProvider (api.deepseek.com — OpenAI-compat)
```

### After
```
runner.py / judge.py
    └── create_provider()
        ├── MockProvider (dry-run)
        └── OpenAICompatibleProvider (openrouter.ai/api/v1)
                → all 13 models via OpenRouter slugs
```

## Updated Model Lineup

### 7 Evaluated Models

| Friendly Name     | OpenRouter Model ID                | Family   |
|--------------------|------------------------------------|----------|
| claude-sonnet-4-6  | `anthropic/claude-sonnet-4.6`     | anthropic|
| gpt-5.1            | `openai/gpt-5.1`                 | openai   |
| gemini-3-flash     | `google/gemini-3-flash-preview`   | google   |
| grok-4.1           | `x-ai/grok-4.1-fast`             | xai      |
| kimi-k2.5          | `moonshotai/kimi-k2.5`           | moonshot |
| deepseek-v3.2      | `deepseek/deepseek-v3.2`         | deepseek |
| mistral-large      | `mistralai/mistral-large-2512`    | mistral  |

### 6 Judge Models

| Judge Name         | OpenRouter Model ID                | Family   |
|--------------------|------------------------------------|----------|
| gpt-5.1-mini       | `openai/gpt-5.1-codex-mini`      | openai   |
| claude-haiku-4-5   | `anthropic/claude-haiku-4-5`      | anthropic|
| gemini-3-flash     | `google/gemini-3-flash-preview`   | google   |
| grok-4.1-fast      | `x-ai/grok-4.1-fast`             | xai      |
| deepseek-v3.2      | `deepseek/deepseek-v3.2`         | deepseek |
| mistral-small      | `mistralai/mistral-small-2503`    | mistral  |

**Notes:**
- Some OpenRouter model IDs (Claude Haiku, Mistral Small) need runtime verification — the API listing was partial during design. Config is easy to update.
- Scale: 7 models × 280 prompts/lang × 10 langs = 19,600 eval calls. 6 judges × 19,600 = 117,600 judge calls.
- self_family flag: every evaluated model family has a matching judge family.

## Structured Output Strategy

All models accessed through OpenRouter use the OpenAI-compatible `json_schema` response format:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "judge_output",
      "schema": { ... },
      "strict": true
    }
  }
}
```

OpenRouter passes this through to each provider. Models that don't support strict schemas gracefully degrade. Our existing retry logic handles parse failures.

## Files to Delete

- `src/providers/anthropic.py` — Anthropic Messages API adapter (replaced by OpenRouter)
- `src/providers/google.py` — Google Gemini API adapter (replaced by OpenRouter)
- `src/providers/xai.py` — xAI adapter (was just OpenAI-compat with different base_url)
- `src/providers/moonshot.py` — Moonshot adapter (was just OpenAI-compat with different base_url)
- `src/providers/deepseek.py` — DeepSeek adapter (was just OpenAI-compat with different base_url)
- `tests/test_providers/test_anthropic.py` — Tests for deleted Anthropic adapter
- `tests/test_providers/test_google.py` — Tests for deleted Google adapter

## Files to Modify

### `src/providers/openai_compat.py`
- Add OpenRouter-recommended headers (`HTTP-Referer`, `X-Title`) for better rate limits
- No structural changes — already handles `/v1/chat/completions` format

### `src/providers/__init__.py`
- Collapse registry to 2 entries: `mock` and `openrouter`
- Factory creates `OpenAICompatibleProvider` with:
  - `base_url="https://openrouter.ai/api/v1"`
  - `api_key` from `OPENROUTER_API_KEY` env var
  - `model_id` passed straight from config (OpenRouter slug)
  - `family` from config

### `config/experiment.yaml`
- All models: `provider: openrouter`
- Model IDs: OpenRouter slugs
- Add 7th model (Mistral Large) and 6th judge (Mistral Small)
- Remove per-provider key references

### `src/config.py`
- Valid providers: `mock`, `openrouter`
- Remove per-provider API key env var logic
- Single `OPENROUTER_API_KEY` validation

### `src/runner.py`
- Replace `NotImplementedError` block with `create_provider()` call
- Load `OPENROUTER_API_KEY` from environment

### `src/judge.py`
- Same as runner — wire in `create_provider()` for non-dry-run mode
- Load `OPENROUTER_API_KEY` from environment

### Tests
- `tests/test_providers/test_registry.py` — Update for new 2-entry registry
- `tests/test_providers/test_openai_compat.py` — Add OpenRouter header test
- `tests/test_e2e.py` — Update config fixture (provider: mock still works)
- `tests/test_runner.py` — Update for create_provider wiring
- `tests/test_judge.py` — Update for create_provider wiring

## Files Unchanged

- `src/schemas.py` — Data contracts are provider-agnostic
- `src/providers/base.py` — ABC unchanged
- `src/mock.py` — Mock provider unchanged
- `src/io.py` — File I/O unchanged
- `src/language_detect.py` — Unchanged
- `src/cost.py` — Unchanged
- `src/translation.py` — Unchanged
- `run.py` — CLI unchanged

## Decisions

1. **Full replacement** — Delete all provider-specific adapters. No fallback to direct APIs.
2. **Latest model versions** — Use what's currently on OpenRouter (Claude 4.6, GPT 5.1, etc.)
3. **json_schema structured output** — Unified across all judges via OpenRouter's pass-through.
4. **6-judge panel** — Added Mistral judge to maintain self_family coverage symmetry.
5. **OpenRouter headers** — Include `HTTP-Referer` and `X-Title` for attribution and rate limit benefits.
