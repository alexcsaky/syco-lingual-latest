import json
from pathlib import Path

import pytest

from src.runner import EvaluationRunner
from src.config import load_config
from src.schemas import ModelResponse


@pytest.fixture
def sample_prompts_path():
    return "tests/fixtures/sample_prompts.jsonl"


@pytest.fixture
def mock_config(tmp_path, sample_prompts_path):
    """Create a minimal config for testing with mock providers."""
    responses_path = str(tmp_path / "responses.jsonl")
    yaml_content = f"""
run_id: "test_run"
random_seed: 42
models:
  mock-model-1:
    provider: mock
    family: mock_family_a
    model_id: "mock-1.0"
  mock-model-2:
    provider: mock
    family: mock_family_b
    model_id: "mock-1.0"
judges: {{}}
evaluation:
  temperature: 0.0
  max_tokens: 1024
  concurrency_per_provider: 5
  max_retries: 3
  retry_initial_delay_seconds: 0.01
judging:
  temperature: 0.0
  max_tokens: 256
  concurrency_per_provider: 5
  max_retries: 3
  validation_subset_fraction: 0.25
paths:
  prompts: "{sample_prompts_path}"
  responses: "{responses_path}"
  judgements: "x"
  judgements_english: "x"
  judge_templates: "x"
  fixtures_dir: "x"
languages:
  en: "English"
  ja: "日本語"
cost_per_million_tokens: {{}}
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)
    return load_config(str(config_path))


class TestEvaluationRunner:
    @pytest.mark.asyncio
    async def test_runs_all_prompts_for_all_models(self, mock_config):
        runner = EvaluationRunner(mock_config, dry_run=True)
        await runner.run()

        responses = []
        with open(mock_config.paths.responses) as f:
            for line in f:
                responses.append(json.loads(line))

        # 6 prompts x 2 models = 12 responses
        assert len(responses) == 12

    @pytest.mark.asyncio
    async def test_responses_have_correct_schema(self, mock_config):
        runner = EvaluationRunner(mock_config, dry_run=True)
        await runner.run()

        with open(mock_config.paths.responses) as f:
            for line in f:
                r = ModelResponse.model_validate_json(line)
                assert r.run_id == "test_run"
                assert r.finish_reason == "stop"
                assert r.reasoning_tokens == 0

    @pytest.mark.asyncio
    async def test_resumability(self, mock_config):
        runner = EvaluationRunner(mock_config, dry_run=True)
        await runner.run()

        # Count initial responses
        with open(mock_config.paths.responses) as f:
            count1 = sum(1 for _ in f)

        # Run again — should skip all, write nothing new
        runner2 = EvaluationRunner(mock_config, dry_run=True)
        await runner2.run()

        with open(mock_config.paths.responses) as f:
            count2 = sum(1 for _ in f)

        assert count1 == count2

    @pytest.mark.asyncio
    async def test_prompt_text_stored_in_response(self, mock_config):
        runner = EvaluationRunner(mock_config, dry_run=True)
        await runner.run()

        with open(mock_config.paths.responses) as f:
            for line in f:
                r = ModelResponse.model_validate_json(line)
                assert len(r.prompt_text) > 0

    @pytest.mark.asyncio
    async def test_non_dry_run_uses_create_provider(self, mock_config):
        """Non-dry-run with mock provider should work (create_provider handles mock)."""
        runner = EvaluationRunner(mock_config, dry_run=False)
        await runner.run()

        responses = []
        with open(mock_config.paths.responses) as f:
            for line in f:
                responses.append(json.loads(line))
        # Same as dry_run: 6 prompts x 2 models = 12 responses
        assert len(responses) == 12

    @pytest.mark.asyncio
    async def test_language_detection_populated(self, mock_config):
        runner = EvaluationRunner(mock_config, dry_run=True)
        await runner.run()

        with open(mock_config.paths.responses) as f:
            for line in f:
                r = ModelResponse.model_validate_json(line)
                assert r.detected_language != ""
                assert isinstance(r.language_match, bool)
