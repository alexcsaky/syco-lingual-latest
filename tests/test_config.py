# tests/test_config.py
import os
import tempfile
from pathlib import Path
from src.config import ExperimentConfig, load_config


class TestExperimentConfig:
    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
run_id: "test_001"
random_seed: 42

models:
  test-model:
    provider: openai
    family: openai
    model_id: "test-model-v1"

judges:
  test-judge:
    provider: openai
    family: openai
    model_id: "test-judge-v1"

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

cost_per_million_tokens: {}
"""
        config_file = tmp_path / "experiment.yaml"
        config_file.write_text(yaml_content)

        config = load_config(str(config_file))
        assert config.run_id == "test_001"
        assert config.random_seed == 42
        assert config.evaluation.temperature == 0.0
        assert config.evaluation.max_tokens == 1024
        assert "test-model" in config.models
        assert config.models["test-model"].family == "openai"
        assert config.languages["en"] == "English"

    def test_model_config_fields(self, tmp_path):
        yaml_content = """
run_id: "test"
random_seed: 1
models:
  my-model:
    provider: anthropic
    family: anthropic
    model_id: "claude-sonnet-4-5-20250929"
judges: {}
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
  prompts: "x"
  responses: "x"
  judgements: "x"
  judgements_english: "x"
  judge_prompts_dir: "x"
  fixtures_dir: "x"
languages: {}
cost_per_million_tokens: {}
"""
        config_file = tmp_path / "experiment.yaml"
        config_file.write_text(yaml_content)
        config = load_config(str(config_file))
        m = config.models["my-model"]
        assert m.provider == "anthropic"
        assert m.family == "anthropic"
        assert m.model_id == "claude-sonnet-4-5-20250929"
