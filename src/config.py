"""Configuration loading and validation for SycoLingual v2.

Loads experiment parameters from YAML, validates via Pydantic.
API keys come from environment variables / .env file.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    provider: str
    family: str
    model_id: str


class EvaluationConfig(BaseModel):
    temperature: float
    max_tokens: int
    concurrency_per_provider: int
    max_retries: int
    retry_initial_delay_seconds: float


class JudgingConfig(BaseModel):
    temperature: float
    max_tokens: int
    concurrency_per_provider: int
    max_retries: int
    validation_subset_fraction: float


class PathsConfig(BaseModel):
    prompts: str
    responses: str
    judgements: str
    judgements_english: str
    judge_prompts_dir: str
    fixtures_dir: str


class CostRate(BaseModel):
    input: float
    output: float


class ExperimentConfig(BaseModel):
    run_id: str
    random_seed: int
    models: dict[str, ModelConfig]
    judges: dict[str, ModelConfig]
    evaluation: EvaluationConfig
    judging: JudgingConfig
    paths: PathsConfig
    languages: dict[str, str]
    cost_per_million_tokens: dict[str, CostRate]


def load_config(yaml_path: str) -> ExperimentConfig:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)
