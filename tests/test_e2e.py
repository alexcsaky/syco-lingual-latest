# tests/test_e2e.py
import json
from pathlib import Path

import pytest

from src.config import load_config
from src.runner import EvaluationRunner
from src.judge import JudgingModule
from src.schemas import ModelResponse, JudgeScore, ScoredItem


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self, tmp_path):
        """Run the complete pipeline with mock providers and verify outputs."""
        # Setup
        prompts_path = "tests/fixtures/sample_prompts.jsonl"
        responses_path = str(tmp_path / "responses.jsonl")
        judgements_path = str(tmp_path / "judgements.jsonl")
        scored_path = str(tmp_path / "scored.jsonl")
        judge_prompts_dir = tmp_path / "judge_prompts"
        judge_prompts_dir.mkdir()
        for facet in ["mirroring", "side_taking", "attribution_bias", "delusion"]:
            for lang in ["en", "ja"]:
                (judge_prompts_dir / f"{facet}_{lang}.txt").write_text("You are a judge.")

        yaml_content = f"""
run_id: "e2e_test"
random_seed: 42
models:
  model-a:
    provider: mock
    family: family_a
    model_id: "mock-1.0"
judges:
  judge-1:
    provider: mock
    family: family_x
    model_id: "mock-1.0"
  judge-2:
    provider: mock
    family: family_y
    model_id: "mock-1.0"
  judge-3:
    provider: mock
    family: family_z
    model_id: "mock-1.0"
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
  prompts: "{prompts_path}"
  responses: "{responses_path}"
  judgements: "{judgements_path}"
  judgements_english: "{tmp_path / 'eng_val.jsonl'}"
  judge_prompts_dir: "{judge_prompts_dir}"
  fixtures_dir: "x"
languages:
  en: "English"
  ja: "\u65e5\u672c\u8a9e"
cost_per_million_tokens: {{}}
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)
        config = load_config(str(config_path))

        # Step D: Evaluation
        runner = EvaluationRunner(config, dry_run=True)
        await runner.run()

        # Verify responses
        responses = []
        with open(responses_path) as f:
            for line in f:
                responses.append(ModelResponse.model_validate_json(line))
        assert len(responses) == 6  # 6 prompts x 1 model

        # Step F: Judging
        module = JudgingModule(config, dry_run=True)
        await module.run()

        # Verify judge scores
        scores = []
        with open(judgements_path) as f:
            for line in f:
                scores.append(JudgeScore.model_validate_json(line))
        assert len(scores) == 18  # 6 responses x 3 judges

        # Step G: Aggregation
        module.aggregate(judgements_path, scored_path)

        scored = []
        with open(scored_path) as f:
            for line in f:
                scored.append(ScoredItem.model_validate_json(line))
        # Aggregation groups by (prompt_id, model).  The fixture has 3 unique
        # prompt_ids (mirror_001_a, mirror_001_b, delusion_001), each appearing
        # in 2 languages, so we get 3 scored items.  Each group has 6 judge
        # scores (3 judges x 2 languages) but the dict is keyed by judge_family,
        # collapsing to 3 entries per group.
        assert len(scored) == 3  # 3 unique (prompt_id, model) pairs
        assert all(s.is_valid for s in scored)  # All have 3+ judges
        assert all(s.median_score is not None for s in scored)
