import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.judge import JudgingModule, compute_median, select_validation_subset
from src.config import load_config
from src.schemas import ModelResponse, JudgeScore, ScoredItem


def _make_response(prompt_id, model, language="en", facet="mirroring", variant="a"):
    return ModelResponse(
        prompt_id=prompt_id, item_id=prompt_id.rsplit("_", 1)[0],
        facet=facet, variant=variant, language=language,
        prompt_text="test prompt", model=model, model_version="v1",
        response_text="test response about the topic",
        response_tokens=50, reasoning_tokens=0, finish_reason="stop",
        detected_language=language, language_match=True,
        timestamp=datetime.now(timezone.utc), latency_ms=100,
        run_id="test", estimated_cost_usd=0.001,
    )


class TestComputeMedian:
    def test_five_scores(self):
        assert compute_median([1, 2, 3, 4, 5]) == 3.0

    def test_three_scores(self):
        assert compute_median([1, 3, 5]) == 3.0

    def test_four_scores(self):
        assert compute_median([1, 2, 3, 4]) == 2.5

    def test_two_scores_returns_none(self):
        assert compute_median([1, 2]) is None

    def test_empty_returns_none(self):
        assert compute_median([]) is None


class TestSelectValidationSubset:
    def test_selects_25_percent(self):
        item_ids = [f"mirror_{i:03d}" for i in range(40)]
        selected = select_validation_subset(
            item_ids_by_facet={"mirroring": item_ids},
            seed=42,
        )
        assert len(selected) == 10  # 25% of 40

    def test_deterministic(self):
        item_ids = [f"mirror_{i:03d}" for i in range(40)]
        s1 = select_validation_subset({"mirroring": item_ids}, seed=42)
        s2 = select_validation_subset({"mirroring": item_ids}, seed=42)
        assert s1 == s2

    def test_different_seed_different_result(self):
        item_ids = [f"mirror_{i:03d}" for i in range(40)]
        s1 = select_validation_subset({"mirroring": item_ids}, seed=42)
        s2 = select_validation_subset({"mirroring": item_ids}, seed=99)
        assert s1 != s2


class TestJudgingModule:
    @pytest.mark.asyncio
    async def test_judges_all_responses(self, tmp_path):
        # Create response file with 2 responses
        responses_path = tmp_path / "responses.jsonl"
        r1 = _make_response("mirror_001_a", "mock-model")
        r2 = _make_response("delusion_001", "mock-model", facet="delusion", variant="none")
        responses_path.write_text(r1.model_dump_json() + "\n" + r2.model_dump_json() + "\n")

        judgements_path = tmp_path / "judgements.jsonl"
        judge_prompts_dir = tmp_path / "judge_prompts"
        judge_prompts_dir.mkdir()
        # Create minimal judge prompt files
        for facet in ["mirroring", "delusion"]:
            (judge_prompts_dir / f"{facet}_en.txt").write_text("You are a judge. Score -5 to 5.")

        config = self._make_config(tmp_path, responses_path, judgements_path, judge_prompts_dir)
        module = JudgingModule(config, dry_run=True)
        await module.run()

        scores = []
        with open(judgements_path) as f:
            for line in f:
                scores.append(JudgeScore.model_validate_json(line))

        # 2 responses x 2 mock judges = 4 judge scores
        assert len(scores) == 4

    @pytest.mark.asyncio
    async def test_self_family_flag(self, tmp_path):
        responses_path = tmp_path / "responses.jsonl"
        r = _make_response("mirror_001_a", "mock-model")
        responses_path.write_text(r.model_dump_json() + "\n")

        judgements_path = tmp_path / "judgements.jsonl"
        judge_prompts_dir = tmp_path / "judge_prompts"
        judge_prompts_dir.mkdir()
        (judge_prompts_dir / "mirroring_en.txt").write_text("Judge prompt.")

        config = self._make_config(tmp_path, responses_path, judgements_path, judge_prompts_dir)
        module = JudgingModule(config, dry_run=True)
        await module.run()

        with open(judgements_path) as f:
            for line in f:
                s = JudgeScore.model_validate_json(line)
                assert isinstance(s.self_family, bool)

    @pytest.mark.asyncio
    async def test_aggregation(self, tmp_path):
        # Create 5 judge scores for one (prompt, model) pair
        judgements_path = tmp_path / "judgements.jsonl"
        scores_text = ""
        for i, family in enumerate(["openai", "anthropic", "google", "xai", "deepseek"]):
            s = JudgeScore(
                prompt_id="mirror_001_a", item_id="mirror_001",
                facet="mirroring", language="en", model="test-model",
                judge_model=f"judge-{family}", judge_family=family,
                self_family=False, score=i + 1,
                justification="test", judging_language="target",
                timestamp=datetime.now(timezone.utc), run_id="test",
            )
            scores_text += s.model_dump_json() + "\n"
        judgements_path.write_text(scores_text)

        scored_path = tmp_path / "scored.jsonl"
        config = self._make_config(tmp_path, tmp_path / "r.jsonl", judgements_path, tmp_path)
        module = JudgingModule(config, dry_run=True)
        module.aggregate(str(judgements_path), str(scored_path))

        with open(scored_path) as f:
            items = [ScoredItem.model_validate_json(line) for line in f]

        assert len(items) == 1
        assert items[0].median_score == 3.0
        assert items[0].valid_judges == 5
        assert items[0].is_valid is True

    @pytest.mark.asyncio
    async def test_non_dry_run_uses_create_provider(self, tmp_path):
        """Non-dry-run with mock provider should work (create_provider handles mock)."""
        responses_path = tmp_path / "responses.jsonl"
        r1 = _make_response("mirror_001_a", "mock-model")
        r2 = _make_response("delusion_001", "mock-model", facet="delusion", variant="none")
        responses_path.write_text(r1.model_dump_json() + "\n" + r2.model_dump_json() + "\n")

        judgements_path = tmp_path / "judgements.jsonl"
        judge_prompts_dir = tmp_path / "judge_prompts"
        judge_prompts_dir.mkdir()
        for facet in ["mirroring", "delusion"]:
            (judge_prompts_dir / f"{facet}_en.txt").write_text("You are a judge.")

        config = self._make_config(tmp_path, responses_path, judgements_path, judge_prompts_dir)
        module = JudgingModule(config, dry_run=False)
        await module.run()

        scores = []
        with open(judgements_path) as f:
            for line in f:
                scores.append(JudgeScore.model_validate_json(line))
        # 2 responses x 2 mock judges = 4 scores
        assert len(scores) == 4

    def _make_config(self, tmp_path, responses_path, judgements_path, judge_prompts_dir):
        yaml_content = f"""
run_id: "test"
random_seed: 42
models:
  mock-model:
    provider: mock
    family: mock_family
    model_id: "mock-1.0"
judges:
  mock-judge-1:
    provider: mock
    family: mock_judge_a
    model_id: "mock-1.0"
  mock-judge-2:
    provider: mock
    family: mock_judge_b
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
  prompts: "x"
  responses: "{responses_path}"
  judgements: "{judgements_path}"
  judgements_english: "{tmp_path / 'eng_val.jsonl'}"
  judge_prompts_dir: "{judge_prompts_dir}"
  fixtures_dir: "x"
languages:
  en: "English"
cost_per_million_tokens: {{}}
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)
        return load_config(str(config_path))
