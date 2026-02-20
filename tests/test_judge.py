import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.judge import JudgingModule, compute_median, parse_judge_score, select_validation_subset
from src.config import load_config
from src.mock import MockProvider
from src.schemas import ModelResponse, JudgeScore, ScoredItem, TranslatedPrompt

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_JUDGES = str(FIXTURES_DIR / "sample_judges.jsonl")


def _make_response(prompt_uid, model, lang="en", facet="mirror", run="stance1"):
    return ModelResponse(
        prompt_uid=prompt_uid, item_id=1,
        facet=facet, run=run, lang=lang,
        chain="deepl_deepl",
        prompt_text="test prompt", model=model, model_version="v1",
        response_text="test response about the topic",
        response_tokens=50, reasoning_tokens=0, finish_reason="stop",
        detected_language=lang, language_match=True,
        timestamp=datetime.now(timezone.utc), latency_ms=100,
        run_id="test", estimated_cost_usd=0.001,
    )


def _make_prompts_file(path: Path):
    """Write a minimal prompts JSONL with stance1/stance2 for mirror facet."""
    prompts = [
        TranslatedPrompt(
            prompt_uid="mirror_001_stance1", item_id=1, facet="mirror",
            run="stance1", lang="en", composed_prompt="stance1 prompt",
            composed_prompt_en="I believe X is true",
            back_translation_en="I believe X is true",
            comet_score=1.0, chrf_score=1.0, word_overlap=1.0,
            sanity_checks=None,
            fields_translated={"stance1": "X is true", "stance2": "X is false"},
            domain_tag="politics",
            difficulty_tag="easy", severity_tag="low", delusion_type="",
            chain="deepl_deepl", fwd_engine="deepl", back_engine="deepl",
            fwd_time_s=0.1, back_time_s=0.1,
        ),
        TranslatedPrompt(
            prompt_uid="mirror_001_stance2", item_id=1, facet="mirror",
            run="stance2", lang="en", composed_prompt="stance2 prompt",
            composed_prompt_en="I believe X is false",
            back_translation_en="I believe X is false",
            comet_score=1.0, chrf_score=1.0, word_overlap=1.0,
            sanity_checks=None,
            fields_translated={"stance1": "X is true", "stance2": "X is false"},
            domain_tag="politics",
            difficulty_tag="easy", severity_tag="low", delusion_type="",
            chain="deepl_deepl", fwd_engine="deepl", back_engine="deepl",
            fwd_time_s=0.1, back_time_s=0.1,
        ),
    ]
    with open(path, "w") as f:
        for p in prompts:
            f.write(p.model_dump_json() + "\n")


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


class TestParseJudgeScore:
    """Tests for the resilient score parser that handles messy judge outputs."""

    # --- Plain integers (fast path) ---

    def test_bare_integer(self):
        assert parse_judge_score("3") == 3

    def test_negative_integer(self):
        assert parse_judge_score("-5") == -5

    def test_zero(self):
        assert parse_judge_score("0") == 0

    def test_whitespace_around_integer(self):
        assert parse_judge_score("  3  ") == 3

    # --- "Score: N" prefix (801 production failures) ---

    def test_score_prefix(self):
        assert parse_judge_score("Score: 3") == 3

    def test_score_prefix_negative(self):
        assert parse_judge_score("Score: -5") == -5

    def test_score_prefix_with_justification(self):
        assert parse_judge_score("Score: 2\n\nThe model partially...") == 2

    # --- Markdown bold "Score: **N**" ---

    def test_score_markdown_bold(self):
        assert parse_judge_score("Score: **-3**") == -3

    def test_score_markdown_bold_positive(self):
        assert parse_judge_score("Score: **4**") == 4

    # --- Integer + trailing text (88 production failures) ---

    def test_integer_with_trailing_justification(self):
        assert parse_judge_score("3\n\nBecause the model...") == 3

    def test_integer_with_newline_only(self):
        assert parse_judge_score("3\n") == 3

    # --- Empty / whitespace → None ---

    def test_empty_string(self):
        assert parse_judge_score("") is None

    def test_whitespace_only(self):
        assert parse_judge_score("   ") is None

    # --- Full text with no extractable integer → None ---

    def test_full_text_no_integer(self):
        assert parse_judge_score("The model did a good job of answering") is None

    def test_refusal_text(self):
        assert parse_judge_score("I cannot provide a score for this response.") is None


class TestJudgingModule:
    @pytest.mark.asyncio
    async def test_judges_all_responses(self, tmp_path):
        # Create prompts file with stances for mirror facet
        prompts_path = tmp_path / "prompts.jsonl"
        _make_prompts_file(prompts_path)

        # Create response file with 2 responses
        responses_path = tmp_path / "responses.jsonl"
        r1 = _make_response("mirror_001_stance1", "mock-model")
        r2 = _make_response("delusion_001", "mock-model", facet="delusion", run="single")
        responses_path.write_text(r1.model_dump_json() + "\n" + r2.model_dump_json() + "\n")

        judgements_path = tmp_path / "judgements.jsonl"

        config = self._make_config(tmp_path, responses_path, judgements_path, prompts_path)
        module = JudgingModule(config, dry_run=True)
        await module.run()

        scores = []
        with open(judgements_path) as f:
            for line in f:
                scores.append(JudgeScore.model_validate_json(line))

        # 2 responses x 2 mock judges = 4 max, but delusion scores
        # outside [0, 5] are discarded by range validation
        assert len(scores) >= 2  # at least 1 mirror response x 2 judges

    @pytest.mark.asyncio
    async def test_self_family_flag(self, tmp_path):
        # Create prompts file with stances for mirror facet
        prompts_path = tmp_path / "prompts.jsonl"
        _make_prompts_file(prompts_path)

        responses_path = tmp_path / "responses.jsonl"
        r = _make_response("mirror_001_stance1", "mock-model")
        responses_path.write_text(r.model_dump_json() + "\n")

        judgements_path = tmp_path / "judgements.jsonl"

        config = self._make_config(tmp_path, responses_path, judgements_path, prompts_path)
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
                prompt_uid="mirror_001_a", item_id=1,
                facet="mirror", run="stance1", lang="en", chain="deepl_deepl",
                model="test-model", judge_model=f"judge-{family}",
                judge_family=family, self_family=False, score=i + 1,
                justification="", judging_language="target",
                timestamp=datetime.now(timezone.utc), run_id="test",
            )
            scores_text += s.model_dump_json() + "\n"
        judgements_path.write_text(scores_text)

        scored_path = tmp_path / "scored.jsonl"
        prompts_path = tmp_path / "prompts.jsonl"
        prompts_path.write_text("")  # empty prompts for aggregate-only test
        config = self._make_config(tmp_path, tmp_path / "r.jsonl", judgements_path, prompts_path)
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
        # Create prompts file with stances for mirror facet
        prompts_path = tmp_path / "prompts.jsonl"
        _make_prompts_file(prompts_path)

        responses_path = tmp_path / "responses.jsonl"
        r1 = _make_response("mirror_001_stance1", "mock-model")
        r2 = _make_response("delusion_001", "mock-model", facet="delusion", run="single")
        responses_path.write_text(r1.model_dump_json() + "\n" + r2.model_dump_json() + "\n")

        judgements_path = tmp_path / "judgements.jsonl"

        config = self._make_config(tmp_path, responses_path, judgements_path, prompts_path)
        module = JudgingModule(config, dry_run=False)
        await module.run()

        scores = []
        with open(judgements_path) as f:
            for line in f:
                scores.append(JudgeScore.model_validate_json(line))
        # 2 responses x 2 mock judges = 4 max, minus any out-of-range delusion scores
        assert len(scores) >= 2

    @pytest.mark.asyncio
    async def test_score_range_validation_rejects_out_of_range(self, tmp_path):
        """Delusion scores outside [0, 5] should be discarded by range validation."""
        from unittest.mock import AsyncMock
        from src.schemas import ProviderResponse

        prompts_path = tmp_path / "prompts.jsonl"
        _make_prompts_file(prompts_path)

        # Create only a delusion response
        responses_path = tmp_path / "responses.jsonl"
        r = _make_response("delusion_001", "mock-model", facet="delusion", run="single")
        responses_path.write_text(r.model_dump_json() + "\n")

        judgements_path = tmp_path / "judgements.jsonl"
        config = self._make_config(tmp_path, responses_path, judgements_path, prompts_path)

        module = JudgingModule(config, dry_run=True)

        # Provider that always returns -3 (out of delusion range [0, 5])
        class FixedScoreProvider(MockProvider):
            async def complete(self, system_prompt, user_message, temperature, max_tokens):
                return ProviderResponse(
                    text="-3", model_version="mock-1.0",
                    input_tokens=10, output_tokens=1, reasoning_tokens=0,
                    finish_reason="stop", raw_response={},
                )
        from src.judge_templates import load_judge_templates
        from src.io import JsonlWriter, load_completed_keys, load_jsonl
        from src.schemas import ModelResponse as MR

        templates = load_judge_templates(config.paths.judge_templates)
        responses = load_jsonl(config.paths.responses, MR)
        prompts = load_jsonl(str(prompts_path), TranslatedPrompt)

        # Build stance lookup
        stance_lookup: dict = {}
        for p in prompts:
            if p.facet == "mirror" and p.fields_translated:
                key = (p.item_id, p.lang, p.chain)
                if key not in stance_lookup:
                    stance_lookup[key] = {
                        "stance1": p.fields_translated.get("stance1", ""),
                        "stance2": p.fields_translated.get("stance2", ""),
                    }

        provider = FixedScoreProvider(family="test", model_id="test-1.0")
        writer = JsonlWriter(str(judgements_path))
        try:
            await module._run_judge(
                judge_name="test-judge",
                judge_family="test_family",
                provider=provider,
                responses=responses,
                model_families={"mock-model": "mock_family"},
                completed=set(),
                writer=writer,
                templates=templates,
                stance_lookup=stance_lookup,
            )
        finally:
            writer.close()

        # All scores should be discarded (delusion range is [0, 5], score was -3)
        scores = []
        if judgements_path.exists():
            with open(judgements_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        scores.append(JudgeScore.model_validate_json(line))
        assert len(scores) == 0, f"Expected 0 scores but got {len(scores)}"

    def _make_config(self, tmp_path, responses_path, judgements_path, prompts_path):
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
  prompts: "{prompts_path}"
  responses: "{responses_path}"
  judgements: "{judgements_path}"
  judgements_english: "{tmp_path / 'eng_val.jsonl'}"
  judge_templates: "{SAMPLE_JUDGES}"
  fixtures_dir: "x"
languages:
  en: "English"
cost_per_million_tokens: {{}}
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)
        return load_config(str(config_path))
