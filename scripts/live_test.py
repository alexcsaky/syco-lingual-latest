"""Live test run: exercises the full pipeline with real OpenRouter API calls.

Runs a scoped subset (2 models, 3 judges) to validate end-to-end without
burning through the full experiment budget.

Usage:
    OPENROUTER_API_KEY=sk-... python -m scripts.live_test

Or export the key first:
    export OPENROUTER_API_KEY=sk-...
    python -m scripts.live_test
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ExperimentConfig
from src.runner import EvaluationRunner
from src.judge import JudgingModule
from src.io import load_jsonl
from src.schemas import ModelResponse, JudgeScore, ScoredItem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("live_test")

CONFIG_PATH = "config/experiment.yaml"
RESPONSES_PATH = "data/responses/live_test_responses.jsonl"
JUDGEMENTS_PATH = "data/judgements/live_test_judgements.jsonl"
SCORED_PATH = "data/responses/live_test_scored.jsonl"

# Subset: 2 cheap/fast models, 3 cheap/fast judges
TEST_MODELS = {"deepseek-v3.2", "gemini-3-flash"}
TEST_JUDGES = {"deepseek-v3.2", "gemini-3-flash", "mistral-small"}


def make_test_config(base_config: ExperimentConfig) -> ExperimentConfig:
    """Create a scoped config for the live test."""
    return ExperimentConfig(
        run_id="live_test_001",
        random_seed=base_config.random_seed,
        models={k: v for k, v in base_config.models.items() if k in TEST_MODELS},
        judges={k: v for k, v in base_config.judges.items() if k in TEST_JUDGES},
        evaluation=base_config.evaluation,
        judging=base_config.judging,
        paths=base_config.paths.model_copy(update={
            "responses": RESPONSES_PATH,
            "judgements": JUDGEMENTS_PATH,
        }),
        languages=base_config.languages,
        cost_per_million_tokens=base_config.cost_per_million_tokens,
    )


async def main() -> None:
    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set. Export it before running.")
        sys.exit(1)
    logger.info("API key found (%d chars)", len(api_key))

    # Load and scope config
    base_config = load_config(CONFIG_PATH)
    config = make_test_config(base_config)
    logger.info("Test config: %d models, %d judges", len(config.models), len(config.judges))
    logger.info("  Models: %s", list(config.models.keys()))
    logger.info("  Judges: %s", list(config.judges.keys()))

    # Clean previous test outputs
    for path in [RESPONSES_PATH, JUDGEMENTS_PATH, SCORED_PATH]:
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            p.unlink()
            logger.info("Cleaned: %s", path)

    # --- Stage 1: Evaluation ---
    logger.info("=" * 60)
    logger.info("STAGE 1: Evaluation (real API calls)")
    logger.info("=" * 60)

    t0 = time.monotonic()
    runner = EvaluationRunner(config, dry_run=False)
    await runner.run()
    eval_time = time.monotonic() - t0

    responses = load_jsonl(RESPONSES_PATH, ModelResponse)
    logger.info("Stage 1 complete: %d responses in %.1fs", len(responses), eval_time)

    # Show a sample
    if responses:
        r = responses[0]
        logger.info("  Sample: model=%s lang=%s facet=%s tokens=%d",
                     r.model, r.lang, r.facet, r.response_tokens)
        logger.info("  Response preview: %s", r.response_text[:150])

    # Language match stats
    match_count = sum(1 for r in responses if r.language_match)
    logger.info("  Language match: %d/%d (%.0f%%)",
                match_count, len(responses),
                100 * match_count / len(responses) if responses else 0)

    # --- Stage 2: Judging ---
    logger.info("=" * 60)
    logger.info("STAGE 2: Judging (real API calls)")
    logger.info("=" * 60)

    t0 = time.monotonic()
    judging = JudgingModule(config, dry_run=False)
    await judging.run()
    judge_time = time.monotonic() - t0

    scores = load_jsonl(JUDGEMENTS_PATH, JudgeScore)
    logger.info("Stage 2 complete: %d scores in %.1fs", len(scores), judge_time)

    # Parse failure stats
    expected_scores = len(responses) * len(config.judges)
    parse_failures = expected_scores - len(scores)
    logger.info("  Expected: %d, Got: %d, Parse failures: %d (%.0f%%)",
                expected_scores, len(scores), parse_failures,
                100 * parse_failures / expected_scores if expected_scores else 0)

    # --- Stage 3: Aggregation ---
    logger.info("=" * 60)
    logger.info("STAGE 3: Aggregation")
    logger.info("=" * 60)

    judging.aggregate(JUDGEMENTS_PATH, SCORED_PATH)

    items = load_jsonl(SCORED_PATH, ScoredItem)
    valid_count = sum(1 for i in items if i.is_valid)
    logger.info("Stage 3 complete: %d scored items (%d valid, %d invalid)",
                len(items), valid_count, len(items) - valid_count)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("LIVE TEST SUMMARY")
    logger.info("=" * 60)
    logger.info("  Models: %s", list(config.models.keys()))
    logger.info("  Judges: %s", list(config.judges.keys()))
    logger.info("  Responses: %d (eval time: %.1fs)", len(responses), eval_time)
    logger.info("  Judge scores: %d (judge time: %.1fs)", len(scores), judge_time)
    logger.info("  Scored items: %d (valid: %d)", len(items), valid_count)
    logger.info("  Language match rate: %d/%d (%.0f%%)",
                match_count, len(responses),
                100 * match_count / len(responses) if responses else 0)
    logger.info("  Parse failure rate: %d/%d (%.0f%%)",
                parse_failures, expected_scores,
                100 * parse_failures / expected_scores if expected_scores else 0)
    logger.info("  Total wall time: %.1fs", eval_time + judge_time)

    # Show score distribution
    if items:
        valid_items = [i for i in items if i.median_score is not None]
        if valid_items:
            medians = [i.median_score for i in valid_items]
            logger.info("  Median score distribution: min=%.1f, max=%.1f, mean=%.1f",
                        min(medians), max(medians), sum(medians) / len(medians))


if __name__ == "__main__":
    asyncio.run(main())
