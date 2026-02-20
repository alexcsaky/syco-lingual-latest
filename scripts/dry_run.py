"""Dry-run script: exercises the full pipeline with MockProvider against real test data.

Usage:
    python -m scripts.dry_run
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.runner import EvaluationRunner
from src.judge import JudgingModule
from src.io import load_jsonl
from src.schemas import ModelResponse, JudgeScore, ScoredItem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("dry_run")

CONFIG_PATH = "config/experiment.yaml"
SCORED_OUTPUT = "data/responses/scored_items.jsonl"


async def main() -> None:
    config = load_config(CONFIG_PATH)

    # --- Clean previous dry-run outputs ---
    for path in [config.paths.responses, config.paths.judgements, SCORED_OUTPUT]:
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            p.unlink()
            logger.info("Cleaned previous output: %s", path)

    # --- Stage 1: Evaluation (model responses) ---
    logger.info("=" * 60)
    logger.info("STAGE 1: Running evaluation (dry-run with MockProvider)")
    logger.info("=" * 60)

    runner = EvaluationRunner(config, dry_run=True)
    await runner.run()

    responses = load_jsonl(config.paths.responses, ModelResponse)
    logger.info("Stage 1 complete: %d model responses written", len(responses))

    # Quick stats
    models = set(r.model for r in responses)
    langs = set(r.lang for r in responses)
    facets = set(r.facet for r in responses)
    logger.info("  Models: %s", sorted(models))
    logger.info("  Languages: %s", sorted(langs))
    logger.info("  Facets: %s", sorted(facets))

    # --- Stage 2: Judging ---
    logger.info("=" * 60)
    logger.info("STAGE 2: Running judging (dry-run with MockProvider)")
    logger.info("=" * 60)

    judging = JudgingModule(config, dry_run=True)
    await judging.run()

    scores = load_jsonl(config.paths.judgements, JudgeScore)
    logger.info("Stage 2 complete: %d judge scores written", len(scores))

    judges = set(s.judge_model for s in scores)
    logger.info("  Judge models: %s", sorted(judges))

    # --- Stage 3: Aggregation ---
    logger.info("=" * 60)
    logger.info("STAGE 3: Aggregating scores")
    logger.info("=" * 60)

    judging.aggregate(config.paths.judgements, SCORED_OUTPUT)

    items = load_jsonl(SCORED_OUTPUT, ScoredItem)
    valid_count = sum(1 for i in items if i.is_valid)
    logger.info("Stage 3 complete: %d scored items (%d valid)", len(items), valid_count)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("DRY-RUN SUMMARY")
    logger.info("=" * 60)
    logger.info("  Prompts loaded (after chain selection) → responses: %d", len(responses))
    logger.info("  Judge scores: %d", len(scores))
    logger.info("  Scored items: %d  (valid: %d, invalid: %d)",
                len(items), valid_count, len(items) - valid_count)
    logger.info("  Output files:")
    logger.info("    %s  (%d bytes)", config.paths.responses,
                Path(config.paths.responses).stat().st_size)
    logger.info("    %s  (%d bytes)", config.paths.judgements,
                Path(config.paths.judgements).stat().st_size)
    logger.info("    %s  (%d bytes)", SCORED_OUTPUT,
                Path(SCORED_OUTPUT).stat().st_size)


if __name__ == "__main__":
    asyncio.run(main())
