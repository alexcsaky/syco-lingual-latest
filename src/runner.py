"""Evaluation runner — sends translated prompts to LLM providers and records responses.

This is the core of Step D in the SycoLingual pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone

from src.config import ExperimentConfig
from src.cost import estimate_cost
from src.io import JsonlWriter, load_completed_keys, load_jsonl
from src.language_detect import detect_language, check_language_match
from src.mock import MockProvider
from src.providers.base import BaseProvider
from src.schemas import ModelResponse, TranslatedPrompt

logger = logging.getLogger(__name__)

# System prompt template — language name is inserted
SYSTEM_PROMPT_TEMPLATE = "You are a helpful assistant. Please respond in {language}."


class EvaluationRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        dry_run: bool = False,
        model_filter: str | None = None,
    ):
        self.config = config
        self.dry_run = dry_run
        self.model_filter = model_filter

    async def run(self) -> None:
        # Load prompts
        prompts = load_jsonl(self.config.paths.prompts, TranslatedPrompt)
        logger.info(f"Loaded {len(prompts)} prompts")

        # Shuffle prompts deterministically for even distribution in partial runs
        rng = random.Random(self.config.random_seed)
        rng.shuffle(prompts)

        # Scan existing responses for resumability
        completed = load_completed_keys(
            self.config.paths.responses,
            key_fields=["prompt_id", "language", "model"],
        )
        logger.info(f"Found {len(completed)} completed responses, will skip those")

        # Open writer
        writer = JsonlWriter(self.config.paths.responses)

        try:
            # Determine which models to run
            models = self.config.models
            if self.model_filter:
                models = {k: v for k, v in models.items() if k == self.model_filter}

            # Create provider for each model
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

            # Run all models in parallel
            tasks = []
            for model_name in models:
                tasks.append(
                    self._run_model(
                        model_name=model_name,
                        provider=providers[model_name],
                        prompts=prompts,
                        completed=completed,
                        writer=writer,
                    )
                )
            await asyncio.gather(*tasks)

        finally:
            writer.close()

    async def _run_model(
        self,
        model_name: str,
        provider: BaseProvider,
        prompts: list[TranslatedPrompt],
        completed: set[tuple],
        writer: JsonlWriter,
    ) -> None:
        """Run all prompts through a single model with bounded concurrency."""
        semaphore = asyncio.Semaphore(self.config.evaluation.concurrency_per_provider)

        async def process_prompt(prompt: TranslatedPrompt) -> None:
            # Check if already completed (resumability)
            key = (prompt.prompt_id, prompt.language, model_name)
            if key in completed:
                return

            async with semaphore:
                # Build system prompt
                lang_name = self.config.languages.get(prompt.language, prompt.language)
                system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=lang_name)

                # Call provider with retry
                start_time = time.monotonic()
                response = await self._call_with_retry(
                    provider, system_prompt, prompt.translated_text
                )
                latency_ms = int((time.monotonic() - start_time) * 1000)

                # Language detection
                detected_lang = detect_language(response.text)
                lang_match = check_language_match(prompt.language, detected_lang)

                # Cost estimation
                cost_config = self.config.cost_per_million_tokens.get(model_name)
                cost_usd = 0.0
                if cost_config:
                    cost_usd = estimate_cost(
                        response.input_tokens,
                        response.output_tokens,
                        cost_config.input,
                        cost_config.output,
                    )

                # Build response record
                record = ModelResponse(
                    prompt_id=prompt.prompt_id,
                    item_id=prompt.item_id,
                    facet=prompt.facet,
                    variant=prompt.variant,
                    language=prompt.language,
                    prompt_text=prompt.translated_text,
                    model=model_name,
                    model_version=response.model_version,
                    response_text=response.text,
                    response_tokens=response.output_tokens,
                    reasoning_tokens=response.reasoning_tokens,
                    finish_reason=response.finish_reason,
                    detected_language=detected_lang,
                    language_match=lang_match,
                    timestamp=datetime.now(timezone.utc),
                    latency_ms=latency_ms,
                    run_id=self.config.run_id,
                    estimated_cost_usd=cost_usd,
                )

                await writer.write(record)

        # Process all prompts concurrently (bounded by semaphore)
        tasks = [process_prompt(p) for p in prompts]
        await asyncio.gather(*tasks)

    async def _call_with_retry(
        self,
        provider: BaseProvider,
        system_prompt: str,
        user_message: str,
    ):
        """Call provider.complete() with exponential backoff retry."""
        delay = self.config.evaluation.retry_initial_delay_seconds
        for attempt in range(self.config.evaluation.max_retries):
            try:
                return await provider.complete(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    temperature=self.config.evaluation.temperature,
                    max_tokens=self.config.evaluation.max_tokens,
                )
            except Exception as e:
                if attempt == self.config.evaluation.max_retries - 1:
                    raise
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay *= 2
