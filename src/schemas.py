"""Pydantic data contracts for the SycoLingual v2 pipeline.

These schemas define the structure of all data flowing between pipeline stages.
They serve as enforceable contracts between the translation pipeline (Tanzim)
and the evaluation/judging pipeline (this codebase).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


# --- Score ranges per facet (used for judge score validation) ---

FACET_SCORE_RANGES: dict[str, tuple[int, int]] = {
    "mirroring": (-5, 5),
    "side_taking": (-5, 5),
    "attribution_bias": (-5, 5),
    "delusion": (0, 5),
}


# --- Input schema (from Tanzim's translation pipeline, step C) ---

class TranslatedPrompt(BaseModel):
    prompt_id: str
    item_id: str
    facet: Literal["mirroring", "side_taking", "attribution_bias", "delusion"]
    variant: Literal["a", "b", "none"]
    language: str
    english_text: str
    translated_text: str
    comet_score: float | None


# --- Provider response (internal, not persisted) ---

class ProviderResponse(BaseModel):
    text: str
    model_version: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    finish_reason: str
    raw_response: dict


# --- Model evaluation output (step E) ---

class ModelResponse(BaseModel):
    prompt_id: str
    item_id: str
    facet: str
    variant: str
    language: str
    prompt_text: str
    model: str
    model_version: str
    response_text: str
    response_tokens: int
    reasoning_tokens: int
    finish_reason: str
    detected_language: str
    language_match: bool
    timestamp: datetime
    latency_ms: int
    run_id: str
    estimated_cost_usd: float


# --- Judging output (step G) ---

class JudgeScore(BaseModel):
    prompt_id: str
    item_id: str
    facet: str
    language: str
    model: str
    judge_model: str
    judge_family: str
    self_family: bool
    score: int
    justification: str
    judging_language: str
    timestamp: datetime
    run_id: str


class ScoredItem(BaseModel):
    prompt_id: str
    item_id: str
    facet: str
    language: str
    model: str
    judge_scores: dict[str, int]
    median_score: float | None
    valid_judges: int
    is_valid: bool
    judging_language: str
    run_id: str
