"""Mock provider for testing the pipeline without real API calls."""

from __future__ import annotations

import hashlib
import json

from src.providers.base import BaseProvider
from src.schemas import ProviderResponse


class MockProvider(BaseProvider):
    def __init__(self, family: str = "mock", model_id: str = "mock-1.0"):
        super().__init__(family=family, model_id=model_id)

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        # Deterministic mock response based on input hash
        h = hashlib.md5(f"{system_prompt}{user_message}".encode()).hexdigest()

        # Detect judge calls: the judging pipeline sends an empty user_message
        # and the system prompt contains the filled judge template.
        # Also keep the legacy English keyword check for unit tests.
        is_judge = (
            not user_message.strip()
            or ("judge" in system_prompt.lower() and "integer" in system_prompt.lower())
        )
        if is_judge:
            score = int(h[:2], 16) % 11 - 5  # Range -5 to +5
            text = str(score)
        else:
            text = f"[MOCK-{h[:8]}] This is a mock response to the given prompt. "
            text += "The model provides a balanced perspective on the topic at hand."

        return ProviderResponse(
            text=text,
            model_version="mock-1.0",
            input_tokens=len(user_message.split()),
            output_tokens=len(text.split()),
            reasoning_tokens=0,
            finish_reason="stop",
            raw_response={},
        )

    async def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        # Deterministic mock score based on input hash
        h = hashlib.md5(f"{system_prompt}{user_message}".encode()).hexdigest()
        score = int(h[:2], 16) % 11 - 5  # Range -5 to +5
        mock_output = json.dumps({
            "score": score,
            "justification": "Mock evaluation of the model response."
        })

        return ProviderResponse(
            text=mock_output,
            model_version="mock-1.0",
            input_tokens=len(user_message.split()),
            output_tokens=len(mock_output.split()),
            reasoning_tokens=0,
            finish_reason="stop",
            raw_response={},
        )
