"""OpenAI-compatible provider adapter.

Works with any API that follows the OpenAI chat completions format:
OpenAI, xAI (Grok), Moonshot (Kimi), DeepSeek.
"""

from __future__ import annotations

import logging

import httpx

from src.providers.base import BaseProvider
from src.schemas import ProviderResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseProvider):
    def __init__(
        self,
        family: str,
        model_id: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
    ):
        super().__init__(family=family, model_id=model_id)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return ProviderResponse(
            text=choice["message"]["content"],
            model_version=data.get("model", self.model_id),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            finish_reason=choice.get("finish_reason", "unknown"),
            raw_response=data,
        )

    async def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        """Call with JSON response format. Uses json_schema response_format where supported."""
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_output",
                    "schema": output_schema,
                    "strict": True,
                },
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return ProviderResponse(
            text=choice["message"]["content"],
            model_version=data.get("model", self.model_id),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            finish_reason=choice.get("finish_reason", "unknown"),
            raw_response=data,
        )
