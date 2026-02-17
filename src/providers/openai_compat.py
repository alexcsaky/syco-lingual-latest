"""OpenAI-compatible provider adapter.

Works with any API that follows the OpenAI chat completions format.
When base_url points to OpenRouter, includes recommended OpenRouter headers.
"""

from __future__ import annotations

import logging

import httpx

from src.providers.base import BaseProvider
from src.schemas import ProviderResponse

logger = logging.getLogger(__name__)

_OPENROUTER_HOST = "openrouter.ai"


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

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if _OPENROUTER_HOST in self._base_url:
            headers["HTTP-Referer"] = "https://github.com/alexcsaky/syco-lingual-latest"
            headers["X-Title"] = "SycoLingual"
        return headers

    def _build_messages(self, system_prompt: str, user_message: str) -> list[dict]:
        """Build messages array with prompt caching for OpenRouter.

        When routing through OpenRouter, uses content array format with
        cache_control on the system prompt. This enables Anthropic prompt
        caching (other providers cache automatically).
        """
        if _OPENROUTER_HOST in self._base_url:
            system_msg = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        else:
            system_msg = {"role": "system", "content": system_prompt}

        return [system_msg, {"role": "user", "content": user_message}]

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        payload = {
            "model": self.model_id,
            "messages": self._build_messages(system_prompt, user_message),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._build_headers(),
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
        """Call with JSON response format. Uses json_schema response_format."""
        payload = {
            "model": self.model_id,
            "messages": self._build_messages(system_prompt, user_message),
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
                headers=self._build_headers(),
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
