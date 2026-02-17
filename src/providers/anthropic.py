"""Anthropic provider adapter for Claude models via the Messages API."""

from __future__ import annotations

import json
import logging

import httpx

from src.providers.base import BaseProvider
from src.schemas import ProviderResponse

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    def __init__(self, model_id: str, api_key: str):
        super().__init__(family="anthropic", model_id=model_id)
        self._api_key = api_key

    def _headers(self) -> dict:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        payload = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message},
            ],
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # Extract text from content blocks
        text_parts = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
        text = "\n".join(text_parts)

        usage = data.get("usage", {})

        return ProviderResponse(
            text=text,
            model_version=data.get("model", self.model_id),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=0,  # No reasoning tokens in standard mode
            finish_reason=data.get("stop_reason", "unknown"),
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
        """Use Anthropic's tool use to get structured JSON output.

        Defines a tool with the desired output schema, then forces the model
        to use that tool via tool_choice.
        """
        tool = {
            "name": "judge_output",
            "description": "Output the judge's score and justification",
            "input_schema": output_schema,
        }

        payload = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message},
            ],
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "judge_output"},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # Extract the tool use input (which is the structured JSON output)
        tool_input = {}
        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "judge_output":
                tool_input = block.get("input", {})
                break

        usage = data.get("usage", {})

        return ProviderResponse(
            text=json.dumps(tool_input),
            model_version=data.get("model", self.model_id),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=0,
            finish_reason=data.get("stop_reason", "unknown"),
            raw_response=data,
        )
