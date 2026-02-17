"""Google Gemini provider adapter."""

from __future__ import annotations

import logging

import httpx

from src.providers.base import BaseProvider
from src.schemas import ProviderResponse

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GoogleProvider(BaseProvider):
    def __init__(self, model_id: str, api_key: str):
        super().__init__(family="google", model_id=model_id)
        self._api_key = api_key

    def _url(self, action: str = "generateContent") -> str:
        return f"{GEMINI_API_BASE}/{self.model_id}:{action}?key={self._api_key}"

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {"role": "user", "parts": [{"text": user_message}]},
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self._url(),
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # Extract text from candidates
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})

        return ProviderResponse(
            text=text,
            model_version=data.get("modelVersion", self.model_id),
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            reasoning_tokens=0,
            finish_reason=candidates[0].get("finishReason", "unknown") if candidates else "unknown",
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
        """Use Gemini's native JSON mode with response_mime_type and response_schema."""
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {"role": "user", "parts": [{"text": user_message}]},
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "responseSchema": output_schema,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self._url(),
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})

        return ProviderResponse(
            text=text,
            model_version=data.get("modelVersion", self.model_id),
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            reasoning_tokens=0,
            finish_reason=candidates[0].get("finishReason", "unknown") if candidates else "unknown",
            raw_response=data,
        )
