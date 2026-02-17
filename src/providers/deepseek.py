"""DeepSeek provider — uses OpenAI-compatible API."""

from __future__ import annotations

from src.providers.openai_compat import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    def __init__(self, model_id: str, api_key: str):
        super().__init__(
            family="deepseek",
            model_id=model_id,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
