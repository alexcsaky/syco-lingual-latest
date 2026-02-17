"""Moonshot (Kimi) provider — uses OpenAI-compatible API."""

from __future__ import annotations

from src.providers.openai_compat import OpenAICompatibleProvider


class MoonshotProvider(OpenAICompatibleProvider):
    def __init__(self, model_id: str, api_key: str):
        super().__init__(
            family="moonshot",
            model_id=model_id,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        )
