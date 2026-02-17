"""Provider registry — factory for creating provider instances from config."""

from __future__ import annotations

from src.config import ModelConfig
from src.providers.base import BaseProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_provider(
    name: str,
    model_config: ModelConfig,
    api_keys: dict[str, str],
) -> BaseProvider:
    """Create a provider instance from config.

    Args:
        name: Friendly name for logging
        model_config: ModelConfig from experiment.yaml
        api_keys: Dict of env var name -> API key value

    Returns:
        A BaseProvider instance ready to make API calls

    Raises:
        ValueError: If provider is unknown or API key is missing
    """
    provider_name = model_config.provider

    if provider_name == "mock":
        from src.mock import MockProvider
        return MockProvider(family=model_config.family, model_id=model_config.model_id)

    if provider_name == "openrouter":
        from src.providers.openai_compat import OpenAICompatibleProvider
        api_key = api_keys.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found for OpenRouter (expected OPENROUTER_API_KEY)"
            )
        return OpenAICompatibleProvider(
            family=model_config.family,
            model_id=model_config.model_id,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )

    raise ValueError(f"Unknown provider: {provider_name}")
