"""Provider registry — factory for creating provider instances from config."""

from __future__ import annotations

from src.config import ModelConfig
from src.providers.base import BaseProvider


# Map provider names to (env var name for API key, constructor function)
_PROVIDER_REGISTRY: dict[str, tuple[str, type]] = {}


def _register_providers() -> None:
    """Lazily populate the registry to avoid circular imports."""
    global _PROVIDER_REGISTRY
    if _PROVIDER_REGISTRY:
        return

    from src.mock import MockProvider
    from src.providers.openai_compat import OpenAICompatibleProvider
    from src.providers.anthropic import AnthropicProvider
    from src.providers.google import GoogleProvider
    from src.providers.xai import XAIProvider
    from src.providers.moonshot import MoonshotProvider
    from src.providers.deepseek import DeepSeekProvider

    _PROVIDER_REGISTRY = {
        "mock": ("", MockProvider),
        "openai": ("OPENAI_API_KEY", OpenAICompatibleProvider),
        "anthropic": ("ANTHROPIC_API_KEY", AnthropicProvider),
        "google": ("GOOGLE_API_KEY", GoogleProvider),
        "xai": ("XAI_API_KEY", XAIProvider),
        "moonshot": ("MOONSHOT_API_KEY", MoonshotProvider),
        "deepseek": ("DEEPSEEK_API_KEY", DeepSeekProvider),
    }


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
    _register_providers()

    provider_name = model_config.provider
    if provider_name not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_name}")

    env_var, provider_class = _PROVIDER_REGISTRY[provider_name]

    # Mock provider needs no API key
    if provider_name == "mock":
        return provider_class(family=model_config.family, model_id=model_config.model_id)

    # Real providers need an API key
    api_key = api_keys.get(env_var)
    if not api_key:
        raise ValueError(f"API key not found for {provider_name} (expected {env_var})")

    # OpenAI-compatible providers need family and base_url
    if provider_name == "openai":
        return provider_class(
            family=model_config.family,
            model_id=model_config.model_id,
            api_key=api_key,
            base_url="https://api.openai.com/v1",
        )

    # All other providers just need model_id and api_key
    return provider_class(model_id=model_config.model_id, api_key=api_key)
