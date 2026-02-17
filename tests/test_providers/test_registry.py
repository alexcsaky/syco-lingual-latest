import pytest
from src.providers import create_provider
from src.config import ModelConfig
from src.providers.base import BaseProvider
from src.mock import MockProvider
from src.providers.openai_compat import OpenAICompatibleProvider


class TestCreateProvider:
    def test_mock_provider(self):
        config = ModelConfig(provider="mock", family="mock", model_id="mock-1.0")
        p = create_provider("test", config, {})
        assert isinstance(p, MockProvider)

    def test_openrouter_provider(self):
        config = ModelConfig(
            provider="openrouter",
            family="openai",
            model_id="openai/gpt-5.1",
        )
        p = create_provider("test", config, {"OPENROUTER_API_KEY": "fake-key"})
        assert isinstance(p, OpenAICompatibleProvider)
        assert p.family == "openai"
        assert p.model_id == "openai/gpt-5.1"
        assert p._base_url == "https://openrouter.ai/api/v1"

    def test_openrouter_passes_family_from_config(self):
        config = ModelConfig(
            provider="openrouter",
            family="mistral",
            model_id="mistralai/mistral-large-2512",
        )
        p = create_provider("test", config, {"OPENROUTER_API_KEY": "fake-key"})
        assert p.family == "mistral"

    def test_unknown_provider_raises(self):
        config = ModelConfig(provider="unknown", family="test", model_id="test")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("test", config, {})

    def test_missing_api_key_raises(self):
        config = ModelConfig(
            provider="openrouter",
            family="openai",
            model_id="openai/gpt-5.1",
        )
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            create_provider("test", config, {})
