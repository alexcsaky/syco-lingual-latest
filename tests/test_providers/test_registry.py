import os
import pytest
from src.providers import create_provider
from src.config import ModelConfig
from src.providers.base import BaseProvider
from src.mock import MockProvider
from src.providers.openai_compat import OpenAICompatibleProvider
from src.providers.anthropic import AnthropicProvider
from src.providers.google import GoogleProvider
from src.providers.xai import XAIProvider
from src.providers.moonshot import MoonshotProvider
from src.providers.deepseek import DeepSeekProvider


class TestCreateProvider:
    def test_mock_provider(self):
        config = ModelConfig(provider="mock", family="mock", model_id="mock-1.0")
        p = create_provider("test", config, {})
        assert isinstance(p, MockProvider)

    def test_openai_provider(self):
        config = ModelConfig(provider="openai", family="openai", model_id="gpt-5")
        p = create_provider("test", config, {"OPENAI_API_KEY": "fake"})
        assert isinstance(p, OpenAICompatibleProvider)
        assert p.family == "openai"

    def test_anthropic_provider(self):
        config = ModelConfig(provider="anthropic", family="anthropic", model_id="claude-sonnet-4-5")
        p = create_provider("test", config, {"ANTHROPIC_API_KEY": "fake"})
        assert isinstance(p, AnthropicProvider)

    def test_google_provider(self):
        config = ModelConfig(provider="google", family="google", model_id="gemini-3.0-flash")
        p = create_provider("test", config, {"GOOGLE_API_KEY": "fake"})
        assert isinstance(p, GoogleProvider)

    def test_xai_provider(self):
        config = ModelConfig(provider="xai", family="xai", model_id="grok-4")
        p = create_provider("test", config, {"XAI_API_KEY": "fake"})
        assert isinstance(p, XAIProvider)

    def test_moonshot_provider(self):
        config = ModelConfig(provider="moonshot", family="moonshot", model_id="kimi-2.5")
        p = create_provider("test", config, {"MOONSHOT_API_KEY": "fake"})
        assert isinstance(p, MoonshotProvider)

    def test_deepseek_provider(self):
        config = ModelConfig(provider="deepseek", family="deepseek", model_id="deepseek-3.2")
        p = create_provider("test", config, {"DEEPSEEK_API_KEY": "fake"})
        assert isinstance(p, DeepSeekProvider)

    def test_unknown_provider_raises(self):
        config = ModelConfig(provider="unknown", family="test", model_id="test")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("test", config, {})

    def test_missing_api_key_raises(self):
        config = ModelConfig(provider="openai", family="openai", model_id="gpt-5")
        with pytest.raises(ValueError, match="API key"):
            create_provider("test", config, {})
