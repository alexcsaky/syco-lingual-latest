import pytest
from src.providers.openai_compat import OpenAICompatibleProvider
from src.providers.xai import XAIProvider
from src.providers.moonshot import MoonshotProvider
from src.providers.deepseek import DeepSeekProvider


class TestOpenAICompatibleProviderInit:
    def test_openai_base_url(self):
        p = OpenAICompatibleProvider(
            family="openai", model_id="gpt-5", api_key="fake",
            base_url="https://api.openai.com/v1",
        )
        assert p.family == "openai"

    def test_xai_subclass(self):
        p = XAIProvider(model_id="grok-4", api_key="fake")
        assert p.family == "xai"

    def test_moonshot_subclass(self):
        p = MoonshotProvider(model_id="kimi-2.5", api_key="fake")
        assert p.family == "moonshot"

    def test_deepseek_subclass(self):
        p = DeepSeekProvider(model_id="deepseek-chat", api_key="fake")
        assert p.family == "deepseek"
