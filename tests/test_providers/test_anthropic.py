import pytest
from src.providers.anthropic import AnthropicProvider


class TestAnthropicProviderInit:
    def test_family(self):
        p = AnthropicProvider(model_id="claude-sonnet-4-5-20250929", api_key="fake")
        assert p.family == "anthropic"

    def test_model_id(self):
        p = AnthropicProvider(model_id="claude-haiku-4-5-20251001", api_key="fake")
        assert p.model_id == "claude-haiku-4-5-20251001"
