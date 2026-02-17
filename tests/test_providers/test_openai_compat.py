import pytest
from src.providers.openai_compat import OpenAICompatibleProvider


class TestOpenAICompatibleProviderInit:
    def test_default_base_url(self):
        p = OpenAICompatibleProvider(
            family="openai", model_id="openai/gpt-5.1", api_key="fake",
        )
        assert p._base_url == "https://api.openai.com/v1"

    def test_custom_base_url(self):
        p = OpenAICompatibleProvider(
            family="openai", model_id="openai/gpt-5.1", api_key="fake",
            base_url="https://openrouter.ai/api/v1",
        )
        assert p._base_url == "https://openrouter.ai/api/v1"

    def test_openrouter_headers_included(self):
        p = OpenAICompatibleProvider(
            family="openai", model_id="openai/gpt-5.1", api_key="fake",
            base_url="https://openrouter.ai/api/v1",
        )
        headers = p._build_headers()
        assert headers["Authorization"] == "Bearer fake"
        assert "HTTP-Referer" in headers
        assert "X-Title" in headers

    def test_non_openrouter_no_extra_headers(self):
        p = OpenAICompatibleProvider(
            family="openai", model_id="gpt-5", api_key="fake",
            base_url="https://api.openai.com/v1",
        )
        headers = p._build_headers()
        assert "HTTP-Referer" not in headers
        assert "X-Title" not in headers
