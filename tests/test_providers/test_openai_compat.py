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


class TestPromptCaching:
    def test_openrouter_messages_have_cache_control(self):
        """OpenRouter messages use content array with cache_control on system prompt."""
        p = OpenAICompatibleProvider(
            family="anthropic", model_id="anthropic/claude-haiku-4-5", api_key="fake",
            base_url="https://openrouter.ai/api/v1",
        )
        messages = p._build_messages("You are a judge.", "Score this response.")
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        # Content should be array format with cache_control
        assert isinstance(system_msg["content"], list)
        assert system_msg["content"][0]["type"] == "text"
        assert system_msg["content"][0]["text"] == "You are a judge."
        assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        # User message should be plain string
        assert messages[1] == {"role": "user", "content": "Score this response."}

    def test_non_openrouter_messages_plain_format(self):
        """Non-OpenRouter messages use plain string content (no cache_control)."""
        p = OpenAICompatibleProvider(
            family="openai", model_id="gpt-5", api_key="fake",
            base_url="https://api.openai.com/v1",
        )
        messages = p._build_messages("You are helpful.", "Hello.")
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hello."}
