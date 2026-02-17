# tests/test_translation.py
import pytest
from src.translation import BackTranslator


class TestBackTranslator:
    @pytest.mark.asyncio
    async def test_mock_translation(self):
        translator = BackTranslator(api_key=None, mock=True)
        result = await translator.translate("テストテキスト", source_lang="ja", target_lang="en")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_english_passthrough(self):
        translator = BackTranslator(api_key=None, mock=True)
        result = await translator.translate("Already English", source_lang="en", target_lang="en")
        assert result == "Already English"
