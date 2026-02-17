"""Back-translation via DeepL API for the English validation subset."""

from __future__ import annotations

import httpx


class BackTranslator:
    DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"

    def __init__(self, api_key: str | None, mock: bool = False):
        self._api_key = api_key
        self._mock = mock

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        if source_lang == target_lang:
            return text

        if self._mock:
            return f"[BACK-TRANSLATED from {source_lang}] {text[:50]}..."

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.DEEPL_API_URL,
                data={
                    "auth_key": self._api_key,
                    "text": text,
                    "source_lang": source_lang.upper(),
                    "target_lang": target_lang.upper(),
                },
            )
            response.raise_for_status()
            return response.json()["translations"][0]["text"]
