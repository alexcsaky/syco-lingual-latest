import pytest
from src.providers.google import GoogleProvider


class TestGoogleProviderInit:
    def test_family(self):
        p = GoogleProvider(model_id="gemini-3.0-flash", api_key="fake")
        assert p.family == "google"

    def test_model_id(self):
        p = GoogleProvider(model_id="gemini-2.0-flash", api_key="fake")
        assert p.model_id == "gemini-2.0-flash"
