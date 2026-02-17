from src.language_detect import detect_language, check_language_match


class TestDetectLanguage:
    def test_english(self):
        lang = detect_language("This is a test sentence in English.")
        assert lang == "en"

    def test_returns_string(self):
        lang = detect_language("Some text here.")
        assert isinstance(lang, str)

    def test_empty_string(self):
        lang = detect_language("")
        assert lang == "unknown"

    def test_very_short_string(self):
        # langdetect can be unreliable on very short strings
        lang = detect_language("Hi")
        assert isinstance(lang, str)


class TestCheckLanguageMatch:
    def test_match(self):
        assert check_language_match("en", "en") is True

    def test_mismatch(self):
        assert check_language_match("en", "ja") is False

    def test_chinese_variants(self):
        # langdetect returns "zh-cn" or "zh-tw", our code uses "zh"
        assert check_language_match("zh", "zh-cn") is True
        assert check_language_match("zh", "zh-tw") is True
