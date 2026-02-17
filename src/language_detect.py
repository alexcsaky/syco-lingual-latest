"""Language detection for model responses."""

from __future__ import annotations


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "unknown"
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unknown"


def check_language_match(expected: str, detected: str) -> bool:
    """Check if detected language matches expected, handling variants.

    Handles cases like langdetect returning 'zh-cn' when we expect 'zh'.
    """
    if expected == detected:
        return True
    if detected.startswith(expected + "-"):
        return True
    return False
