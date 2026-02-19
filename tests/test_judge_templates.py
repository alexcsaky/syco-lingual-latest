"""Tests for judge template loader (JSONL format with placeholder filling)."""

import json
import tempfile
from pathlib import Path

import pytest

from src.judge_templates import JudgeTemplate, load_judge_templates, fill_judge_template


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIRROR_JA = {
    "judge_uid": "deepl__ja__mirror",
    "facet": "mirror",
    "lang": "ja",
    "judge_frame": "Judge prompt in Japanese with {original_prompt} and {response} and {stance1} and {stance2}",
    "judge_frame_en": "Judge prompt in English with {original_prompt} and {response} and {stance1} and {stance2}",
    "placeholders": ["original_prompt", "response", "stance1", "stance2"],
    "chain": "deepl_deepl",
}

PICKSIDE_JA = {
    "judge_uid": "deepl__ja__pickside",
    "facet": "pickside",
    "lang": "ja",
    "judge_frame": "Pickside judge {original_prompt} {response}",
    "judge_frame_en": "Pickside judge EN {original_prompt} {response}",
    "placeholders": ["original_prompt", "response"],
    "chain": "deepl_deepl",
}

MIRROR_JA_ALT_CHAIN = {
    "judge_uid": "gpt4__ja__mirror",
    "facet": "mirror",
    "lang": "ja",
    "judge_frame": "Alt chain mirror judge {original_prompt} {response}",
    "judge_frame_en": "Alt chain mirror judge EN {original_prompt} {response}",
    "placeholders": ["original_prompt", "response", "stance1", "stance2"],
    "chain": "gpt4_gpt4",
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadJudgeTemplates:
    def test_loads_templates_by_facet_lang(self, tmp_path):
        """Templates are loaded and keyed by (facet, lang) tuple."""
        jsonl_path = tmp_path / "judges.jsonl"
        _write_jsonl(jsonl_path, [MIRROR_JA, PICKSIDE_JA])

        templates = load_judge_templates(str(jsonl_path))

        assert ("mirror", "ja") in templates
        assert ("pickside", "ja") in templates
        assert len(templates) == 2

        t = templates[("mirror", "ja")]
        assert t.judge_uid == "deepl__ja__mirror"
        assert t.facet == "mirror"
        assert t.lang == "ja"
        assert t.chain == "deepl_deepl"

    def test_deduplicates_by_facet_lang(self, tmp_path):
        """When multiple chains provide the same (facet, lang), keeps the first."""
        jsonl_path = tmp_path / "judges.jsonl"
        _write_jsonl(jsonl_path, [MIRROR_JA, MIRROR_JA_ALT_CHAIN])

        templates = load_judge_templates(str(jsonl_path))

        assert len(templates) == 1
        t = templates[("mirror", "ja")]
        # First one wins
        assert t.chain == "deepl_deepl"
        assert t.judge_uid == "deepl__ja__mirror"

    def test_ignores_extra_jsonl_fields(self, tmp_path):
        """Extra fields from Tanzim's JSONL (e.g. elapsed_s) are silently ignored."""
        record_with_extras = {
            **MIRROR_JA,
            "chain_fwd_engine": "deepl",
            "placeholders_ok": 4,
            "placeholders_total": 4,
            "placeholders_missing": [],
            "elapsed_s": 0.605,
        }
        jsonl_path = tmp_path / "judges.jsonl"
        _write_jsonl(jsonl_path, [record_with_extras])

        templates = load_judge_templates(str(jsonl_path))
        assert ("mirror", "ja") in templates


class TestFillJudgeTemplate:
    def test_fills_basic_placeholders(self):
        """Fills {original_prompt} and {response} in a template."""
        template = JudgeTemplate(
            judge_uid="test__en__pickside",
            facet="pickside",
            lang="en",
            judge_frame="Prompt: {original_prompt}\nResponse: {response}",
            judge_frame_en="Prompt: {original_prompt}\nResponse: {response}",
            placeholders=["original_prompt", "response"],
            chain="test_chain",
        )

        result = fill_judge_template(
            template,
            original_prompt="Is the sky blue?",
            response="Yes, the sky is blue.",
        )

        assert result == "Prompt: Is the sky blue?\nResponse: Yes, the sky is blue."

    def test_fills_stance_placeholders(self):
        """Fills {stance1} and {stance2} for mirror facet templates."""
        template = JudgeTemplate(
            judge_uid="test__en__mirror",
            facet="mirror",
            lang="en",
            judge_frame=(
                "Prompt: {original_prompt}\n"
                "Response: {response}\n"
                "Stance1: {stance1}\n"
                "Stance2: {stance2}"
            ),
            judge_frame_en=(
                "Prompt: {original_prompt}\n"
                "Response: {response}\n"
                "Stance1: {stance1}\n"
                "Stance2: {stance2}"
            ),
            placeholders=["original_prompt", "response", "stance1", "stance2"],
            chain="test_chain",
        )

        result = fill_judge_template(
            template,
            original_prompt="Remote work is better",
            response="I agree with that view",
            stance1="Remote work is superior",
            stance2="Office work is superior",
        )

        assert "Stance1: Remote work is superior" in result
        assert "Stance2: Office work is superior" in result
        assert "Prompt: Remote work is better" in result
        assert "Response: I agree with that view" in result

    def test_missing_optional_stance_leaves_no_error(self):
        """Template without stance placeholders works fine when stances aren't provided."""
        template = JudgeTemplate(
            judge_uid="test__en__pickside",
            facet="pickside",
            lang="en",
            judge_frame="Prompt: {original_prompt}\nResponse: {response}",
            judge_frame_en="Prompt: {original_prompt}\nResponse: {response}",
            placeholders=["original_prompt", "response"],
            chain="test_chain",
        )

        # No stance arguments — should work without errors
        result = fill_judge_template(
            template,
            original_prompt="Test prompt",
            response="Test response",
        )

        assert result == "Prompt: Test prompt\nResponse: Test response"
