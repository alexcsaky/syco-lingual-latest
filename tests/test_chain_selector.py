"""Tests for the chain selection module.

Verifies that select_best_chains() correctly picks the best translation
chain per language based on average COMET score.
"""

import pytest

from src.chain_selector import select_best_chains
from src.schemas import TranslatedPrompt


def _make_prompt(
    *,
    lang: str = "ja",
    chain: str = "deepl_deepl",
    facet: str = "mirror",
    run: str = "stance1",
    item_id: int = 1,
    comet_score: float | None = 0.90,
) -> TranslatedPrompt:
    """Helper to build a TranslatedPrompt with sensible defaults."""
    return TranslatedPrompt(
        prompt_uid=f"{chain}__{lang}__{facet}__{item_id}__{run}",
        item_id=item_id,
        facet=facet,
        run=run,
        lang=lang,
        composed_prompt="translated text",
        composed_prompt_en="original english",
        back_translation_en="back translated",
        comet_score=comet_score,
        chrf_score=0.70,
        word_overlap=0.50,
        sanity_checks={"passed": True},
        domain_tag="work/career",
        difficulty_tag="moderate",
        severity_tag="",
        delusion_type="",
        chain=chain,
        fwd_engine=chain.split("_")[0],
        back_engine=chain.split("_")[1],
        fwd_time_s=0.5,
        back_time_s=0.4,
    )


class TestSelectBestChains:
    def test_picks_highest_comet(self):
        """Two prompts with different chains, same (lang, facet, run).

        The chain with higher COMET should win.
        """
        low = _make_prompt(chain="deepl_deepl", comet_score=0.80)
        high = _make_prompt(chain="google_google", comet_score=0.95)

        result = select_best_chains([low, high])

        assert len(result) == 1
        assert result[0].chain == "google_google"

    def test_preserves_all_runs_from_best_chain(self):
        """4 prompts (2 chains x 2 runs). All prompts from the best chain
        should be preserved, not just the single best one.
        """
        prompts = [
            _make_prompt(chain="deepl_deepl", run="stance1", comet_score=0.90),
            _make_prompt(chain="deepl_deepl", run="stance2", comet_score=0.88),
            _make_prompt(chain="google_google", run="stance1", comet_score=0.70),
            _make_prompt(chain="google_google", run="stance2", comet_score=0.72),
        ]

        result = select_best_chains(prompts)

        assert len(result) == 2
        assert all(p.chain == "deepl_deepl" for p in result)
        runs = {p.run for p in result}
        assert runs == {"stance1", "stance2"}

    def test_different_best_chain_per_language(self):
        """Different languages can have different best chains."""
        prompts = [
            # Japanese: google_google is better
            _make_prompt(lang="ja", chain="deepl_deepl", comet_score=0.70),
            _make_prompt(lang="ja", chain="google_google", comet_score=0.95),
            # French: deepl_deepl is better
            _make_prompt(lang="fr", chain="deepl_deepl", comet_score=0.92),
            _make_prompt(lang="fr", chain="google_google", comet_score=0.80),
        ]

        result = select_best_chains(prompts)

        assert len(result) == 2
        by_lang = {p.lang: p.chain for p in result}
        assert by_lang["ja"] == "google_google"
        assert by_lang["fr"] == "deepl_deepl"

    def test_handles_none_comet(self):
        """None COMET scores should be treated as 0.0 and not win
        over chains with actual scores.
        """
        prompts = [
            _make_prompt(chain="deepl_deepl", comet_score=None),
            _make_prompt(chain="google_google", comet_score=0.50),
        ]

        result = select_best_chains(prompts)

        assert len(result) == 1
        assert result[0].chain == "google_google"
