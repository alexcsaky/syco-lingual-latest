"""Chain selection module for the SycoLingual v2 pipeline.

Selects the best translation chain per language by comparing average COMET
scores across all prompts for each (language, chain) group. This avoids
quadrupling evaluation cost by only keeping prompts from the winning chain.
"""

from __future__ import annotations

from collections import defaultdict

from src.schemas import TranslatedPrompt


def select_best_chains(prompts: list[TranslatedPrompt]) -> list[TranslatedPrompt]:
    """Pick the best translation chain per language using average COMET score.

    Algorithm:
        1. Group prompts by (lang, chain) and compute the average COMET score
           for each group. ``None`` COMET scores are treated as 0.0.
        2. For each language, select the chain with the highest average COMET.
        3. Return all prompts belonging to the winning chains.

    Args:
        prompts: List of translated prompts (potentially from multiple chains
            and languages).

    Returns:
        Filtered list containing only prompts from the best chain per language.
    """
    # --- Step 1: Compute average COMET per (lang, chain) ---
    comet_sums: dict[tuple[str, str], float] = defaultdict(float)
    comet_counts: dict[tuple[str, str], int] = defaultdict(int)

    for p in prompts:
        key = (p.lang, p.chain)
        comet_sums[key] += p.comet_score if p.comet_score is not None else 0.0
        comet_counts[key] += 1

    avg_comet: dict[tuple[str, str], float] = {
        key: comet_sums[key] / comet_counts[key] for key in comet_sums
    }

    # --- Step 2: Find the best chain per language ---
    best_chain_per_lang: dict[str, str] = {}
    best_score_per_lang: dict[str, float] = {}

    for (lang, chain), score in avg_comet.items():
        if lang not in best_score_per_lang or score > best_score_per_lang[lang]:
            best_chain_per_lang[lang] = chain
            best_score_per_lang[lang] = score

    # --- Step 3: Filter prompts to only winning chains ---
    return [p for p in prompts if best_chain_per_lang[p.lang] == p.chain]
