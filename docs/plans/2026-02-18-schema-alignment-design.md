# Schema Alignment with Translation Pipeline — Design Document

**Date:** 2026-02-18
**Context:** Tanzim delivered test data in `data/TESTFOLDER/multi_benchmark/` with 252 prompts, 144 judge templates, and quality metrics. The schema doesn't match our pipeline. This document captures the design decisions and changes needed.

## Key Decisions

1. **Translation chains**: Pick best chain per language using COMET score (1x cost)
2. **Facet names**: Adopt Tanzim's names — `mirror`, `pickside`, `whosaid`, `delusion`
3. **Run field**: Adopt Tanzim's `run` field (replaces `variant`) — `stance1`, `stance2`, `s1_first`, `s2_first`, `user_attributed`, `generic_attributed`, `single`
4. **Field names**: Adopt Tanzim's field names throughout (`prompt_uid`, `lang`, `composed_prompt`, etc.)
5. **Judge templates**: Load from `judges.jsonl`, fill placeholders at runtime
6. **Judge output**: Use plain `complete()` (raw integer), NOT `complete_structured()` — matches Tanzim's template design
7. **Sanity check failures**: Include all prompts, preserve sanity_checks field
8. **Mirror stance extraction**: Pair stance1/stance2 prompts to fill judge template placeholders

## Schema Changes

### TranslatedPrompt (input)

```
OLD → NEW
prompt_id: str           → prompt_uid: str
item_id: str             → item_id: int
facet: Literal[4 values] → facet: str  ("mirror", "pickside", "whosaid", "delusion")
variant: Literal[a,b,none] → run: str  ("stance1", "stance2", ...)
language: str            → lang: str
english_text: str        → composed_prompt_en: str
translated_text: str     → composed_prompt: str
comet_score: float|None  → comet_score: float|None
(new fields)             → back_translation_en, chrf_score, word_overlap,
                           sanity_checks, domain_tag, difficulty_tag,
                           severity_tag, delusion_type, chain,
                           fwd_engine, back_engine, fwd_time_s, back_time_s
```

### ModelResponse (output from evaluation)

Rename fields to match: `prompt_uid`, `lang`, `run` (replaces `variant`). Add `chain: str`.

### JudgeScore (output from judging)

Rename fields to match: `prompt_uid`, `lang`. Drop `justification` (make optional with default ""). Add `chain: str`.

### ScoredItem (aggregation output)

Rename fields to match: `prompt_uid`, `lang`. Add `chain: str`.

### FACET_SCORE_RANGES

```python
FACET_SCORE_RANGES = {
    "mirror": (-5, 5),
    "pickside": (-5, 5),
    "whosaid": (-5, 5),
    "delusion": (0, 5),
}
```

## Judge Template Architecture

### Current (broken)
- Load plain text from `config/judge_prompts/{facet}_{language}.txt`
- Send as system prompt
- Use `complete_structured()` for JSON output

### New
- Load `judges.jsonl` into dict keyed by `(facet, lang)`
- Fill placeholders at runtime: `{original_prompt}`, `{response}`, `{stance1}`, `{stance2}`
- Send filled template as system prompt
- Use plain `complete()` — template already instructs "output only a single integer"
- Parse raw integer from response text

### Mirror Stance Pairing
The mirror judge template needs `{stance1}` and `{stance2}` — the two opposing positions.
These are extracted by pairing prompts with the same `(item_id, lang, chain)` across `stance1` and `stance2` runs.
The `composed_prompt_en` from each stance provides the stance text.

## Chain Selection

New preprocessing module `src/chain_selector.py`:
1. Group prompts by `(item_id, facet, run, lang)`
2. For each group, pick the chain with highest COMET score
3. Return filtered list with best-chain prompts only

## Config Changes

- `PathsConfig.judge_prompts_dir` → `PathsConfig.judge_templates` (path to judges.jsonl)
- Update `experiment.yaml` to point to test data paths

## Mock Provider Changes

Mock `complete()` needs to return a raw integer string (for judge simulation), not JSON.
Add a `complete_judge()` or change how the judge module uses mock — simplest: mock returns "3" as text, judge parses it.
Alternative: keep mock returning prose, judge falls back gracefully in tests.

Actually simplest: update MockProvider.complete_structured() is not used for judges anymore.
The judge now uses complete(). MockProvider.complete() returns prose text.
For test purposes, update mock to optionally return a raw integer when the system prompt contains judge-like instructions, OR just make the judge parsing robust enough to handle mock responses (return score=0 on parse failure in tests).
