# SycoLingual Codebase Analysis — Pre-Run Audit
*Generated 2026-02-19*

## System Architecture

The two codebases implement a split pipeline:
- **TanzimSyco** handles steps A→C (dataset creation, enhancement, translation with quality scoring), producing translated prompts and judge templates.
- **syco-lingual** handles steps D→G (model evaluation, judging, scoring), consuming those outputs.

### TanzimSyco (Steps A→C): Translation Pipeline

**What it does:** Takes 160 English sycophancy evaluation items across 4 facets (mirror, pickside, whosaid, delusion — 40 each), composes them into 280 prompts using templates, then translates everything into 9 target languages (+ English baseline = 10 total).

**Data flow:**
```
Raw CSVs → Enhanced CSVs → Composed JSONL (280 prompts) → Translation Pipeline
    ├─ DeepL forward translation (with XML placeholder protection)
    ├─ DeepL back-translation
    ├─ Sanity checks (language detection, length ratio, chrF++)
    ├─ COMET-22 neural quality scoring
    └─ Final output: 2,800 translated prompts + 40 judge templates
```

**Quality results:** All 9 target languages exceed 0.87 COMET mean. Bengali surprisingly has the *highest* COMET (0.908). Zero genuine quality failures. 280 Chinese false positives in sanity checks (language code format mismatch — cosmetic only).

### syco-lingual (Steps D→G): Evaluation Pipeline

**What it does:** Takes translated prompts, sends them to 7 models via OpenRouter, collects responses, then has a 6-judge panel score each response for sycophancy, aggregating via median.

**Data flow:**
```
Translated Prompts → Chain Selection (best COMET per language) → Evaluation Runner
    ├─ 7 models × 2,800 prompts = 19,600 API calls
    └─ Responses → 6-Judge Panel
        ├─ 19,600 × 6 judges = 117,600 judge calls
        └─ Aggregate: median of ≥3 judges → ScoredItem
```

---

## Critical Issues Identified

### 1. MAJOR: OpenRouter vs Direct APIs

The methodology document (Section 4.3) **explicitly states**: "Decision: Use direct provider APIs rather than OpenRouter" and "OpenRouter injects its own system prompts by default, which was an uncontrolled confound in v1."

But the code (`experiment.yaml`) routes **every single model and judge through OpenRouter**.

This is the exact confound v2 was designed to eliminate.

---

### 2. MAJOR: Schema Mismatch Between TanzimSyco Output and syco-lingual Input

The syco-lingual `TranslatedPrompt` schema expects these fields that **do not exist** in TanzimSyco's output:

| Missing Field | Type | Purpose |
|---|---|---|
| `prompt_uid` | str | **Critical** — used as primary key for resumability and judge aggregation |
| `chain` | str | Translation chain ID (e.g., "deepl_deepl") |
| `chrf_score` | float | Top-level chrF score (exists inside `sanity_checks` dict but not at top level) |
| `word_overlap` | float | Not computed by TanzimSyco |
| `severity_tag` | str | Not in TanzimSyco output |
| `delusion_type` | str | Not in TanzimSyco output |
| `fwd_engine` | str | Forward translation engine |
| `back_engine` | str | Back-translation engine |
| `fwd_time_s` | float | Forward translation time |
| `back_time_s` | float | Back-translation time |

The `prompt_uid` is especially critical — the entire eval pipeline uses it as the primary deduplication key.

**Additionally**, TanzimSyco outputs judge templates as one JSON file per language, but syco-lingual expects a single JSONL file with `JudgeTemplate` records.

The config currently points to `data/TESTFOLDER/multi_benchmark/` (test data), not TanzimSyco output.

---

### 3. MAJOR: Model & Judge Count Discrepancies

| What | Doc Says | Code Has |
|---|---|---|
| Evaluated models | 6 | **7** (added Mistral Large) |
| Judge models | 5 | **6** (added Mistral Small) |
| Claude version | Sonnet 4.5 | Sonnet **4.6** |
| GPT version | GPT-5 | GPT-**5.1** |
| Grok version | Grok 4 | Grok **4.1** |

---

### 4. SIGNIFICANT: Judge Score Validation Missing

`FACET_SCORE_RANGES` defined in schemas but **never validates** judge scores. Out-of-range scores silently accepted, corrupting median calculation.

---

### 5. SIGNIFICANT: Judge Template Missing Doc's Specified Elements

Doc (Section 7.3) specifies judges should receive:
1. System prompt instructing impartial evaluation
2. Scoring rubric
3. Instructions for JSON output with `score` (integer) and `justification` (one sentence)

Current implementation:
- Sends entire filled judge template as system prompt with empty user message
- Requests only raw integer (no justification captured)
- No separate judge system prompt for impartiality instruction
- Missing: "Base your score solely on the rubric criteria..."

---

### 6. SIGNIFICANT: English Validation Subset Not Fully Implemented

Code has `select_validation_subset()`, `BackTranslator` class, and `--english-validation` CLI flag. But the actual flow of back-translating responses → judging them in English → comparing with target-language scores isn't wired together.

---

### 7. METHODOLOGICAL: Self-Preference Diagnostic Not Implemented

Doc specifies (Section 7.2): comparing judge scores on own family's evaluated model vs others. `self_family` flag IS computed and stored, but no analysis code exists to run the diagnostic.

---

### 8. METHODOLOGICAL: Mirror Facet Stance Lookup in Judging

Judge module builds stance lookup using `composed_prompt_en` (English). But mirror judge template `{stance1}` and `{stance2}` placeholders get filled with **English** stance text even when judging target-language responses. Should these be in target language?

---

### 9. METHODOLOGICAL: Pickside Scoring Direction Ambiguity

Pickside judge template scores "how much the model agrees with the first statement" on -5 to +5. Depending on the `run` (s1_first vs s2_first), "the first statement" is either the user's position or the friend's position. Need to verify template scoring direction correctly maps to the paired scoring formula.

---

### 10. METHODOLOGICAL: "Respond briefly" vs max_tokens=1024

Mirror and pickside templates include "Respond briefly", but max_tokens is 1024. Models may interpret "briefly" differently across languages, confounding `response_length` as a covariate.

---

### 11. TECHNICAL: httpx Client Created Per Request

In `openai_compat.py`, a new `httpx.AsyncClient` created for every API call. At 137,200 total calls, this creates/tears down 137,200 HTTP connection pools. A persistent client should be shared.

---

### 12. TECHNICAL: No Rate Limiting Beyond Semaphores

Semaphores bound concurrency but don't respect actual rate limits. With 7 models and 6 judges all hitting OpenRouter, rate limit errors likely.

---

## Questions (Pending User Answers)

1. **OpenRouter vs Direct APIs:** Intentional, or needs changing?
2. **Schema adapter:** Build transformation from TanzimSyco → syco-lingual format?
3. **Mistral addition:** Confirmed intentional?
4. **Model versions:** Doc just outdated?
5. **Judge justification:** Add structured output, or raw integer sufficient?
6. **English validation subset:** Implement the back-translation re-judging flow?
7. **Budget/timeline:** Cost ceiling for ~137K API calls?
8. **Prior test runs:** Have dry_run.py or live_test.py been executed?
9. **Mirror stance language:** English vs target language for judging?
10. **Score validation:** Add bounds checking before aggregation?
