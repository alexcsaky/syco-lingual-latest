# SycoLingual Eval Run — Ongoing Notes
*Started 2026-02-19*

This document captures observations, decisions, and potential issues throughout the eval run. Review after completion.

---

## Methodological Notes

### "Respond briefly" vs max_tokens=1024
- Mirror and pickside prompt templates include "Respond briefly" / "Briefly describe"
- max_tokens is 1024, which is generous
- Models may interpret "briefly" differently across languages (brevity norms differ)
- This means `response_length` as a covariate is confounded with both:
  (a) the model's interpretation of "briefly" in that language
  (b) the language's inherent verbosity
- **Decision:** Accept as-is; response_length is already planned as a covariate in the mixed-effects model (Section 8.3 of methodology)

---

## Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-19 | Keep OpenRouter (not direct APIs) | Deliberate design choice despite doc saying otherwise |
| 2026-02-19 | Doc model/judge counts outdated | 7 models + 6 judges in code is correct |
| 2026-02-19 | Pickside scoring direction verified | s1_first/s2_first correctly swap user position; sum > 0 = sycophancy |

---

## Issues to Watch During Run

- [ ] Monitor judge parse failure rates per language (especially CJK)
- [ ] Check for rate limiting errors from OpenRouter
- [ ] Verify language_match rates (flag if any language < 90%)
- [ ] Watch for judge scores outside expected ranges (after validation is added)

---

## Post-Run Analysis Tasks

- [ ] Self-preference diagnostic: compare self_family=True vs False score distributions
- [ ] English validation subset: back-translate 25% of responses, re-judge in English
- [ ] Response length analysis: check if verbosity varies by language/model
- [ ] Judge agreement analysis: inter-judge correlation per facet/language

---

## Pre-Run Pipeline Fixes (2026-02-19)

### Changes Applied
1. **Schema update**: Added `fields_translated` field to `TranslatedPrompt`; added defaults to optional fields for adapter compatibility
2. **Data adapter**: `scripts/adapt_tanzim_data.py` converts TanzimSyco output → 2,800 prompts + 40 judge templates in `data/production/`
3. **Mirror stance fix**: Judge now fills `{stance1}`/`{stance2}` from `fields_translated` (target-language bare stance text) instead of `composed_prompt_en` (full English prompt)
4. **Score validation**: Out-of-range scores discarded with warning (delusion [0,5], others [-5,5])
5. **Translated system prompts**: Per-language system prompts in `config/experiment.yaml`; runner reads from config instead of hardcoded English template
6. **Persistent httpx client**: `OpenAICompatibleProvider` reuses a single `httpx.AsyncClient` across all calls; cleanup in `finally` blocks
7. **Config paths**: Updated to `data/production/prompts.jsonl` and `data/production/judges.jsonl`

### Known Limitations
- **System prompt translations**: Need native-speaker verification. Current translations via DeepL, spot-checked only.
- **`fields_translated` wording**: Target-language stances were independently translated from composed English fragments. Same meaning as `composed_prompt` stances, slightly different wording. Minor limitation, not a confound.
- **Single DeepL chain**: Within-language COMET variation is minimal; covariate captures between-language differences.
- **OpenRouter routing**: May route to slightly different model versions during long runs — `model_version` field logs actual version used.
- **Judge empty user_message**: Judge module sends empty `user_message`; monitor parse failure rates per judge.
- **Token counts**: Vary by tokenizer; character count derivable from `response_text` during analysis.
