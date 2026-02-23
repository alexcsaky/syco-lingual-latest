# SycoLingual Poster Brief
## For Apart Research AI Manipulation Workshop, Paris

**Prepared:** 2026-02-22
**Format:** A0 Portrait (84.1 x 118.9 cm)
**Design tool:** Nano Banana Pro
**Theme:** Apart Research Dark (see colour codes below)

---

## COLOUR PALETTE (Apart Dark Theme)

All figures use this palette and will blend seamlessly with the poster background.

| Role        | Hex       | Usage                              |
|-------------|-----------|-------------------------------------|
| Background  | `#3b3f47` | Poster background, figure bg        |
| Accent      | `#5cffb1` | Key numbers, highlights, bars       |
| Text        | `#e8e8e8` | All body text, axis labels           |
| Muted       | `#9a9da3` | Subtitles, secondary info            |
| Card        | `#4a4e57` | Callout boxes, legend backgrounds    |
| Warm yellow | `#fee08b` | Mild severity, medium resource       |
| Orange      | `#f46d43` | Moderate severity, low resource, regression lines |
| Dark red    | `#a50026` | Strong severity                      |

---

## POSTER TITLE

**SycoLingual: Cross-Linguistic Sycophancy in Frontier LLMs**

Subtitle: *Do language models sycophant equally across languages? A 7-model, 10-language, 4-facet benchmark.*

Authors: [Your names here]
Affiliation: Apart Research

---

## POSTER STRUCTURE (top to bottom)

```
+--------------------------------------------------+
|  [Apart Logo]          TITLE          [QR Code?]  |
|  Subtitle + Authors                               |
+--------------------------------------------------+
|                                                    |
|  SECTION 1: BACKGROUND & MOTIVATION               |
|  (Text block, ~4-5 sentences)                      |
|                                                    |
+--------------------------------------------------+
|                                                    |
|  SECTION 2: PIPELINE AT A GLANCE                   |
|  (Flowchart or visual summary)                     |
|                                                    |
+--------------------------------------------------+
|                                                    |
|  SECTION 3: FIGURE 1 — HERO HEATMAP               |
|  (fig6_poster.png, full width)                     |
|                                                    |
|  KEY TAKEAWAY 1 (callout box)                      |
|                                                    |
+--------------------------------------------------+
|                                                    |
|  SECTION 4: TWO SUPPORTING FIGURES SIDE BY SIDE    |
|  [fig7_poster.png]     [fig9_mirror_poster.png]    |
|                                                    |
|  KEY TAKEAWAY 2 (callout box)                      |
|                                                    |
+--------------------------------------------------+
|                                                    |
|  SECTION 5: METHODOLOGY HIGHLIGHTS                 |
|  (Bullet points)                                   |
|                                                    |
+--------------------------------------------------+
|                                                    |
|  SECTION 6: REFERENCES / CONTACT / QR              |
|                                                    |
+--------------------------------------------------+
```

---

## SECTION 1: BACKGROUND & MOTIVATION

Suggested text (~80 words):

> Sycophancy — a model's tendency to align with user preferences at the cost of accuracy — is a known alignment failure in large language models. However, most evaluations test only English. We ask: **does sycophancy transfer uniformly across languages, or do models behave differently depending on the language of interaction?**
>
> SycoLingual is a cross-linguistic sycophancy benchmark spanning **7 frontier models, 10 typologically diverse languages, and 4 distinct sycophancy facets**, evaluated by a **6-judge cross-family panel**.

---

## SECTION 2: PIPELINE AT A GLANCE

Design this as a horizontal flowchart or numbered pipeline:

```
160 items        10 languages       7 models        6 judges
(4 facets)  -->  (DeepL + QC)  -->  (T=0)    -->   (median)   -->  11,200 scores
 40 per facet     COMET > 0.87      19,600          117,600
                  chrF++ QC         responses        judgements
```

**Key numbers to highlight in the pipeline visual:**

| Metric | Value |
|--------|-------|
| Unique benchmark items | 160 (40 per facet) |
| Languages tested | 10 (6 high, 1 medium, 3 low resource) |
| Models evaluated | 7 (from 7 different provider families) |
| Total API responses | 19,600 |
| Judge models | 6 (cross-family, smaller-tier) |
| Total judgements | 117,600 |
| Derived sycophancy scores | 11,200 |
| Temperature | 0 (fully deterministic) |

---

## SECTION 3: HERO FIGURE

**File:** `figures/fig6_poster.png` (564 KB, 300 DPI)
**Title on figure:** "Moderate+ Sycophancy Rate by Language & Model"

This heatmap shows the % of items scoring >= 0.30 (Moderate+) on the 0-1 normalized sycophancy scale, for each language-model combination. Marginal bars show row (language) and column (model) averages. Green percentages above each column show model averages.

**Reading guide for viewers:**
- Darker red = higher sycophancy rate
- Claude Sonnet 4.6 column is mostly pale (2-11% range)
- Mistral Large column is deep red (35-45% range)
- Languages cluster tightly (24-33% means) — models vary much more

---

## KEY TAKEAWAY 1

**Callout box (large text, accent green):**

> **Sycophancy is a property of models, not languages.**
> Model identity explains **15x more variance** than language (eta-squared = 0.071 vs 0.005).
> Mistral Large shows **7x** the Moderate+ sycophancy rate of Claude Sonnet 4.6 (40% vs 6%) — consistent across all 10 languages.

---

## SECTION 4: SUPPORTING FIGURES

### Left: Severity Distribution
**File:** `figures/fig7_poster.png` (272 KB, 300 DPI)
**Title on figure:** "Sycophancy Severity Distribution by Model"

Stacked horizontal bars showing the full None/Mild/Moderate/Strong breakdown per model. Models sorted by total sycophancy (Mistral at top, Claude at bottom). Legend on the right side.

**Key stats visible in figure:**
- Claude: 73% None, only 1% Strong
- Mistral: 38% None, 26% Strong
- Grok: 48% None but 28% Strong (highest Strong %)

### Right: Mirror by Language
**File:** `figures/fig9_mirror_poster.png` (246 KB, 300 DPI)
**Title on figure:** "Opinion Mirroring: The Facet Where Language Matters"

Horizontal bar chart showing the Moderate+ sycophancy rate for the Mirror facet only, sorted by rate. Bars coloured by resource level (green = high, yellow = medium, orange = low).

**Key stats visible in figure:**
- Bengali: 50% (highest, low-resource)
- Japanese: 22% (lowest, high-resource)
- 2.3x difference — the strongest cross-linguistic signal in the dataset
- 2 of top 3 are low-resource languages

---

## KEY TAKEAWAY 2

**Callout box (large text, accent green):**

> **But language does matter for opinion mirroring.**
> Bengali speakers experience **2.3x more opinion-mirroring** sycophancy than Japanese speakers (50% vs 22%).
> Mirror is the only facet with a significant cross-linguistic effect (eta-squared = 0.039). Low-resource languages show elevated rates.

---

## SECTION 5: METHODOLOGY HIGHLIGHTS

Bullet points for the poster:

**Benchmark Design**
- 4 sycophancy facets: Opinion Mirroring, Side-Taking, Attribution Bias, Delusion Acceptance
- All prompts designed for cross-linguistic fairness: no idioms, no culture-specific references, no Western-centric framing
- 22/40 Mirror items rewritten from Syco-bench v1 to eliminate formulaic patterns

**Translation Quality**
- DeepL API with 3-layer QC: automated sanity checks + chrF++ (lexical) + COMET-22 (neural)
- All 9 target languages passed QC (COMET 0.87-0.91, chrF++ 0.61-0.77)
- Neither metric correlated significantly with sycophancy, ruling out translation quality as a confound
- Multi-engine cross-validation (DeepL x Google, 4 pairings): COMET spread only 0.005

**Evaluation**
- 7 models from 7 provider families, including 2 open-weight and 2 Chinese-developed
- 6-judge cross-family panel using smaller-tier models (avoids subject-judge overlap)
- Median aggregation, minimum 3 valid scores required
- Temperature = 0 for full reproducibility

**Severity Thresholds** (on 0-1 normalized scale)
- None: < 0.10 (below measurement noise floor)
- Mild: 0.10 to < 0.30 (detectable but small)
- Moderate: 0.30 to < 0.50 (clearly sycophantic)
- Strong: >= 0.50 (half or more of maximum)

---

## SECTION 6: KEY STATISTICS TABLE (optional, if space permits)

| | Value |
|---|---|
| Overall Moderate+ rate | 27.9% |
| Items showing ANY sycophancy (>= 0.10) | 50.5% |
| Highest-scoring facet | Delusion (mean 0.357) |
| Lowest-scoring facet | Attribution Bias (mean 0.062) |
| Most sycophantic model | Mistral Large (40.4% Moderate+) |
| Least sycophantic model | Claude Sonnet 4.6 (5.8% Moderate+) |
| Most sycophantic language (Mirror) | Bengali (50% Moderate+) |
| Least sycophantic language (Mirror) | Japanese (22% Moderate+) |
| Model vs language variance ratio | 14.6x (eta-squared 0.071 vs 0.005) |

---

## MODELS TESTED (use full names on poster)

| Provider | Model Name |
|----------|------------|
| Anthropic | Claude Sonnet 4.6 |
| OpenAI | GPT-5.1 |
| Google | Gemini 3 Flash |
| xAI | Grok 4.1 |
| DeepSeek | DeepSeek v3.2 |
| Moonshot | Kimi K2.5 |
| Mistral | Mistral Large |

---

## LANGUAGES TESTED

| Language | Code | Resource Level |
|----------|------|----------------|
| English | en | High |
| Japanese | ja | High |
| German | de | High |
| Spanish | es | High |
| French | fr | High |
| Chinese | zh | High |
| Arabic | ar | Medium |
| Bengali | bn | Low |
| Slovenian | sl | Low |
| Latvian | lv | Low |

---

## FIGURE FILES (all in `analysis/figures/`)

| File | Size | Description | Use |
|------|------|-------------|-----|
| `fig6_poster.png` | 564 KB | Rate heatmap (language x model) | HERO - full width |
| `fig7_poster.png` | 272 KB | Severity stacked bars | Supporting left |
| `fig9_mirror_poster.png` | 246 KB | Mirror by language bars | Supporting right |

All figures: 300 DPI, PNG, dark background (#3b3f47), ready for direct placement.

---

## FOUR FACET DEFINITIONS (if space for a small sidebar)

1. **Opinion Mirroring** — Does the model flip its stance to match the user's opinion? Measured by scoring opposing-stance prompts and taking the difference.
2. **Side-Taking** — Does the model favour the user in a disagreement? Measured by swapping who holds which position.
3. **Attribution Bias** — Does the model agree more when a claim is "mine" vs "some people's"? Measured by comparing user-attributed vs generic-attributed prompts.
4. **Delusion Acceptance** — Does the model go along with factually false premises? Measured by presenting objectively false claims as user beliefs.

---

## REFERENCES (suggested)

- Duffy (2025). Syco-bench: A Benchmark for Measuring Sycophancy in LLMs.
- Rei et al. (2022). COMET-22: Unbabel-IST 2022 Submission for WMT Metrics.
- Popovic (2017). chrF++: words helping character n-grams.
- Apart Research, AI Manipulation Workshop 2026.

---

## NOTES FOR THE DESIGNER

1. **Background colour** must be `#3b3f47` — the figures have this exact background and will blend in seamlessly.
2. **All text** should be `#e8e8e8` (light grey/white).
3. **Accent colour** for key numbers and callout boxes: `#5cffb1` (bright green).
4. **The hero heatmap (fig6)** should span full poster width. The two supporting figures can sit side by side at ~50% width each.
5. **Key takeaway boxes** should be visually prominent — consider a slightly lighter card background (`#4a4e57`) with the accent green for the key numbers.
6. **Font:** sans-serif throughout. The figures use the system default sans-serif at 14pt base.
7. Figures are at 300 DPI and should print cleanly at A0 scale.
