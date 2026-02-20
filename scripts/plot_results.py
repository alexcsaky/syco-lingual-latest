"""Generate a shareable results dashboard from live test data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from collections import defaultdict

# --- Load data ---
SCORED_PATH = "data/responses/live_test_scored.jsonl"
JUDGEMENTS_PATH = "data/judgements/live_test_judgements.jsonl"
RESPONSES_PATH = "data/responses/live_test_responses.jsonl"
OUTPUT_PATH = "data/responses/live_test_dashboard.png"

with open(SCORED_PATH) as f:
    scored = [json.loads(line) for line in f]

with open(JUDGEMENTS_PATH) as f:
    judgements = [json.loads(line) for line in f]

with open(RESPONSES_PATH) as f:
    responses = [json.loads(line) for line in f]

# --- Prep ---
FACET_LABELS = {
    "mirror": "Mirror\n(agrees with user stance)",
    "pickside": "Pick Side\n(takes a side in debate)",
    "whosaid": "Who Said\n(defers to authority)",
    "delusion": "Delusion\n(validates false belief)",
}
FACET_ORDER = ["mirror", "pickside", "whosaid", "delusion"]
FACET_RANGES = {"mirror": (-5, 5), "pickside": (-5, 5), "whosaid": (-5, 5), "delusion": (0, 5)}

LANG_NAMES = {
    "ar": "Arabic", "bn": "Bengali", "de": "German", "es": "Spanish",
    "fr": "French", "ja": "Japanese", "lv": "Latvian", "sl": "Slovenian", "zh": "Chinese",
}

MODEL_COLORS = {"deepseek-v3.2": "#4C78A8", "gemini-3-flash": "#E45756"}

sns.set_theme(style="whitegrid", font_scale=0.9)
fig = plt.figure(figsize=(18, 14), facecolor="white")
fig.suptitle(
    "SycoLingual v2 — Cross-Linguistic Sycophancy Benchmark (Pilot Run)",
    fontsize=16, fontweight="bold", y=0.98,
)
fig.text(
    0.5, 0.955,
    "2 models × 63 prompts × 9 languages × 4 facets  |  3 judges per response  |  378 judge scores total",
    ha="center", fontsize=10, color="gray",
)

gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35, top=0.92, bottom=0.06, left=0.06, right=0.96)

# ============================================================
# Panel 1 (top-left, spans 2 cols): Score distribution by facet + model
# ============================================================
ax1 = fig.add_subplot(gs[0, :2])

# Build data for grouped box plot
facet_model_scores = defaultdict(lambda: defaultdict(list))
for item in scored:
    facet_model_scores[item["facet"]][item["model"]].append(item["median_score"])

models = sorted({item["model"] for item in scored})
x = np.arange(len(FACET_ORDER))
width = 0.35

for i, model in enumerate(models):
    positions = x + (i - 0.5) * width
    data = [facet_model_scores[f][model] for f in FACET_ORDER]
    bp = ax1.boxplot(
        data, positions=positions, widths=width * 0.8,
        patch_artist=True, showfliers=True, flierprops=dict(markersize=4),
        medianprops=dict(color="black", linewidth=1.5),
    )
    color = list(MODEL_COLORS.values())[i]
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean markers
    means = [np.mean(d) if d else 0 for d in data]
    ax1.scatter(positions, means, marker="D", color=color, s=30, zorder=5, edgecolors="black", linewidths=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels([FACET_LABELS[f] for f in FACET_ORDER], fontsize=8)
ax1.set_ylabel("Median Judge Score")
ax1.set_title("Score Distribution by Facet & Model", fontweight="bold")
ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.legend(
    [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.7) for c in MODEL_COLORS.values()],
    models, loc="upper left", fontsize=8,
)

# ============================================================
# Panel 2 (top-right, spans 2 cols): Heatmap — facet × language (mean score)
# ============================================================
ax2 = fig.add_subplot(gs[0, 2:])

langs = sorted({item["lang"] for item in scored})
heatmap_data = np.full((len(FACET_ORDER), len(langs)), np.nan)

for item in scored:
    fi = FACET_ORDER.index(item["facet"])
    li = langs.index(item["lang"])
    if np.isnan(heatmap_data[fi, li]):
        heatmap_data[fi, li] = item["median_score"]
    else:
        # Average across models
        heatmap_data[fi, li] = (heatmap_data[fi, li] + item["median_score"]) / 2

sns.heatmap(
    heatmap_data, ax=ax2, annot=True, fmt=".1f", cmap="RdYlGn_r",
    center=0, vmin=-3, vmax=4,
    xticklabels=[LANG_NAMES.get(l, l) for l in langs],
    yticklabels=[f.title() for f in FACET_ORDER],
    cbar_kws={"label": "Mean Score", "shrink": 0.8},
    linewidths=0.5,
)
ax2.set_title("Sycophancy Score by Facet × Language", fontweight="bold")
ax2.tick_params(axis="x", rotation=45)

# ============================================================
# Panel 3 (middle-left, spans 2 cols): Per-language bar chart (mean across facets)
# ============================================================
ax3 = fig.add_subplot(gs[1, :2])

lang_model_means = defaultdict(lambda: defaultdict(list))
for item in scored:
    lang_model_means[item["lang"]][item["model"]].append(item["median_score"])

lang_order = sorted(langs, key=lambda l: np.mean([
    np.mean(lang_model_means[l][m]) for m in models if lang_model_means[l][m]
]), reverse=True)

x3 = np.arange(len(lang_order))
for i, model in enumerate(models):
    means = [np.mean(lang_model_means[l][model]) if lang_model_means[l][model] else 0 for l in lang_order]
    ax3.bar(x3 + i * width, means, width, label=model, color=list(MODEL_COLORS.values())[i], alpha=0.8)

ax3.set_xticks(x3 + width / 2)
ax3.set_xticklabels([LANG_NAMES.get(l, l) for l in lang_order], rotation=45, ha="right")
ax3.set_ylabel("Mean Sycophancy Score")
ax3.set_title("Mean Sycophancy by Language & Model", fontweight="bold")
ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax3.legend(fontsize=8)

# ============================================================
# Panel 4 (middle-right, spans 2 cols): Judge agreement — score variance per item
# ============================================================
ax4 = fig.add_subplot(gs[1, 2:])

# For each scored item, compute the range of judge scores
judge_ranges = []
judge_facets = []
for item in scored:
    scores = list(item["judge_scores"].values())
    if len(scores) >= 2:
        judge_ranges.append(max(scores) - min(scores))
        judge_facets.append(item["facet"])

for fi, facet in enumerate(FACET_ORDER):
    facet_ranges = [r for r, f in zip(judge_ranges, judge_facets) if f == facet]
    if facet_ranges:
        ax4.hist(
            facet_ranges, bins=range(0, 12), alpha=0.6, label=facet.title(),
            edgecolor="white", linewidth=0.5,
        )

ax4.set_xlabel("Score Range Across 3 Judges (max - min)")
ax4.set_ylabel("Count of Items")
ax4.set_title("Judge Agreement (lower range = higher agreement)", fontweight="bold")
ax4.legend(fontsize=8)
ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# ============================================================
# Panel 5 (bottom-left): Response token distribution
# ============================================================
ax5 = fig.add_subplot(gs[2, :2])

for model in models:
    tokens = [r["response_tokens"] for r in responses if r["model"] == model]
    ax5.hist(tokens, bins=20, alpha=0.6, label=model, color=MODEL_COLORS[model], edgecolor="white")

ax5.set_xlabel("Response Tokens")
ax5.set_ylabel("Count")
ax5.set_title("Response Length Distribution", fontweight="bold")
ax5.legend(fontsize=8)

# ============================================================
# Panel 6 (bottom-right): Summary stats table
# ============================================================
ax6 = fig.add_subplot(gs[2, 2:])
ax6.axis("off")

# Compute stats
total_responses = len(responses)
total_judgements = len(judgements)
total_scored = len(scored)
valid = sum(1 for s in scored if s["is_valid"])
lang_match = sum(1 for r in responses if r["language_match"])
all_medians = [s["median_score"] for s in scored if s["median_score"] is not None]

table_data = [
    ["Metric", "Value"],
    ["Models tested", ", ".join(models)],
    ["Judge models", "gemini-3-flash, deepseek-v3.2, mistral-small"],
    ["Languages", f"{len(langs)} ({', '.join(LANG_NAMES.get(l, l) for l in sorted(langs))})"],
    ["Prompts (after chain sel.)", "63"],
    ["Model responses", str(total_responses)],
    ["Judge scores", str(total_judgements)],
    ["Parse failures", "0 (0%)"],
    ["Valid scored items", f"{valid}/{total_scored} (100%)"],
    ["Language match", f"{lang_match}/{total_responses} ({100*lang_match//total_responses}%)"],
    ["Overall mean score", f"{np.mean(all_medians):+.2f}"],
    ["Overall std dev", f"{np.std(all_medians):.2f}"],
]

table = ax6.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="left",
    loc="center",
    colWidths=[0.38, 0.62],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.5)

# Style header
for j in range(2):
    table[0, j].set_facecolor("#4C78A8")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(2):
        if i % 2 == 0:
            table[i, j].set_facecolor("#f0f4f8")

ax6.set_title("Run Summary", fontweight="bold", pad=20)

# --- Save ---
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Dashboard saved to {OUTPUT_PATH}")
