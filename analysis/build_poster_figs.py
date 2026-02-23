"""Insert a poster-figures cell into the notebook and run it."""
import json

NB_PATH = "poster_analysis.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

CELL_SOURCE = """# Poster-Ready Figures — Apart Dark Theme
# Generates fig5_poster.png, fig6_poster.png, fig7_poster.png

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from scipy import stats as sp_stats

# --- Apart Dark Theme ---
APART_BG = "#3b3f47"
APART_GREEN = "#5cffb1"
APART_TEXT = "#e8e8e8"
APART_MUTED = "#9a9da3"
APART_CARD = "#4a4e57"

# Full model names for poster
MODEL_FULL = {
    "deepseek-v3.2": "DeepSeek v3.2",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
    "gpt-5.1": "GPT-5.1",
    "grok-4.1": "Grok 4.1",
    "kimi-k2.5": "Kimi K2.5",
    "mistral-large": "Mistral Large",
}

plt.rcParams.update({
    "figure.facecolor": APART_BG,
    "axes.facecolor": APART_BG,
    "savefig.facecolor": APART_BG,
    "text.color": APART_TEXT,
    "axes.labelcolor": APART_TEXT,
    "xtick.color": APART_TEXT,
    "ytick.color": APART_TEXT,
    "axes.edgecolor": APART_MUTED,
    "grid.color": "#555960",
    "font.family": "sans-serif",
    "font.size": 14,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
sns.set_style("darkgrid", {
    "axes.facecolor": APART_BG,
    "figure.facecolor": APART_BG,
    "grid.color": "#555960",
})

# ========== FIGURE 6 POSTER: Rate Heatmap ==========

rate_data = syc.groupby(["lang", "model"])["norm_score"].apply(
    lambda x: (x >= 0.30).mean()
).unstack("model")
rate_data = rate_data.reindex(index=LANG_ORDER, columns=MODEL_ORDER)

rate_row_means = rate_data.mean(axis=1)
rate_col_means = rate_data.mean(axis=0)

rate_display = rate_data.copy()
rate_display.index = [LANG_NAMES[l] for l in rate_display.index]
rate_display.columns = [MODEL_FULL[m] for m in rate_display.columns]

fig, (ax_main, ax_bar) = plt.subplots(
    1, 2, figsize=(16, 9),
    gridspec_kw={"width_ratios": [7, 1.2], "wspace": 0.04},
)
fig.subplots_adjust(top=0.88)

cmap_rate = mcolors.LinearSegmentedColormap.from_list(
    "apart_heat", ["#3b3f47", "#fee08b", "#f46d43", "#a50026"], N=256
)

annot_pct = rate_display.map(lambda v: f"{v:.0%}")
sns.heatmap(
    rate_display, annot=annot_pct, fmt="", cmap=cmap_rate,
    vmin=0, vmax=0.55,
    linewidths=1.5, linecolor=APART_BG,
    cbar_kws={"label": "", "shrink": 0.7, "format": mticker.PercentFormatter(1.0)},
    ax=ax_main,
    annot_kws={"fontsize": 15, "fontweight": "bold", "color": APART_TEXT},
)
cbar = ax_main.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color=APART_TEXT)
for label in cbar.ax.get_yticklabels():
    label.set_color(APART_TEXT)

ax_main.set_xlabel("")
ax_main.set_ylabel("")
ax_main.set_title(
    "Moderate+ Sycophancy Rate by Language & Model",
    fontsize=22, fontweight="bold", color=APART_TEXT, pad=35,
)
ax_main.text(
    0.5, -0.07,
    "% of items scoring \\u2265 0.30 on 0\\u20131 normalized scale  |  19,600 responses \\u00d7 6-judge panel",
    transform=ax_main.transAxes, ha="center", fontsize=11, color=APART_MUTED, style="italic",
)
# Force all tick labels white
ax_main.tick_params(labelsize=13, colors=APART_TEXT)
for label in ax_main.get_xticklabels() + ax_main.get_yticklabels():
    label.set_color(APART_TEXT)

# Column means — positioned between title and heatmap top edge
for j, val in enumerate(rate_col_means):
    ax_main.text(j + 0.5, -0.6, f"{val:.0%}", ha="center", va="bottom",
                 fontsize=11, color=APART_GREEN, fontweight="bold")

y_pos = np.arange(len(rate_row_means))
ax_bar.barh(y_pos, rate_row_means.values, color=APART_GREEN, edgecolor=APART_BG, height=0.75, alpha=0.85)
ax_bar.set_yticks([])
ax_bar.set_xlim(0, rate_row_means.max() * 1.35)
ax_bar.set_xlabel("Mean", fontsize=11, color=APART_MUTED)
ax_bar.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax_bar.invert_yaxis()
for i, val in enumerate(rate_row_means.values):
    ax_bar.text(val + 0.008, i, f"{val:.0%}", va="center", fontsize=11,
                color=APART_TEXT, fontweight="bold")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.spines["left"].set_visible(False)
ax_bar.tick_params(labelsize=10, colors=APART_TEXT)

fig.savefig(OUT_FIG / "fig6_poster.png")
plt.show()
print(f"Saved: {OUT_FIG / 'fig6_poster.png'}")


# ========== FIGURE 7 POSTER: Severity Stacked Bars ==========

sev_counts = syc.groupby(["model", "severity"]).size().unstack("severity", fill_value=0)
sev_counts = sev_counts.reindex(columns=SEVERITY_LABELS, fill_value=0)
sev_frac = sev_counts.div(sev_counts.sum(axis=1), axis=0)

sev_frac["syc_total"] = 1 - sev_frac["None"]
sev_frac = sev_frac.sort_values("syc_total", ascending=True)
sev_frac = sev_frac.drop(columns="syc_total")
sev_frac.index = [MODEL_FULL[m] for m in sev_frac.index]

tier_colors = {
    "None": "#555960",
    "Mild": "#fee08b",
    "Moderate": "#f46d43",
    "Strong": "#a50026",
}

fig, ax = plt.subplots(figsize=(16, 6))

left_vals = np.zeros(len(sev_frac))
for tier in SEVERITY_LABELS:
    vals = sev_frac[tier].values
    ax.barh(sev_frac.index, vals, left=left_vals, color=tier_colors[tier],
            edgecolor=APART_BG, linewidth=1.5, label=tier, height=0.65)
    for i, (v, lv) in enumerate(zip(vals, left_vals)):
        if v >= 0.07:
            tc = APART_TEXT if tier in ("None", "Strong") else "#333"
            ax.text(lv + v / 2, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=tc)
    left_vals += vals

ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlabel("Proportion of Items", fontsize=14, color=APART_TEXT)
ax.set_title("Sycophancy Severity Distribution by Model",
             fontsize=22, fontweight="bold", color=APART_TEXT, pad=20)
ax.tick_params(labelsize=14, colors=APART_TEXT)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color(APART_TEXT)

legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True,
                   fontsize=12, title="Severity", facecolor=APART_CARD,
                   edgecolor=APART_MUTED, borderpad=1)
legend.get_title().set_color(APART_TEXT)
for text in legend.get_texts():
    text.set_color(APART_TEXT)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(right=0.85)
fig.savefig(OUT_FIG / "fig7_poster.png")
plt.show()
print(f"Saved: {OUT_FIG / 'fig7_poster.png'}")


# ========== FIGURE 5 POSTER: COMET Scatter ==========

lang_agg = syc[syc["lang"] != "en"].groupby("lang").agg(
    mean_syc=("norm_score", "mean"),
    mean_comet=("comet_score", "mean"),
).reset_index()
lang_agg["lang_name"] = lang_agg["lang"].map(LANG_NAMES)

r_val, p_val = sp_stats.pearsonr(lang_agg["mean_comet"], lang_agg["mean_syc"])

fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(lang_agg["mean_comet"], lang_agg["mean_syc"],
           s=140, zorder=5, color=APART_GREEN, edgecolor=APART_BG, linewidth=1.5)

for _, row in lang_agg.iterrows():
    ax.annotate(
        row["lang_name"], (row["mean_comet"], row["mean_syc"]),
        textcoords="offset points", xytext=(10, 5),
        fontsize=13, color=APART_TEXT, fontweight="bold",
    )

x_fit = np.linspace(lang_agg["mean_comet"].min() - 0.005,
                     lang_agg["mean_comet"].max() + 0.005, 50)
slope, intercept = np.polyfit(lang_agg["mean_comet"], lang_agg["mean_syc"], 1)
ax.plot(x_fit, slope * x_fit + intercept, color="#f46d43", ls="--", lw=2, alpha=0.7)

sig_label = "n.s." if p_val >= 0.05 else f"p={p_val:.3f}"
ax.text(
    0.05, 0.95,
    f"r = {r_val:.2f}  ({sig_label})\\nn = {len(lang_agg)} languages",
    transform=ax.transAxes, fontsize=14, va="top", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor=APART_CARD,
              edgecolor=APART_MUTED, alpha=0.9),
    color=APART_TEXT,
)

ax.set_xlabel("Mean COMET Score (translation quality)", fontsize=15, color=APART_TEXT)
ax.set_ylabel("Mean Sycophancy (norm_score)", fontsize=15, color=APART_TEXT)
ax.set_title("Translation Quality Does Not\\nExplain Sycophancy Differences",
             fontsize=20, fontweight="bold", color=APART_TEXT, pad=16)
ax.tick_params(labelsize=12, colors=APART_TEXT)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color(APART_TEXT)

fig.savefig(OUT_FIG / "fig5_poster.png")
plt.show()
print(f"Saved: {OUT_FIG / 'fig5_poster.png'}")

# ========== FIGURE 9 POSTER: Mirror Facet by Language ==========

mirror = syc[syc["facet"] == "mirror"].copy()

# Moderate+ rate by language for mirror only
mirror_rate = mirror.groupby("lang")["norm_score"].apply(lambda x: (x >= 0.30).mean())
mirror_mean = mirror.groupby("lang")["norm_score"].mean()

# Build a combined df for plotting
mirror_lang = pd.DataFrame({
    "lang": mirror_rate.index,
    "rate": mirror_rate.values,
    "mean": mirror_mean.values,
})
mirror_lang["lang_name"] = mirror_lang["lang"].map(LANG_NAMES)
mirror_lang["resource"] = mirror_lang["lang"].map(RESOURCE_LEVELS)
mirror_lang = mirror_lang.sort_values("rate", ascending=True)

# Color by resource level
res_colors = {"High": APART_GREEN, "Medium": "#fee08b", "Low": "#f46d43"}

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(
    mirror_lang["lang_name"], mirror_lang["rate"],
    color=[res_colors[r] for r in mirror_lang["resource"]],
    edgecolor=APART_BG, linewidth=1.5, height=0.65, alpha=0.9,
)

# Annotate each bar with rate %
for i, (_, row) in enumerate(mirror_lang.iterrows()):
    ax.text(row["rate"] + 0.01, i, f'{row["rate"]:.0%}',
            va="center", fontsize=14, fontweight="bold", color=APART_TEXT)

ax.set_xlim(0, mirror_lang["rate"].max() * 1.25)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlabel("", fontsize=14, color=APART_TEXT)
ax.set_title("Opinion Mirroring: The Facet Where Language Matters",
             fontsize=22, fontweight="bold", color=APART_TEXT, pad=20)
ax.tick_params(labelsize=14, colors=APART_TEXT)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color(APART_TEXT)

# Subtitle below x-axis
ax.text(0.5, -0.09,
    "Moderate+ rate (norm_score \u2265 0.30)  |  Mirror facet only  |  \u03b7\u00b2 = 0.039, the strongest cross-linguistic effect",
    transform=ax.transAxes, ha="center", fontsize=11, color=APART_MUTED, style="italic")

# Resource level legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=APART_GREEN, edgecolor=APART_BG, label="High resource"),
    Patch(facecolor="#fee08b", edgecolor=APART_BG, label="Medium resource"),
    Patch(facecolor="#f46d43", edgecolor=APART_BG, label="Low resource"),
]
legend = ax.legend(handles=legend_elements, loc="lower right", frameon=True,
                   fontsize=12, facecolor=APART_CARD, edgecolor=APART_MUTED)
for text in legend.get_texts():
    text.set_color(APART_TEXT)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(OUT_FIG / "fig9_mirror_poster.png")
plt.show()
print(f"Saved: {OUT_FIG / 'fig9_mirror_poster.png'}")

# Reset style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "text.color": "black",
    "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
})
print("All poster figures generated.")
"""

poster_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": CELL_SOURCE,
}

# Find and replace existing poster cell, or insert before last cell
poster_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []) if isinstance(cell.get("source"), list) else cell.get("source", ""))
    if "Poster-Ready Figures" in src and "Apart Dark Theme" in src:
        poster_idx = i
        break

if poster_idx is not None:
    nb["cells"][poster_idx] = poster_cell
    print(f"Replaced existing poster cell at index {poster_idx}.")
else:
    nb["cells"].insert(len(nb["cells"]) - 1, poster_cell)
    print("Inserted new poster cell.")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Poster cell inserted successfully.")
