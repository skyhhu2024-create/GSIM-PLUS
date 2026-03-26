"""
Figure 6 — Super-Long Gap Analysis (25+ months, excluding abnormal stations)
  Fig6a: Bar chart (NSE & KGE, 8 methods)
  Fig6b: 8-method scatter panel (2×4)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(r"D:\1-Phd_work\1-Gobel_hypower_predict\Data\999-论文\GSIM全球插补1995-2015")
SUMMARY = ROOT / "07_SuperLong_Gap_Analysis" / "super_long_summary_excluding_abnormal.csv"
PRED_DIR = ROOT / "07_SuperLong_Gap_Analysis" / "super_long_25plus"
OUTDIR = ROOT / "111-paper"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
})

# ── config ───────────────────────────────────────────────────────────────
NPG_COLORS = {
    "MAML":          "#E64B35",
    "MAML_DonorTrend":"#00A087",
    "RandomForest":  "#3C5488",
    "Linear":        "#4DBBD5",
    "LSTM":          "#F39B7F",
    "KNN":           "#8491B4",
    "SeasonalMean":  "#91D1C2",
    "IDW":           "#B09C85",
}
METHOD_LABELS = {
    "MAML":          "MAML",
    "MAML_DonorTrend":"MAML DonorTrend",
    "RandomForest":  "Random Forest",
    "Linear":        "Linear",
    "LSTM":          "LSTM",
    "KNN":           "KNN",
    "SeasonalMean":  "Seasonal Mean",
    "IDW":           "IDW",
}
DISPLAY_ORDER = ["MAML", "MAML_DonorTrend", "RandomForest", "Linear",
                 "LSTM", "KNN", "SeasonalMean", "IDW"]

# ── data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(SUMMARY)
df = df[df["Method"].isin(DISPLAY_ORDER)]
df = df.set_index("Method").loc[DISPLAY_ORDER].reset_index()

# ── Fig6a: Bar chart ────────────────────────────────────────────────────
fig_a, ax = plt.subplots(figsize=(130 / 25.4, 55 / 25.4), dpi=300)
x = np.arange(len(df))
w = 0.35

colors = [NPG_COLORS[m] for m in df["Method"]]
colors_light = [to_rgba(c, alpha=0.45) for c in colors]

bars_nse = ax.bar(x - w/2, df["NSE"], w, label="NSE",
                  color=colors, edgecolor="white", linewidth=0.3)
bars_kge = ax.bar(x + w/2, df["KGE"], w, label="KGE",
                  color=colors_light, edgecolor=colors, linewidth=0.5)

for bar in bars_nse:
    h = bar.get_height()
    y_pos = h + 0.008 if h >= 0 else h - 0.008
    va = "bottom" if h >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{h:.3f}",
            ha="center", va=va, fontsize=5, rotation=90)
for bar in bars_kge:
    h = bar.get_height()
    y_pos = h + 0.008 if h >= 0 else h - 0.008
    va = "bottom" if h >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{h:.3f}",
            ha="center", va=va, fontsize=5, rotation=90)

ax.axhline(0, color="black", lw=0.3, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABELS[m] for m in df["Method"]],
                   rotation=40, ha="right", fontsize=6.5)
ax.set_ylabel("Score")
ax.set_ylim(-0.4, 0.85)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=6.5, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig_a.savefig(str(OUTDIR / "Fig6a_superlong_bar.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig_a.savefig(str(OUTDIR / "Fig6a_superlong_bar.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_a)
print("Saved Fig6a_superlong_bar")

# ── Fig6b: 8-method scatter panel (2×4) ─────────────────────────────────
fig_b, axes = plt.subplots(2, 4, figsize=(180 / 25.4, 90 / 25.4), dpi=300)
axes = axes.flatten()

metrics = df.set_index("Method")

for i, method in enumerate(DISPLAY_ORDER):
    ax = axes[i]
    pred_file = PRED_DIR / f"{method}_predictions.csv"
    pred = pd.read_csv(pred_file)

    if len(pred) > 15000:
        plot_df = pred.sample(15000, random_state=42)
    else:
        plot_df = pred

    ax.scatter(plot_df["true"], plot_df["pred"], s=1.0, alpha=0.25,
               c=NPG_COLORS[method], linewidths=0, rasterized=True)

    lims = [0, max(plot_df["true"].quantile(0.999), plot_df["pred"].quantile(0.999))]
    ax.plot(lims, lims, "k-", lw=0.5, alpha=0.4, zorder=5)

    nse = metrics.loc[method, "NSE"]
    kge = metrics.loc[method, "KGE"]
    ax.text(0.05, 0.95, f"{METHOD_LABELS[method]}\nNSE={nse:.3f}\nKGE={kge:.3f}",
            transform=ax.transAxes, fontsize=5.5, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.3, alpha=0.9))

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if i >= 4:
        ax.set_xlabel(r"Observed (m$^3$/s)", fontsize=6.5)
    if i % 4 == 0:
        ax.set_ylabel(r"Predicted (m$^3$/s)", fontsize=6.5)

fig_b.subplots_adjust(hspace=0.3, wspace=0.3)
fig_b.savefig(str(OUTDIR / "Fig6b_superlong_scatter.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig_b.savefig(str(OUTDIR / "Fig6b_superlong_scatter.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_b)
print("Saved Fig6b_superlong_scatter")
