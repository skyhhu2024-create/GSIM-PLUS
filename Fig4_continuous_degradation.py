"""
Figure 4 — Continuous Gap Degradation (Line charts)
  Fig4a: NSE vs gap length (8 methods)
  Fig4b: KGE vs gap length (8 methods)
  MAML & MAML_DonorTrend plotted last (on top) with thicker lines for emphasis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(r"D:\1-Phd_work\1-Gobel_hypower_predict\Data\999-论文\GSIM全球插补1995-2015")
SUMMARY = ROOT / "05_Continuous_Gap_Validation" / "continuous_gap_summary.csv"
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
# Draw other methods first, MAML & MAML_DonorTrend last (on top)
OTHER_METHODS = ["RandomForest", "Linear",
                 "LSTM", "KNN", "SeasonalMean", "IDW"]
HIGHLIGHT_METHODS = ["MAML", "MAML_DonorTrend"]

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
METHOD_MARKERS = {
    "MAML": "o", "MAML_DonorTrend": "p",
    "RandomForest": "s", "Linear": "^", "LSTM": "D",
    "KNN": "v", "SeasonalMean": "h", "IDW": "P",
}

# ── data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(SUMMARY)
gap_map = {"3_months": 3, "6_months": 6, "12_months": 12}
df["gap_length"] = df["scenario"].map(gap_map)

# ── Fig4a/b: Line charts ────────────────────────────────────────────────
for metric, suffix in [("NSE", "a"), ("KGE", "b")]:
    fig, ax = plt.subplots(figsize=(85 / 25.4, 70 / 25.4), dpi=300)

    # draw other methods first (thinner, lower zorder)
    for method in OTHER_METHODS:
        sub = df[df["method"] == method].sort_values("gap_length")
        ax.plot(sub["gap_length"], sub[metric],
                color=NPG_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=4.0, lw=0.9, alpha=0.7,
                markeredgewidth=0.4, markeredgecolor="white",
                label=METHOD_LABELS[method], zorder=2)

    # draw MAML & MAML_DonorTrend last — thicker, fully opaque, higher zorder
    for method in HIGHLIGHT_METHODS:
        sub = df[df["method"] == method].sort_values("gap_length")
        ax.plot(sub["gap_length"], sub[metric],
                color=NPG_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=6.0, lw=2.0, alpha=1.0,
                markeredgewidth=0.5, markeredgecolor="white",
                label=METHOD_LABELS[method], zorder=10)

    ax.set_xticks([3, 6, 12])
    ax.set_xticklabels(["3 months", "6 months", "12 months"], fontsize=6.5)
    ax.set_xlabel("Gap length", fontsize=8)
    ax.set_ylabel(metric, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', pad=3)

    # y-axis: give room so lines don't overlap with legend
    if metric == "NSE":
        ax.set_ylim(0.1, 1.02)
    else:  # KGE
        ax.set_ylim(0.4, 1.02)

    # legend: MAML_DonorTrend first, then MAML, then others — placed outside top
    handles, labels = ax.get_legend_handles_labels()
    dt_idx = labels.index("MAML DonorTrend")
    maml_idx = labels.index("MAML")
    other_idx = [i for i in range(len(labels)) if i not in (dt_idx, maml_idx)]
    order = [dt_idx, maml_idx] + other_idx
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper center", bbox_to_anchor=(0.5, 1.18),
              fontsize=5.5, ncol=4, columnspacing=0.8, handletextpad=0.3)

    fname = f"Fig4{suffix}_degradation_{metric}"
    fig.savefig(str(OUTDIR / f"{fname}.png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUTDIR / f"{fname}.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {fname}")
