"""
Figure 5 — Hybrid Scenario Validation H1/H2/H3
  H1: from original summary; H2/H3: from excluding_abnormal summary
  Fig5a: Heatmap NSE
  Fig5b: Heatmap KGE
  Fig5c/d: Grouped bar NSE / KGE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(r"D:\1-Phd_work\1-Gobel_hypower_predict\Data\999-论文\GSIM全球插补1995-2015")
SUMMARY_FULL = ROOT / "06_Hybrid_Validation" / "hybrid_h123_summary.csv"
SUMMARY_EXCL = ROOT / "06_Hybrid_Validation" / "hybrid_h123_summary_excluding_abnormal.csv"
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
METHOD_ORDER = ["MAML", "MAML_DonorTrend", "RandomForest", "Linear",
                "LSTM", "KNN", "SeasonalMean", "IDW"]
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
SCENARIO_LABELS = {
    "H1_sparse_dominant":   "H1 (Sparse)",
    "H2_balanced_mixed":    "H2 (Balanced)",
    "H3_long_gap_dominant": "H3 (Long gap)",
}
SCENARIO_ORDER = ["H1_sparse_dominant", "H2_balanced_mixed", "H3_long_gap_dominant"]
SCENARIO_COLORS = ["#3C5488", "#E64B35", "#00A087"]

# ── data: H1 from full summary, H2/H3 from excluding_abnormal ───────────
df_full = pd.read_csv(SUMMARY_FULL)
df_excl = pd.read_csv(SUMMARY_EXCL)

# H1 from original (no abnormal issue in H1)
df_h1 = df_full[df_full["scenario"] == "H1_sparse_dominant"].copy()
# H2/H3 from excluding_abnormal
df_h23 = df_excl[df_excl["scenario"].isin(["H2_balanced_mixed", "H3_long_gap_dominant"])].copy()

df = pd.concat([df_h1, df_h23], ignore_index=True)

# ── Heatmaps (one per metric) ───────────────────────────────────────────
for metric, suffix in [("NSE", "a"), ("KGE", "b")]:
    fig, ax = plt.subplots(figsize=(85 / 25.4, 60 / 25.4), dpi=300)

    pivot = df.pivot_table(index="method", columns="scenario", values=metric)
    pivot = pivot.loc[METHOD_ORDER, SCENARIO_ORDER]

    vmin = pivot.values.min() - 0.02
    vmax = pivot.values.max() + 0.02

    im = ax.imshow(pivot.values, cmap="RdYlBu", aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            col_best = pivot.iloc[:, j].idxmax()
            weight = "bold" if pivot.index[i] == col_best else "normal"
            norm_val = (val - vmin) / (vmax - vmin)
            color = "white" if norm_val < 0.35 or norm_val > 0.75 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=6, fontweight=weight, color=color)

    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER], fontsize=6.5)
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=6.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.ax.tick_params(labelsize=5.5)

    fname = f"Fig5{suffix}_hybrid_heatmap_{metric}"
    fig.savefig(str(OUTDIR / f"{fname}.png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUTDIR / f"{fname}.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {fname}")

# ── Grouped bar (one per metric) ────────────────────────────────────────
for metric, suffix in [("NSE", "c"), ("KGE", "d")]:
    fig, ax = plt.subplots(figsize=(85 / 25.4, 65 / 25.4), dpi=300)

    n_methods = len(METHOD_ORDER)
    bar_width = 0.25
    x = np.arange(n_methods)

    for si, scenario in enumerate(SCENARIO_ORDER):
        vals = []
        for method in METHOD_ORDER:
            row = df[(df["method"] == method) & (df["scenario"] == scenario)]
            vals.append(row[metric].values[0])
        offset = (si - 1) * bar_width
        ax.bar(x + offset, vals, bar_width,
               color=SCENARIO_COLORS[si], edgecolor="white", linewidth=0.3,
               alpha=0.85, label=SCENARIO_LABELS[scenario])

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER],
                       rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel(metric, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=5.5)

    fname = f"Fig5{suffix}_hybrid_bar_{metric}"
    fig.savefig(str(OUTDIR / f"{fname}.png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUTDIR / f"{fname}.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {fname}")
