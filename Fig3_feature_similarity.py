"""
Figure 3 — Feature Importance & Similarity Distribution
(a) Horizontal bar: 15 feature weights, grouped by category
(b) Histogram: Top-5 donor similarity score distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(r"D:\1-Phd_work\1-Gobel_hypower_predict\Data\999-论文\GSIM全球插补1995-2015")
WEIGHTS = ROOT / "03_Similarity_Matching" / "feature_weights.csv"
TOP5 = ROOT / "03_Similarity_Matching" / "top_5_similar_stations.csv"
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

# ── feature group config ────────────────────────────────────────────────
GROUP_MAP = {
    "mean_flow_m3s":       "Hydrological",
    "upstream_area_km2":   "Hydrological",
    "area_local_hybas_km2":"Hydrological",
    "e_mean":   "Climate",
    "e_std":    "Climate",
    "tp_mean":  "Climate",
    "tp_std":   "Climate",
    "tp_cv":    "Climate",
    "t2m_mean": "Climate",
    "t2m_std":  "Climate",
    "latitude": "Spatial",
    "longitude":"Spatial",
    "altitude": "Topography",
    "slope":    "Topography",
    "snd":      "Soil",
    "slt":      "Soil",
    "scl":      "Soil",
}
GROUP_COLORS = {
    "Hydrological":"#4DBBD5",
    "Climate":     "#E64B35",
    "Spatial":     "#3C5488",
    "Topography":  "#00A087",
    "Soil":        "#F39B7F",
}
FEATURE_DISPLAY = {
    "mean_flow_m3s":       "Mean flow",
    "upstream_area_km2":   "Upstream area",
    "area_local_hybas_km2":"Local basin area",
    "e_mean":   "Evaporation (mean)",
    "e_std":    "Evaporation (std)",
    "latitude": "Latitude",
    "slope":    "Slope",
    "scl":      "Clay content",
    "slt":      "Silt content",
    "snd":      "Sand content",
    "longitude":"Longitude",
    "altitude": "Altitude",
    "tp_cv":    "Precipitation (CV)",
    "tp_std":   "Precipitation (std)",
    "tp_mean":  "Precipitation (mean)",
    "t2m_std":  "Temperature (std)",
    "t2m_mean": "Temperature (mean)",
}

# ── data ─────────────────────────────────────────────────────────────────
wdf = pd.read_csv(WEIGHTS)
wdf["group"] = wdf["feature"].map(GROUP_MAP)
wdf["display"] = wdf["feature"].map(FEATURE_DISPLAY)
wdf["pct"] = wdf["weight"] / wdf["weight"].sum() * 100
# sort by weight descending (bottom-to-top in horizontal bar)
wdf = wdf.sort_values("weight", ascending=True).reset_index(drop=True)

sim = pd.read_csv(TOP5)

# ══════════════════════════════════════════════════════════════════════════
# (a) Feature weights — horizontal bar
# ══════════════════════════════════════════════════════════════════════════
fig_a, ax = plt.subplots(figsize=(85 / 25.4, 90 / 25.4), dpi=300)

colors = [GROUP_COLORS[g] for g in wdf["group"]]
bars = ax.barh(range(len(wdf)), wdf["pct"], color=colors, edgecolor="white",
               linewidth=0.3, height=0.7)

# percentage labels
for i, (pct, w) in enumerate(zip(wdf["pct"], wdf["weight"])):
    if pct > 1:
        ax.text(pct + 0.5, i, f"{pct:.1f}%", va="center", fontsize=5.5)
    else:
        ax.text(max(pct, 0) + 0.5, i, f"<0.1%", va="center", fontsize=5, color="#999999")

ax.set_yticks(range(len(wdf)))
ax.set_yticklabels(wdf["display"], fontsize=6.5)
ax.set_xlabel("Weight (%)", fontsize=8)
ax.set_xlim(0, 25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# group legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=GROUP_COLORS[g], label=f"{g}")
                   for g in ["Hydrological", "Climate", "Spatial", "Topography", "Soil"]]
ax.legend(handles=legend_elements, loc="lower right", fontsize=6,
          handlelength=1.0, handletextpad=0.4)

fig_a.savefig(str(OUTDIR / "Fig3a_feature_weights.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig_a.savefig(str(OUTDIR / "Fig3a_feature_weights.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_a)
print("Saved Fig3a_feature_weights")

# ══════════════════════════════════════════════════════════════════════════
# (b) Similarity distribution
# ══════════════════════════════════════════════════════════════════════════
fig_b, ax2 = plt.subplots(figsize=(85 / 25.4, 65 / 25.4), dpi=300)

# all top-5 similarities
all_sim = sim["similarity"].values

# separate by rank
rank1 = sim[sim["rank"] == 1]["similarity"]
rank2_5 = sim[sim["rank"] > 1]["similarity"]

ax2.hist(rank2_5, bins=80, range=(0.7, 1.0), color="#8491B4", alpha=0.6,
         edgecolor="white", linewidth=0.2, label="Rank 2–5", density=True)
ax2.hist(rank1, bins=80, range=(0.7, 1.0), color="#E64B35", alpha=0.7,
         edgecolor="white", linewidth=0.2, label="Rank 1 (best)", density=True)

# stats annotation
mean_s = np.mean(all_sim)
std_s = np.std(all_sim)
median_s = np.median(all_sim)
ax2.axvline(mean_s, color="black", ls="--", lw=0.6, alpha=0.7)
ax2.text(mean_s - 0.005, ax2.get_ylim()[1] * 0.92,
         f"Mean = {mean_s:.3f}\nMedian = {median_s:.3f}\nStd = {std_s:.3f}",
         fontsize=6, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.3, alpha=0.9))

ax2.set_xlabel("Similarity score", fontsize=8)
ax2.set_ylabel("Density", fontsize=8)
ax2.set_xlim(0.7, 1.0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(loc="upper left", fontsize=6)

fig_b.savefig(str(OUTDIR / "Fig3b_similarity_distribution.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig_b.savefig(str(OUTDIR / "Fig3b_similarity_distribution.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_b)
print("Saved Fig3b_similarity_distribution")
