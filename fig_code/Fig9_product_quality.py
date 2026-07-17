"""
Figure 9 — GSIM-PLUS Product Quality Overview (Target + Anchor stations)
  Fig9a: Quality flag pie chart (Q0/Q1/Q2/Q3)
  Fig9b: Before/after completeness CDF
  Fig9c: Global map — stations colored by dominant quality flag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__import__("os").environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
TARGET_DIR = ROOT / "08_GSIM_PLUS_Product" / "dtrr_guarded" / "GSIM_fill"
ANCHOR_DIR = ROOT / "08_GSIM_PLUS_Product" / "DTRR_Guarded_Anchor" / "GSIM_fill_anchor"
FEAT_CSV = ROOT / "02_Feature_Table" / "station_features_with_meteo.csv"
OUTDIR = ROOT / "111-paper"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
})

# ── colors ───────────────────────────────────────────────────────────────
Q_COLORS = {
    "Q0": "#999999",   # observed - neutral grey
    "Q1": "#56B4E9",   # short gap - sky blue
    "Q2": "#E69F00",   # medium gap - orange
    "Q3": "#CC79A7",   # long gap - reddish purple
}
Q_LABELS = {
    "Q0": "Q0 (Observed)",
    "Q1": "Q1 (Gap ≤ 3 months)",
    "Q2": "Q2 (4–24 months)",
    "Q3": "Q3 (> 24 months)",
}

# ── data ─────────────────────────────────────────────────────────────────
print("Loading target + anchor station CSVs...")
dfs = []
for src_dir, label in [(TARGET_DIR, "target"), (ANCHOR_DIR, "anchor")]:
    csvs = sorted(src_dir.glob("*.csv"))
    print(f"  {label}: {len(csvs)} files")
    for fp in csvs:
        tmp = pd.read_csv(fp)
        tmp["station_type"] = label
        dfs.append(tmp)
df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df):,}, stations: {df['station_id'].nunique():,}")
feat = pd.read_csv(FEAT_CSV)

# ══════════════════════════════════════════════════════════════════════════
# Fig9a: Quality flag pie chart
# ══════════════════════════════════════════════════════════════════════════
fig_a, ax = plt.subplots(figsize=(85 / 25.4, 75 / 25.4), dpi=600)

qf_counts = df["quality_flag"].value_counts()
sizes = [qf_counts.get(q, 0) for q in ["Q0", "Q1", "Q2", "Q3"]]
colors = [Q_COLORS[q] for q in ["Q0", "Q1", "Q2", "Q3"]]
labels = [Q_LABELS[q] for q in ["Q0", "Q1", "Q2", "Q3"]]

wedges, texts, autotexts = ax.pie(
    sizes, labels=None, colors=colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.45, edgecolor="white", linewidth=0.8),
    textprops=dict(fontsize=8),
)
for t in autotexts:
    t.set_fontsize(8)

ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.05, 0.5),
          fontsize=8, handlelength=1.0, handletextpad=0.4)

fig_a.savefig(str(OUTDIR / "Fig9a_quality_pie.png"), dpi=600, bbox_inches="tight", facecolor="white")
fig_a.savefig(str(OUTDIR / "Fig9a_quality_pie.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_a)
print("Saved Fig9a_quality_pie")

# ══════════════════════════════════════════════════════════════════════════
# Fig9b: Before/after completeness CDF
# ══════════════════════════════════════════════════════════════════════════
per_station = df.groupby("station_id").agg(
    observed=("fill_method", lambda x: (x == "OBSERVED").sum()),
    total=("date", "count"),
).reset_index()
per_station["pct_before"] = per_station["observed"] / 252 * 100
per_station["pct_after"] = per_station["total"] / 252 * 100

fig_b, ax = plt.subplots(figsize=(130 / 25.4, 55 / 25.4), dpi=600)

for data, label, color, ls in [
    (per_station["pct_before"], "Before gap-filling", "#0072B2", "--"),
    (per_station["pct_after"],  "After gap-filling",  "#E69F00", "-"),
]:
    sorted_d = np.sort(data.values)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, cdf, color=color, ls=ls, lw=1.5, label=label)

ax.set_xlabel("Data completeness (%)", fontsize=8)
ax.set_ylabel("CDF", fontsize=8)
ax.set_xlim(25, 105)
ax.set_ylim(0, 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", fontsize=8)

# annotation
median_before = per_station["pct_before"].median()
median_after = per_station["pct_after"].median()
ax.axvline(median_before, color="#0072B2", ls=":", lw=0.8, alpha=0.7)
ax.axvline(median_after, color="#E69F00", ls=":", lw=0.8, alpha=0.7)
ax.text(median_before - 1, 0.55, f"Median\n{median_before:.1f}%",
        fontsize=8, ha="right", color="#0072B2")
ax.text(median_after + 1, 0.45, f"Median\n{median_after:.1f}%",
        fontsize=8, ha="left", color="#E69F00")

fig_b.savefig(str(OUTDIR / "Fig9b_completeness_cdf.png"), dpi=600, bbox_inches="tight", facecolor="white")
fig_b.savefig(str(OUTDIR / "Fig9b_completeness_cdf.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_b)
print("Saved Fig9b_completeness_cdf")

# ══════════════════════════════════════════════════════════════════════════
# Fig9c: Global map — stations colored by fill percentage
# ══════════════════════════════════════════════════════════════════════════
per_station = per_station.merge(feat[["station_id", "latitude", "longitude"]], on="station_id")
per_station["fill_pct"] = (per_station["total"] - per_station["observed"]) / per_station["total"] * 100

# dominant quality flag per station
dom_q = df.groupby("station_id")["quality_flag"].agg(
    lambda x: x.value_counts().index[0] if x.value_counts().index[0] != "Q0"
    else (x.value_counts().index[1] if len(x.value_counts()) > 1 else "Q0")
)
per_station = per_station.merge(dom_q.rename("dom_flag"), left_on="station_id", right_index=True)

# for stations with no filling, dom_flag = Q0
per_station.loc[per_station["fill_pct"] == 0, "dom_flag"] = "Q0"

fig_c, ax = plt.subplots(figsize=(180 / 25.4, 90 / 25.4), dpi=600)

# coastlines
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig_c, ax = plt.subplots(figsize=(180 / 25.4, 90 / 25.4), dpi=600,
                              subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#cccccc")

    for q in ["Q0", "Q1", "Q2", "Q3"]:
        sub = per_station[per_station["dom_flag"] == q]
        if len(sub) == 0:
            continue
        ax.scatter(sub["longitude"], sub["latitude"],
                   s=1.5, c=Q_COLORS[q], alpha=0.6, linewidths=0,
                   transform=ccrs.PlateCarree(), label=Q_LABELS[q], rasterized=True)

    ax.legend(loc="lower left", fontsize=8, markerscale=4,
              handletextpad=0.3, borderpad=0.3)

except ImportError:
    # fallback without cartopy
    for q in ["Q0", "Q1", "Q2", "Q3"]:
        sub = per_station[per_station["dom_flag"] == q]
        if len(sub) == 0:
            continue
        ax.scatter(sub["longitude"], sub["latitude"],
                   s=1.5, c=Q_COLORS[q], alpha=0.6, linewidths=0,
                   label=Q_LABELS[q], rasterized=True)

    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, markerscale=4)

fig_c.savefig(str(OUTDIR / "Fig9c_quality_map.png"), dpi=600, bbox_inches="tight", facecolor="white")
fig_c.savefig(str(OUTDIR / "Fig9c_quality_map.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_c)
print("Saved Fig9c_quality_map")


