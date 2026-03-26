"""
Figure 1 — GSIM Global Station Overview (4 separate figures, ESSD style)
(a) All GSIM stations worldwide
(b) Anchor / Target / Insufficient classification
(c) Missing-fraction histogram by category
(d) Missing-fraction CDF by category
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(r"D:\1-Phd_work\1-Gobel_hypower_predict\Data\999-论文\GSIM全球插补1995-2015")
ATTR = ROOT / "999 material" / "GSIM_attribute.csv"
ALL_STATIONS = ROOT / "01_Station_Selection" / "all_stations_1995_2015.csv"
FEAT = ROOT / "02_Feature_Table" / "station_features_with_meteo.csv"
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

# ── colors (NPG style) ──────────────────────────────────────────────────
C_ANC = "#00A087"   # Anchor — green
C_TAR = "#E64B35"   # Target — red
C_INS = "#8491B4"   # Insufficient — grey-blue

# ── data ─────────────────────────────────────────────────────────────────
attr = pd.read_csv(ATTR)
all_st = pd.read_csv(ALL_STATIONS)
feat = pd.read_csv(FEAT)

all_st["missing_frac"] = 1 - all_st["completeness"]
coord = attr[["gsim.no", "latitude", "longitude"]].rename(columns={"gsim.no": "station_id"})

anchor_st = all_st[all_st["category"] == "anchor"]
target_st = all_st[all_st["category"] == "target"]
insuff_st = all_st[all_st["category"] == "insufficient"]

proj = ccrs.Robinson()
pc = ccrs.PlateCarree()

def style_map(ax):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#999999", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#cccccc", zorder=1)

# ── (a) All GSIM stations ───────────────────────────────────────────────
fig_a, ax_a = plt.subplots(figsize=(180 / 25.4, 90 / 25.4), dpi=300,
                            subplot_kw={"projection": proj})
style_map(ax_a)
ax_a.scatter(
    attr["longitude"], attr["latitude"],
    s=0.4, c="#3C5488", alpha=0.35, linewidths=0,
    transform=pc, zorder=2, rasterized=True,
)
ax_a.text(
    0.02, 0.06, f"All GSIM stations (n = {len(attr):,})",
    transform=ax_a.transAxes, fontsize=6.5, color="#333333",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.3, alpha=0.9),
)
fig_a.savefig(str(OUTDIR / "Fig1a_all_stations.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig_a.savefig(str(OUTDIR / "Fig1a_all_stations.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_a)
print("Saved Fig1a")

# ── (b) Anchor / Target / Insufficient map ──────────────────────────────
fig_b, ax_b = plt.subplots(figsize=(180 / 25.4, 90 / 25.4), dpi=300,
                            subplot_kw={"projection": proj})
style_map(ax_b)

insuff_geo = insuff_st.merge(coord, on="station_id", how="left").dropna(subset=["latitude"])
anchor_geo = anchor_st.merge(coord, on="station_id", how="left").dropna(subset=["latitude"])
target_geo = target_st.merge(coord, on="station_id", how="left").dropna(subset=["latitude"])

ax_b.scatter(insuff_geo["longitude"], insuff_geo["latitude"],
             s=0.4, c=C_INS, alpha=0.25, linewidths=0, transform=pc, zorder=2,
             label=f"Insufficient (n = {len(insuff_st):,})", rasterized=True)
ax_b.scatter(anchor_geo["longitude"], anchor_geo["latitude"],
             s=0.5, c=C_ANC, alpha=0.45, linewidths=0, transform=pc, zorder=3,
             label=f"Anchor (n = {len(anchor_st):,})", rasterized=True)
ax_b.scatter(target_geo["longitude"], target_geo["latitude"],
             s=0.5, c=C_TAR, alpha=0.45, linewidths=0, transform=pc, zorder=4,
             label=f"Target (n = {len(target_st):,})", rasterized=True)
ax_b.legend(loc="lower left", markerscale=10, handletextpad=0.3,
            borderpad=0.4, labelspacing=0.3, fontsize=6)

fig_b.savefig(str(OUTDIR / "Fig1b_anchor_target.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig_b.savefig(str(OUTDIR / "Fig1b_anchor_target.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_b)
print("Saved Fig1b")

# ── (c) Histogram ───────────────────────────────────────────────────────
fig_c, ax_c = plt.subplots(figsize=(130 / 25.4, 55 / 25.4), dpi=300)

bins = np.linspace(0, 1, 41)
ax_c.hist(anchor_st["missing_frac"], bins=bins, color=C_ANC, alpha=0.75,
          label=f"Anchor (n = {len(anchor_st):,})", edgecolor="white", linewidth=0.3)
ax_c.hist(target_st["missing_frac"], bins=bins, color=C_TAR, alpha=0.75,
          label=f"Target (n = {len(target_st):,})", edgecolor="white", linewidth=0.3)
ax_c.hist(insuff_st["missing_frac"], bins=bins, color=C_INS, alpha=0.55,
          label=f"Insufficient (n = {len(insuff_st):,})", edgecolor="white", linewidth=0.3)

ax_c.axvline(0.1, color=C_ANC, ls="--", lw=0.7, alpha=0.8)
ax_c.axvline(0.7, color=C_TAR, ls="--", lw=0.7, alpha=0.8)
ax_c.text(0.105, ax_c.get_ylim()[1] * 0.93, "10%", fontsize=5.5, color=C_ANC, va="top")
ax_c.text(0.705, ax_c.get_ylim()[1] * 0.93, "70%", fontsize=5.5, color=C_TAR, va="top")

ax_c.set_xlabel("Missing fraction (1995–2015)")
ax_c.set_ylabel("Number of stations")
ax_c.legend(loc="upper center", fontsize=6)
ax_c.set_xlim(-0.02, 1.02)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

fig_c.savefig(str(OUTDIR / "Fig1c_histogram.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig_c.savefig(str(OUTDIR / "Fig1c_histogram.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_c)
print("Saved Fig1c")

# ── (d) CDF ─────────────────────────────────────────────────────────────
fig_d, ax_d = plt.subplots(figsize=(130 / 25.4, 55 / 25.4), dpi=300)

for data, label, color, lw in [
    (all_st["missing_frac"], f"All (n = {len(all_st):,})", "#333333", 1.2),
    (anchor_st["missing_frac"], f"Anchor (n = {len(anchor_st):,})", C_ANC, 1.0),
    (target_st["missing_frac"], f"Target (n = {len(target_st):,})", C_TAR, 1.0),
    (insuff_st["missing_frac"], f"Insufficient (n = {len(insuff_st):,})", C_INS, 1.0),
]:
    sorted_vals = np.sort(data.values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax_d.plot(sorted_vals, cdf, color=color, lw=lw, label=label)

ax_d.axvline(0.1, color=C_ANC, ls=":", lw=0.5, alpha=0.6)
ax_d.axvline(0.7, color=C_TAR, ls=":", lw=0.5, alpha=0.6)
ax_d.set_xlabel("Missing fraction (1995–2015)")
ax_d.set_ylabel("Cumulative proportion")
ax_d.legend(loc="center right", fontsize=6)
ax_d.set_xlim(-0.02, 1.02)
ax_d.set_ylim(-0.02, 1.02)
ax_d.spines["top"].set_visible(False)
ax_d.spines["right"].set_visible(False)

fig_d.savefig(str(OUTDIR / "Fig1d_cdf.png"), dpi=300, bbox_inches="tight", facecolor="white")
fig_d.savefig(str(OUTDIR / "Fig1d_cdf.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig_d)
print("Saved Fig1d")

print(f"\nAll Fig1 saved to {OUTDIR}")
