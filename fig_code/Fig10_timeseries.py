"""
Figure 10 — Representative Station Time Series (Before/After Gap-Filling)
  5 stations from different regions and climate zones
  Gray = observed, colored = MAML-filled, background shading by quality flag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__import__("os").environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
FILL_DIR = ROOT / "08_GSIM_PLUS_Product" / "dtrr_guarded" / "GSIM_fill"
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

# ── config ───────────────────────────────────────────────────────────────
STATIONS = [
    ("AU_0002448", "Australia (Temperate, C)"),
    ("BR_0001432", "South America (Tropical, A)"),
    ("CA_0003714", "North America (Continental, D)"),
    ("US_0009172", "Alaska (Continental, D)"),
    ("AU_0002656", "Australia (Arid, B)"),
]

Q_BG_COLORS = {
    "Q1": "#56B4E930",   # sky blue
    "Q2": "#E69F0030",   # orange
    "Q3": "#CC79A730",   # reddish purple
}

# ── plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(STATIONS), 1, figsize=(180 / 25.4, 145 / 25.4), dpi=600)

for idx, (station_id, region_label) in enumerate(STATIONS):
    ax = axes[idx]
    fpath = FILL_DIR / f"{station_id}.csv"
    df = pd.read_csv(fpath)
    df["date"] = pd.to_datetime(df["date"])

    obs_mask = df["fill_method"] == "OBSERVED"
    fill_mask = df["fill_method"] != "OBSERVED"

    # background shading for filled segments by quality flag
    for _, row in df[fill_mask].iterrows():
        qf = row["quality_flag"]
        if qf in Q_BG_COLORS:
            ax.axvspan(row["date"] - pd.Timedelta(days=15),
                       row["date"] + pd.Timedelta(days=15),
                       facecolor=Q_BG_COLORS[qf], edgecolor="none", zorder=0)

    # observed line — break at gaps (insert NaN where filled)
    obs_series = df["final_streamflow"].copy()
    obs_series[fill_mask] = np.nan
    ax.plot(df["date"], obs_series,
            color="#333333", lw=0.8, alpha=0.9, zorder=2)

    # filled segments — connect with line + markers
    # group consecutive filled points into segments
    fill_df = df[fill_mask].copy()
    if len(fill_df) > 0:
        fill_idx = fill_df.index.values
        segments = np.split(fill_idx, np.where(np.diff(fill_idx) > 1)[0] + 1)
        for seg in segments:
            # extend one point before and after to connect to observed line
            i_start = max(0, seg[0] - 1)
            i_end = min(len(df) - 1, seg[-1] + 1)
            seg_ext = list(range(i_start, i_end + 1))
            seg_df = df.iloc[seg_ext]
            ax.plot(seg_df["date"], seg_df["final_streamflow"],
                    color="#0072B2", lw=1.0, ls="--", alpha=0.95, zorder=3)
        # markers only on filled points
        ax.plot(fill_df["date"], fill_df["final_streamflow"],
                color="#0072B2", lw=0, marker="o", markersize=2.2,
                alpha=0.9, zorder=4)

    # label
    n_filled = fill_mask.sum()
    n_total = len(df)
    ax.text(0.01, 0.92, f"{station_id} — {region_label}\n"
            f"Filled: {n_filled}/{n_total} months ({n_filled/n_total*100:.0f}%)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.3, alpha=0.9))

    ax.set_ylabel(r"Q (m$^3$/s)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if idx < len(STATIONS) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Year", fontsize=10)

# shared legend at top
legend_elements = [
    plt.Line2D([0], [0], color="#333333", lw=1.0, label="Observed"),
    plt.Line2D([0], [0], color="#0072B2", lw=1.0, ls="--", marker="o",
               markersize=3, label="GSIM-PLUS filled"),
    Patch(facecolor="#56B4E950", label="Q1 (≤ 3 months)"),
    Patch(facecolor="#E69F0050", label="Q2 (4–24 months)"),
    Patch(facecolor="#CC79A750", label="Q3 (> 24 months)"),
]
fig.legend(handles=legend_elements, loc="upper center",
           bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=8,
           handlelength=1.2, handletextpad=0.3, columnspacing=1.0)

fig.subplots_adjust(hspace=0.25)
fig.savefig(str(OUTDIR / "Fig10_timeseries.png"), dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(str(OUTDIR / "Fig10_timeseries.pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved Fig10_timeseries")

