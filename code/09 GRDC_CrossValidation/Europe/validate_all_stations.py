"""
GRDC Cross-Validation: All 15 stations with overlap
Generate per-station time series comparison plots.
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import geopandas as gpd

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[3])).resolve()
EXTERNAL = ROOT / "external_data" / "GRDC" / "Europe"
GRDC_DIR = Path(os.environ.get("GSIM_PLUS_EU_GRDC_DIR", EXTERNAL / "parsed_csv"))
GSIM_FILL_TARGET = ROOT / "08_GSIM_PLUS_Product" / "GSIM_fill_target"
GSIM_FILL_ANCHOR = ROOT / "08_GSIM_PLUS_Product" / "GSIM_fill_anchor"
OUTDIR = ROOT / "09 GRDC交叉验证" / "Europe" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── build matching table ─────────────────────────────────────────────────
gdf1 = gpd.read_file(Path(os.environ.get("GSIM_PLUS_EU_GRDC_SHP", EXTERNAL / "grdc_stations.shp")))
gdf2 = gpd.read_file(Path(os.environ.get("GSIM_PLUS_EU_GSIM_SHP", EXTERNAL / "gsim_stations.shp")))

coords1 = np.array([(g.x, g.y) for g in gdf1.geometry])
coords2 = np.array([(g.x, g.y) for g in gdf2.geometry])

THRESH = 0.01
match_pairs = []
for i, (x1, y1) in enumerate(coords1):
    best_j, best_d = -1, 999
    for j, (x2, y2) in enumerate(coords2):
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist < THRESH and dist < best_d:
            best_j, best_d = j, dist
    if best_j >= 0:
        match_pairs.append({
            "grdc_id": str(gdf1.iloc[i]["station_id"]),
            "grdc_name": str(gdf1.iloc[i]["station"]),
            "gsim_id": str(gdf2.iloc[best_j]["station_id"]),
        })
match_df = pd.DataFrame(match_pairs)

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

# ── process each station ─────────────────────────────────────────────────
summary = []

for _, row in match_df.iterrows():
    grdc_id, gsim_id, grdc_name = row["grdc_id"], row["gsim_id"], row["grdc_name"]

    grdc_path = GRDC_DIR / f"{grdc_id}.csv"
    if not grdc_path.exists():
        continue
    grdc = pd.read_csv(grdc_path)
    grdc["date"] = pd.to_datetime(grdc["data"])
    grdc = grdc[["date", "MEAN"]].dropna(subset=["MEAN"])
    grdc = grdc[(grdc["date"] >= "1995-01-01") & (grdc["date"] <= "2015-12-31")]
    grdc["ym"] = grdc["date"].dt.to_period("M")

    gsim_path = GSIM_FILL_TARGET / f"{gsim_id}.csv"
    if not gsim_path.exists():
        gsim_path = GSIM_FILL_ANCHOR / f"{gsim_id}.csv"
    if not gsim_path.exists():
        continue
    gsim = pd.read_csv(gsim_path)
    gsim["date"] = pd.to_datetime(gsim["date"])
    gsim["ym"] = gsim["date"].dt.to_period("M")

    merged = gsim.merge(grdc[["ym", "MEAN"]], on="ym", how="inner")
    fill_mask = merged["fill_method"] != "OBSERVED"

    if fill_mask.sum() == 0:
        continue

    # metrics on filled points only
    gsim_v = merged.loc[fill_mask, "final_streamflow"].values
    grdc_v = merged.loc[fill_mask, "MEAN"].values

    mae = np.mean(np.abs(gsim_v - grdc_v))
    rmse = np.sqrt(np.mean((gsim_v - grdc_v) ** 2))
    if np.all(grdc_v == 0):
        pbias = np.nan
        nse_fill = np.nan
    else:
        pbias = np.mean((gsim_v - grdc_v) / np.where(grdc_v == 0, 1e-6, grdc_v)) * 100
        ss_res = np.sum((gsim_v - grdc_v) ** 2)
        ss_tot = np.sum((grdc_v - np.mean(grdc_v)) ** 2)
        nse_fill = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # correlation on filled points
    r_fill = np.corrcoef(gsim_v, grdc_v)[0, 1] if len(gsim_v) > 1 else np.nan

    summary.append({
        "grdc_id": grdc_id, "gsim_id": gsim_id, "name": grdc_name,
        "n_filled": int(fill_mask.sum()),
        "mae": mae, "rmse": rmse, "pbias": pbias,
        "nse_fill": nse_fill, "r_fill": r_fill,
        "mean_grdc": np.mean(grdc_v), "mean_gsim": np.mean(gsim_v),
    })

    # ── plot ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(180 / 25.4, 100 / 25.4), dpi=300,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # top: time series (zoom to +/- 12 months around filled period)
    fill_dates = merged.loc[fill_mask, "date"]
    t_min = fill_dates.min() - pd.DateOffset(months=12)
    t_max = fill_dates.max() + pd.DateOffset(months=12)
    zoom = merged[(merged["date"] >= t_min) & (merged["date"] <= t_max)]

    ax1.plot(zoom["date"], zoom["MEAN"],
             color="#3C5488", lw=1.0, alpha=0.9, label="GRDC observed", zorder=2)
    ax1.plot(zoom["date"], zoom["final_streamflow"],
             color="#E64B35", lw=1.0, alpha=0.9, label="GSIM-PLUS", zorder=3)

    for _, r in merged[fill_mask].iterrows():
        ax1.axvspan(r["date"] - pd.Timedelta(days=15),
                     r["date"] + pd.Timedelta(days=15),
                     facecolor="#E64B3520", edgecolor="none", zorder=0)

    ax1.set_ylabel(r"Streamflow (m$^3$/s)", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", fontsize=6.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.set_xticklabels([])

    metrics_str = f"n={fill_mask.sum()}"
    if not np.isnan(r_fill):
        metrics_str += f", R={r_fill:.3f}"
    if not np.isnan(nse_fill):
        metrics_str += f", NSE={nse_fill:.3f}"
    metrics_str += f", RMSE={rmse:.1f}"

    ax1.text(0.01, 0.95,
             f"{grdc_name} (GRDC:{grdc_id} / GSIM:{gsim_id})\n{metrics_str}",
             transform=ax1.transAxes, fontsize=6, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.3, alpha=0.9))

    # bottom: residual (zoomed)
    zoom_fill = zoom.copy()
    residual = zoom_fill["final_streamflow"].values - zoom_fill["MEAN"].values
    colors = np.where(residual >= 0, "#E64B35", "#3C5488")
    ax2.bar(zoom_fill["date"], residual, width=25, color=colors, alpha=0.7, linewidth=0)
    ax2.axhline(0, color="black", lw=0.3)
    ax2.set_ylabel("Residual", fontsize=7)
    ax2.set_xlabel("Date", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fname = f"val_{grdc_id}_{gsim_id}_{grdc_name.replace(' ', '_').replace('/', '_')}"
    fig.savefig(str(OUTDIR / f"{fname}.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {fname}")

# ── summary table ────────────────────────────────────────────────────────
sdf = pd.DataFrame(summary).sort_values("nse_fill", ascending=False)
print("\n=== Summary (sorted by NSE on filled points) ===")
print(sdf.to_string(index=False))
sdf.to_csv(str(OUTDIR.parent / "validation_summary.csv"), index=False)
print("\nSaved validation_summary.csv")
