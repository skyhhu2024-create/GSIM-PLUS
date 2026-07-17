"""
GRDC cross-validation for the current GSIM target product.

This version compares GRDC observations against the latest target-station
outputs in 08_GSIM_PLUS_Product/GSIM_fill, using the same plotting style as
validate_all_stations.py.
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import geopandas as gpd


ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[3])).resolve()
EXTERNAL = ROOT / "external_data" / "GRDC" / "USA"
GRDC_DIR = Path(os.environ.get("GSIM_PLUS_USA_GRDC_DIR", EXTERNAL / "parsed_csv"))
GSIM_FILL = ROOT / "08_GSIM_PLUS_Product" / "GSIM_fill"
OUTROOT = ROOT / "09 GRDC交叉验证" / "USA_current_target"
OUTDIR = OUTROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)


gdf1 = gpd.read_file(Path(os.environ.get("GSIM_PLUS_USA_GRDC_SHP", EXTERNAL / "grdc_stations.shp")))
gdf2 = gpd.read_file(Path(os.environ.get("GSIM_PLUS_USA_GSIM_SHP", EXTERNAL / "gsim_stations.shp")))

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
        match_pairs.append(
            {
                "grdc_id": str(gdf1.iloc[i]["station_id"]),
                "grdc_name": str(gdf1.iloc[i]["station"]),
                "gsim_id": str(gdf2.iloc[best_j]["station_id"]),
            }
        )
match_df = pd.DataFrame(match_pairs)


plt.rcParams.update(
    {
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
    }
)


def compute_metrics(pred, obs):
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    mae = np.mean(np.abs(pred - obs))
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    if np.all(obs == 0):
        pbias, nse = np.nan, np.nan
    else:
        pbias = np.mean((pred - obs) / np.where(obs == 0, 1e-6, obs)) * 100
        ss_res = np.sum((pred - obs) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r = np.corrcoef(pred, obs)[0, 1] if len(pred) > 1 else np.nan
    return mae, rmse, pbias, nse, r


summary = []

for _, row in match_df.iterrows():
    grdc_id = row["grdc_id"]
    gsim_id = row["gsim_id"]
    grdc_name = row["grdc_name"]

    grdc_path = GRDC_DIR / f"{grdc_id}.csv"
    gsim_path = GSIM_FILL / f"{gsim_id}.csv"
    if not grdc_path.exists() or not gsim_path.exists():
        continue

    grdc = pd.read_csv(grdc_path)
    grdc["date"] = pd.to_datetime(grdc["data"])
    grdc = grdc[["date", "MEAN"]].dropna(subset=["MEAN"])
    grdc = grdc[(grdc["date"] >= "1995-01-01") & (grdc["date"] <= "2015-12-31")]
    grdc["ym"] = grdc["date"].dt.to_period("M")

    gsim = pd.read_csv(gsim_path)
    gsim["date"] = pd.to_datetime(gsim["date"])
    gsim["ym"] = gsim["date"].dt.to_period("M")

    merged = gsim.merge(grdc[["ym", "MEAN"]], on="ym", how="inner")
    fill_mask = merged["fill_method"] != "OBSERVED"
    if fill_mask.sum() == 0:
        continue

    all_mae, all_rmse, all_pbias, all_nse, all_r = compute_metrics(
        merged["final_streamflow"].values,
        merged["MEAN"].values,
    )
    fill_mae, fill_rmse, fill_pbias, fill_nse, fill_r = compute_metrics(
        merged.loc[fill_mask, "final_streamflow"].values,
        merged.loc[fill_mask, "MEAN"].values,
    )

    fill_dates = merged.loc[fill_mask, "date"]
    t_min = fill_dates.min() - pd.DateOffset(months=12)
    t_max = fill_dates.max() + pd.DateOffset(months=12)
    zoom = merged[(merged["date"] >= t_min) & (merged["date"] <= t_max)]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(180 / 25.4, 100 / 25.4),
        dpi=300,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    ax1.plot(
        zoom["date"],
        zoom["MEAN"],
        color="#3C5488",
        lw=1.0,
        alpha=0.9,
        label="GRDC observed",
        zorder=2,
    )
    ax1.plot(
        zoom["date"],
        zoom["final_streamflow"],
        color="#E64B35",
        lw=1.0,
        alpha=0.9,
        label="GSIM-PLUS",
        zorder=3,
    )

    for _, r in merged.loc[fill_mask].iterrows():
        ax1.axvspan(
            r["date"] - pd.Timedelta(days=15),
            r["date"] + pd.Timedelta(days=15),
            facecolor="#E64B3520",
            edgecolor="none",
            zorder=0,
        )

    ax1.set_ylabel(r"Streamflow (m$^3$/s)", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", fontsize=6.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.set_xticklabels([])

    line1 = f"{grdc_name} (GRDC:{grdc_id} / GSIM:{gsim_id})"
    line2 = f"All: R={all_r:.3f}, NSE={all_nse:.3f}, RMSE={all_rmse:.1f}"
    line3 = f"Filled(n={fill_mask.sum()}): MAE={fill_mae:.1f}, RMSE={fill_rmse:.1f}, PBias={fill_pbias:.1f}%"
    ax1.text(
        0.01,
        0.95,
        f"{line1}\n{line2}\n{line3}",
        transform=ax1.transAxes,
        fontsize=5.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.3, alpha=0.9),
    )

    residual = zoom["final_streamflow"].values - zoom["MEAN"].values
    colors = np.where(residual >= 0, "#E64B35", "#3C5488")
    ax2.bar(zoom["date"], residual, width=25, color=colors, alpha=0.7, linewidth=0)
    ax2.axhline(0, color="black", lw=0.3)
    ax2.set_ylabel("Residual", fontsize=7)
    ax2.set_xlabel("Date", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    safe_name = (
        grdc_name.replace(" ", "_")
        .replace("/", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )
    fname = f"val_{grdc_id}_{gsim_id}_{safe_name}.png"
    fig.savefig(str(OUTDIR / fname), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary.append(
        {
            "grdc_id": grdc_id,
            "gsim_id": gsim_id,
            "name": grdc_name,
            "n_total": int(len(merged)),
            "n_filled": int(fill_mask.sum()),
            "all_R": all_r,
            "all_NSE": all_nse,
            "all_RMSE": all_rmse,
            "fill_MAE": fill_mae,
            "fill_RMSE": fill_rmse,
            "fill_PBias": fill_pbias,
            "fill_NSE": fill_nse,
            "fill_R": fill_r,
            "mean_grdc": float(np.mean(merged.loc[fill_mask, "MEAN"].values)),
            "mean_gsim": float(np.mean(merged.loc[fill_mask, "final_streamflow"].values)),
            "plot_file": fname,
        }
    )
    print(f"Saved {fname}")


sdf = pd.DataFrame(summary).sort_values("fill_NSE", ascending=False)
sdf.to_csv(str(OUTROOT / "validation_summary_current_target.csv"), index=False)
print(f"\nSaved {len(sdf)} stations to {OUTROOT / 'validation_summary_current_target.csv'}")
