"""
Figure 9d — Completeness Improvement: Full Dataset vs Target Stations
Dual CDF comparison to highlight target station improvement
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__import__("os").environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
TARGET_DIR = ROOT / "08_GSIM_PLUS_Product" / "dtrr_guarded" / "GSIM_fill"
ANCHOR_DIR = ROOT / "08_GSIM_PLUS_Product" / "DTRR_Guarded_Anchor" / "GSIM_fill_anchor"
OUTDIR = ROOT / "111-paper"
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8, "legend.frameon": False,
    "mathtext.fontset": "stix",
})

# Load data
print("Loading target + anchor data...")
dfs = []
for src_dir, label in [(TARGET_DIR, "target"), (ANCHOR_DIR, "anchor")]:
    for fp in src_dir.glob("*.csv"):
        tmp = pd.read_csv(fp)
        tmp["station_type"] = label
        dfs.append(tmp)
df = pd.concat(dfs, ignore_index=True)

# Calculate per-station completeness
per_station = df.groupby(["station_id", "station_type"]).agg(
    observed=("fill_method", lambda x: (x == "OBSERVED").sum()),
    total=("date", "count")
).reset_index()
per_station["pct_before"] = per_station["observed"] / 252 * 100
per_station["pct_after"] = per_station["total"] / 252 * 100

# Split by station type
target_stations = per_station[per_station["station_type"] == "target"]
full_dataset = per_station

print(f"Target stations: {len(target_stations)}")
print(f"Full dataset: {len(full_dataset)}")

# Create dual CDF plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180/25.4, 68/25.4), dpi=600)

# Left panel: Full dataset
for data, label, color, ls in [
    (full_dataset["pct_before"], "Before gap-filling", "#0072B2", "--"),
    (full_dataset["pct_after"], "After gap-filling", "#E69F00", "-"),
]:
    sorted_d = np.sort(data.values)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax1.plot(sorted_d, cdf, color=color, ls=ls, lw=1.5, label=label)

median_before_full = full_dataset["pct_before"].median()
median_after_full = full_dataset["pct_after"].median()
ax1.axvline(median_before_full, color="#0072B2", ls=":", lw=0.8, alpha=0.7)
ax1.axvline(median_after_full, color="#E69F00", ls=":", lw=0.8, alpha=0.7)
ax1.text(median_before_full - 1, 0.55, f"Median\n{median_before_full:.1f}%",
         fontsize=8, ha="right", color="#0072B2")
ax1.text(median_after_full + 1, 0.45, f"Median\n{median_after_full:.1f}%",
         fontsize=8, ha="left", color="#E69F00")

ax1.set_xlabel("Data completeness (%)", fontsize=10)
ax1.set_ylabel("CDF", fontsize=10)
ax1.set_xlim(25, 105)
ax1.set_ylim(0, 1.02)
ax1.set_title("(a) Full dataset (16,054 stations)", fontsize=10, loc="left")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(loc="lower right", fontsize=8)

# Right panel: Target stations only
for data, label, color, ls in [
    (target_stations["pct_before"], "Before gap-filling", "#0072B2", "--"),
    (target_stations["pct_after"], "After gap-filling", "#E69F00", "-"),
]:
    sorted_d = np.sort(data.values)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax2.plot(sorted_d, cdf, color=color, ls=ls, lw=1.5, label=label)

median_before_target = target_stations["pct_before"].median()
median_after_target = target_stations["pct_after"].median()
ax2.axvline(median_before_target, color="#0072B2", ls=":", lw=0.8, alpha=0.7)
ax2.axvline(median_after_target, color="#E69F00", ls=":", lw=0.8, alpha=0.7)
ax2.text(median_before_target - 1, 0.55, f"Median\n{median_before_target:.1f}%",
         fontsize=8, ha="right", color="#0072B2")
ax2.text(median_after_target + 1, 0.45, f"Median\n{median_after_target:.1f}%",
         fontsize=8, ha="left", color="#E69F00")

ax2.set_xlabel("Data completeness (%)", fontsize=10)
ax2.set_ylabel("CDF", fontsize=10)
ax2.set_xlim(25, 105)
ax2.set_ylim(0, 1.02)
ax2.set_title("(b) Target stations (8,731 stations)", fontsize=10, loc="left")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(loc="lower right", fontsize=8)

plt.tight_layout()
fig.savefig(OUTDIR / "Fig9d_completeness_dual_cdf.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(OUTDIR / "Fig9d_completeness_dual_cdf.pdf", bbox_inches="tight")
plt.close(fig)

print("\n=== Summary ===")
print(f"Full dataset: {median_before_full:.1f}% → {median_after_full:.1f}% (+{median_after_full-median_before_full:.1f}%)")
print(f"Target stations: {median_before_target:.1f}% → {median_after_target:.1f}% (+{median_after_target-median_before_target:.1f}%)")
print("\nSaved: Fig9d_completeness_dual_cdf")




