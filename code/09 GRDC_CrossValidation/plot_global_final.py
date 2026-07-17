"""
Final global GRDC validation map with all continents including updated Europe and USA
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[2])).resolve()
GRDC_BASE = ROOT / "09 GRDC交叉验证"
OUTDIR = GRDC_BASE
GSIM_ATTR = ROOT / "999 material" / "GSIM_attribute.csv"

gsim_attr = pd.read_csv(GSIM_ATTR)
gsim_attr = gsim_attr[['gsim.no', 'latitude', 'longitude']].rename(
    columns={'gsim.no': 'gsim_id', 'latitude': 'lat', 'longitude': 'lon'})

all_results = []

# Load 4 continents
for continent in ['AS', 'AU', 'SA', 'AF']:
    summary_file = GRDC_BASE / "GRDC" / f"{continent}-GRDC" / "plots" / f"{continent}_summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df['continent'] = continent
        all_results.append(df)
        print(f"Loaded {continent}: {len(df)} stations")

# Load Europe and USA
for region in ['Europe', 'USA']:
    summary_file = GRDC_BASE / region / "validation_summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df = df.rename(columns={'fill_NSE': 'nse_fill', 'fill_R': 'r_fill', 'n_filled': 'n_filled'})
        df['continent'] = region
        all_results.append(df)
        print(f"Loaded {region}: {len(df)} stations")

results = pd.concat(all_results, ignore_index=True)
results = results.merge(gsim_attr, on='gsim_id', how='left')
results = results[np.isfinite(results['nse_fill'])]

print(f"\nTotal stations: {len(results)}")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
})

fig = plt.figure(figsize=(220/25.4, 120/25.4), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#e0f3ff', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#666', zorder=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#999', linestyle=':', zorder=1)

nse_vals = results['nse_fill'].values
colors = np.where(nse_vals >= 0.5, '#2E7D32',
         np.where(nse_vals >= 0, '#FFA726', '#D32F2F'))

ax.scatter(results['lon'], results['lat'], c=colors, s=8, alpha=0.7,
           edgecolors='white', linewidths=0.3,
           transform=ccrs.PlateCarree(), zorder=2)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E7D32', label=f'NSE ≥ 0.5 (n={np.sum(nse_vals >= 0.5)})'),
    Patch(facecolor='#FFA726', label=f'0 ≤ NSE < 0.5 (n={np.sum((nse_vals >= 0) & (nse_vals < 0.5))})'),
    Patch(facecolor='#D32F2F', label=f'NSE < 0 (n={np.sum(nse_vals < 0)})')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=7, frameon=True,
          facecolor='white', edgecolor='#ccc', framealpha=0.9)

# Continent statistics
stats_text = "Continent Statistics:\n"
for continent in ['AS', 'AU', 'SA', 'AF', 'Europe', 'USA']:
    subset = results[results['continent'] == continent]
    if len(subset) > 0:
        nse = subset['nse_fill']
        good_pct = 100 * (nse > 0.5).sum() / len(subset)
        stats_text += f"{continent}: n={len(subset)}, NSE>0.5: {good_pct:.1f}%, median={nse.median():.2f}\n"

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=6.5,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', lw=0.3, alpha=0.95))

ax.set_global()
ax.set_title('GRDC Cross-Validation: GSIM-PLUS Performance (All Continents)', fontsize=11, pad=10)

plt.tight_layout()
fig.savefig(OUTDIR / "global_validation_final.png", dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved to {OUTDIR / 'global_validation_final.png'}")

