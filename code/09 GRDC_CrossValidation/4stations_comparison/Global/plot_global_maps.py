"""
Generate global R and NSE maps for GRDC validation
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[4])).resolve()
DATADIR = ROOT / "09 GRDC交叉验证" / "4stations_comparison" / "Global"
GSIM_ATTR = ROOT / "999 material" / "GSIM_attribute.csv"
OUTDIR = DATADIR

gsim_attr = pd.read_csv(GSIM_ATTR)
gsim_attr = gsim_attr[['gsim.no', 'latitude', 'longitude']].rename(
    columns={'gsim.no': 'gsim_id', 'latitude': 'lat', 'longitude': 'lon'})

csv_files = sorted(list(DATADIR.glob("*_comparison.csv")))
results = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    fill_mask = df['fill_method'] != 'OBSERVED'

    if fill_mask.sum() >= 1:
        gsim_v = df.loc[fill_mask, 'gsim_filled'].values
        grdc_v = df.loc[fill_mask, 'grdc_observed'].values

        r = np.corrcoef(gsim_v, grdc_v)[0, 1] if len(gsim_v) > 1 else 0
        nse = 1 - np.sum((gsim_v - grdc_v)**2) / np.sum((grdc_v - np.mean(grdc_v))**2)

        station_name = csv_file.stem.replace('_comparison', '')
        gsim_id = station_name.split('_')[0] + '_' + station_name.split('_')[1]

        results.append({'gsim_id': gsim_id, 'r': r, 'nse': nse})

results_df = pd.DataFrame(results)
results_df = results_df.merge(gsim_attr, on='gsim_id', how='left')
results_df = results_df.dropna(subset=['lat', 'lon'])

print(f"Total stations: {len(results_df)}")

# Load region info
all_stations = []
for continent in ['AS', 'AU', 'SA', 'AF']:
    summary_file = ROOT / "09 GRDC交叉验证" / "GRDC" / f"{continent}-GRDC" / "plots" / f"{continent}_summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df['region'] = continent
        all_stations.append(df[['gsim_id', 'region']])

for region in ['Europe', 'USA']:
    summary_file = ROOT / "09 GRDC交叉验证" / region / "validation_summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df['region'] = region
        all_stations.append(df[['gsim_id', 'region']])
        print(f"Loaded {region}: {len(df)} stations")

region_df = pd.concat(all_stations, ignore_index=True)
results_df = results_df.merge(region_df, on='gsim_id', how='left')

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
})

# NSE Map
fig = plt.figure(figsize=(220/25.4, 110/25.4), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#e0f3ff', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#666', zorder=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#999', linestyle=':', zorder=1)

nse_vals = results_df['nse'].values
colors = np.where(nse_vals >= 0.5, '#2E7D32',
         np.where(nse_vals >= 0, '#FFA726', '#D32F2F'))

ax.scatter(results_df['lon'], results_df['lat'], c=colors, s=10, alpha=0.7,
           edgecolors='white', linewidths=0.3, transform=ccrs.PlateCarree(), zorder=2)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E7D32', label=f'NSE ≥ 0.5 (n={np.sum(nse_vals >= 0.5)})'),
    Patch(facecolor='#FFA726', label=f'0 ≤ NSE < 0.5 (n={np.sum((nse_vals >= 0) & (nse_vals < 0.5))})'),
    Patch(facecolor='#D32F2F', label=f'NSE < 0 (n={np.sum(nse_vals < 0)})')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=7, frameon=True,
          facecolor='white', edgecolor='#ccc', framealpha=0.9)

# Add region statistics for NSE
stats_text = "Region Statistics:\n"
for region in ['AS', 'AU', 'SA', 'AF', 'Europe', 'USA']:
    subset = results_df[results_df['region'] == region]
    if len(subset) > 0:
        nse_vals_region = subset['nse'].values
        good_pct = 100 * (nse_vals_region >= 0.5).sum() / len(subset)
        stats_text += f"{region}: n={len(subset)}, NSE≥0.5: {good_pct:.1f}%\n"

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=6.5,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', lw=0.3, alpha=0.95))

ax.set_global()
ax.set_title('Global GRDC Validation: NSE Performance', fontsize=11, pad=10)

plt.tight_layout()
fig.savefig(OUTDIR / "global_NSE_map.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved global_NSE_map.png")

# R Map
fig = plt.figure(figsize=(220/25.4, 110/25.4), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#e0f3ff', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#666', zorder=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#999', linestyle=':', zorder=1)

r_vals = results_df['r'].values
colors = np.where(r_vals >= 0.7, '#2E7D32',
         np.where(r_vals >= 0.5, '#FFA726', '#D32F2F'))

ax.scatter(results_df['lon'], results_df['lat'], c=colors, s=10, alpha=0.7,
           edgecolors='white', linewidths=0.3, transform=ccrs.PlateCarree(), zorder=2)

legend_elements = [
    Patch(facecolor='#2E7D32', label=f'R ≥ 0.7 (n={np.sum(r_vals >= 0.7)})'),
    Patch(facecolor='#FFA726', label=f'0.5 ≤ R < 0.7 (n={np.sum((r_vals >= 0.5) & (r_vals < 0.7))})'),
    Patch(facecolor='#D32F2F', label=f'R < 0.5 (n={np.sum(r_vals < 0.5)})')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=7, frameon=True,
          facecolor='white', edgecolor='#ccc', framealpha=0.9)

# Add region statistics for R
stats_text = "Region Statistics:\n"
for region in ['AS', 'AU', 'SA', 'AF', 'Europe', 'USA']:
    subset = results_df[results_df['region'] == region]
    if len(subset) > 0:
        r_vals_region = subset['r'].values
        good_pct = 100 * (r_vals_region >= 0.7).sum() / len(subset)
        stats_text += f"{region}: n={len(subset)}, R≥0.7: {good_pct:.1f}%\n"

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=6.5,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', lw=0.3, alpha=0.95))

ax.set_global()
ax.set_title('Global GRDC Validation: Correlation (R) Performance', fontsize=11, pad=10)

plt.tight_layout()
fig.savefig(OUTDIR / "global_R_map.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved global_R_map.png")
