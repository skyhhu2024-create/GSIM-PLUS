"""
GRDC Cross-Validation - Match by coordinates
Generate plots for all GRDC stations with GSIM overlap
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os
import sys

CONTINENT = sys.argv[1] if len(sys.argv) > 1 else "AU"
ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[3])).resolve()
GRDC_BASE = ROOT / "09 GRDC交叉验证" / "GRDC"
GSIM_PRODUCT1 = ROOT / "08_GSIM_PLUS_Product" / "DTRR_Guarded_Anchor" / "GSIM_fill_anchor"
GSIM_PRODUCT2 = ROOT / "08_GSIM_PLUS_Product" / "dtrr_guarded" / "GSIM_fill"
GSIM_ATTR = ROOT / "999 material" / "GSIM_attribute.csv"

CONTINENT_DIR = GRDC_BASE / f"{CONTINENT}-GRDC"
OUTDIR = CONTINENT_DIR / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Processing {CONTINENT}...")
print(f"Loading GSIM attributes...")
gsim_attr = pd.read_csv(GSIM_ATTR)
gsim_attr = gsim_attr[['gsim.no', 'latitude', 'longitude']].rename(columns={'gsim.no': 'station_id'})
print(f"Loaded {len(gsim_attr)} GSIM stations")

def parse_grdc_txt(filepath):
    with open(filepath, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    # Extract metadata
    lat, lon = None, None
    for line in lines:
        if line.startswith('# Latitude'):
            lat = float(line.split(':')[1].strip())
        elif line.startswith('# Longitude'):
            lon = float(line.split(':')[1].strip())

    # Find data
    data_start = next((i+1 for i, l in enumerate(lines) if l.startswith('# YYYY-MM-DD')), 0)
    data = []
    for line in lines[data_start:]:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split(';')
            if len(parts) >= 3:
                try:
                    # Try Original column first (parts[2]), then Calculated (parts[3])
                    val = float(parts[2])
                    if val == -999.0 and len(parts) >= 4:
                        val = float(parts[3])
                    if val != -999.0:
                        data.append({'date': parts[0], 'MEAN': val})
                except:
                    continue
    return pd.DataFrame(data), lat, lon

# Find GRDC files
grdc_dirs = list(CONTINENT_DIR.glob("**/2026-*"))
if not grdc_dirs:
    # Try direct folder
    grdc_files = list(CONTINENT_DIR.glob("*_Q_Month.txt"))
    if not grdc_files:
        print(f"No GRDC data found")
        sys.exit(1)
else:
    grdc_dir = grdc_dirs[0]
    grdc_files = list(grdc_dir.glob("*_Q_Month.txt"))
print(f"Found {len(grdc_files)} GRDC stations")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 6.5, "legend.frameon": False, "mathtext.fontset": "stix",
})

summary = []
for grdc_file in grdc_files:
    grdc_id = grdc_file.stem.replace('_Q_Month', '')
    grdc, lat, lon = parse_grdc_txt(grdc_file)

    if len(grdc) == 0 or lat is None or lon is None:
        print(f"GRDC {grdc_id}: Skipped - no data or no coordinates")
        continue

    grdc['date'] = pd.to_datetime(grdc['date'])
    grdc = grdc[(grdc['date'] >= '1995-01-01') & (grdc['date'] <= '2015-12-31')]
    if len(grdc) < 12:
        print(f"GRDC {grdc_id}: Skipped - only {len(grdc)} months in 1995-2015")
        continue
    grdc['ym'] = grdc['date'].dt.to_period('M')

    # Find nearest GSIM station
    dist = np.sqrt((gsim_attr['latitude'] - lat)**2 + (gsim_attr['longitude'] - lon)**2)
    nearest_idx = dist.idxmin()
    if dist[nearest_idx] > 0.5:  # >0.5 degree, skip
        continue

    gsim_id = gsim_attr.loc[nearest_idx, 'station_id']
    print(f"\nGRDC {grdc_id} -> GSIM {gsim_id} (dist={dist[nearest_idx]:.3f}°)")

    # Load GSIM - try both folders
    gsim_path = GSIM_PRODUCT1 / f"{gsim_id}.csv"
    if not gsim_path.exists():
        gsim_path = GSIM_PRODUCT2 / f"{gsim_id}.csv"
    if not gsim_path.exists():
        print(f"  GSIM file not found")
        continue

    gsim = pd.read_csv(gsim_path)
    gsim['date'] = pd.to_datetime(gsim['date'])
    gsim['ym'] = gsim['date'].dt.to_period('M')

    merged = gsim.merge(grdc[['ym', 'MEAN']], on='ym', how='inner')
    if len(merged) == 0:
        print(f"  No overlap")
        continue

    fill_mask = merged['fill_method'] != 'OBSERVED'
    if fill_mask.sum() == 0:
        print(f"  No filled points")
        continue

    gsim_v = merged.loc[fill_mask, 'final_streamflow'].values
    grdc_v = merged.loc[fill_mask, 'MEAN'].values

    mae = np.mean(np.abs(gsim_v - grdc_v))
    rmse = np.sqrt(np.mean((gsim_v - grdc_v) ** 2))
    r_fill = np.corrcoef(gsim_v, grdc_v)[0, 1] if len(gsim_v) > 1 else np.nan
    nse_fill = 1 - np.sum((gsim_v - grdc_v)**2) / np.sum((grdc_v - np.mean(grdc_v))**2) if np.sum((grdc_v - np.mean(grdc_v))**2) > 0 else np.nan

    summary.append({'grdc_id': grdc_id, 'gsim_id': gsim_id, 'n_filled': int(fill_mask.sum()),
                    'mae': mae, 'rmse': rmse, 'nse_fill': nse_fill, 'r_fill': r_fill})

    print(f"  n={fill_mask.sum()}, NSE={nse_fill:.3f}, R={r_fill:.3f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(180/25.4, 100/25.4), dpi=300,
                                     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})

    fill_dates = merged.loc[fill_mask, 'date']
    t_min = fill_dates.min() - pd.DateOffset(months=12)
    t_max = fill_dates.max() + pd.DateOffset(months=12)
    zoom = merged[(merged['date'] >= t_min) & (merged['date'] <= t_max)]

    ax1.plot(zoom['date'], zoom['MEAN'], color='#3C5488', lw=1.0, alpha=0.9, label='GRDC', zorder=2)
    ax1.plot(zoom['date'], zoom['final_streamflow'], color='#E64B35', lw=1.0, alpha=0.9, label='GSIM-PLUS', zorder=3)

    for _, r in merged[fill_mask].iterrows():
        ax1.axvspan(r['date'] - pd.Timedelta(days=15), r['date'] + pd.Timedelta(days=15),
                     facecolor='#E64B3520', edgecolor='none', zorder=0)

    ax1.set_ylabel(r'Streamflow (m$^3$/s)', fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', fontsize=6.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.set_xticklabels([])

    metrics_str = f"n={fill_mask.sum()}, R={r_fill:.3f}, NSE={nse_fill:.3f}, RMSE={rmse:.1f}"
    ax1.text(0.01, 0.95, f"GRDC:{grdc_id} / GSIM:{gsim_id}\n{metrics_str}",
             transform=ax1.transAxes, fontsize=6, va='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', lw=0.3, alpha=0.9))

    ax2.scatter(merged.loc[fill_mask, 'MEAN'], merged.loc[fill_mask, 'final_streamflow'],
                s=10, color='#E64B35', alpha=0.6, edgecolors='none')
    lim_max = max(merged.loc[fill_mask, 'MEAN'].max(), merged.loc[fill_mask, 'final_streamflow'].max())
    ax2.plot([0, lim_max], [0, lim_max], 'k--', lw=0.8, alpha=0.5)
    ax2.set_xlabel(r'GRDC (m$^3$/s)', fontsize=8)
    ax2.set_ylabel(r'GSIM-PLUS (m$^3$/s)', fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUTDIR / f"{CONTINENT}_{grdc_id}_{gsim_id}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

pd.DataFrame(summary).to_csv(OUTDIR / f"{CONTINENT}_summary.csv", index=False)
print(f"\n\nProcessed {len(summary)} stations, saved to {OUTDIR}")


