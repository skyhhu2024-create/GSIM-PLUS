"""
Generate publication-quality GRDC validation figures for 16 stations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT = Path(__import__("os").environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[2])).resolve()
DATADIR = ROOT / "09 GRDC交叉验证" / "4stations_comparison"
OUTDIR = ROOT / "111-paper" / "GRDC独立验证"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Load all comparison CSV files
csv_files = sorted(list(DATADIR.glob("*_comparison.csv")))
print(f"Found {len(csv_files)} comparison files")

all_data = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])

    fill_mask = df['fill_method'] != 'OBSERVED'

    # Don't filter out stations with few filled points
    gsim_v = df.loc[fill_mask, 'gsim_filled'].values
    grdc_v = df.loc[fill_mask, 'grdc_observed'].values

    if len(gsim_v) >= 1:  # Keep all stations with at least 1 filled point
        r = np.corrcoef(gsim_v, grdc_v)[0, 1] if len(gsim_v) > 1 else 0
        std_gsim = np.std(gsim_v) if len(gsim_v) > 1 else 0
        std_grdc = np.std(grdc_v) if len(grdc_v) > 1 else 1

        station_name = csv_file.stem.replace('_comparison', '')

        all_data.append({
            'name': station_name,
            'df': df,
            'fill_mask': fill_mask,
            'gsim_v': gsim_v,
            'grdc_v': grdc_v,
            'r': r,
            'std_gsim': std_gsim,
            'std_grdc': std_grdc
        })

print(f"Loaded {len(all_data)} stations")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "legend.frameon": False, "mathtext.fontset": "stix",
})

# 1. Taylor Diagram
fig = plt.figure(figsize=(120/25.4, 120/25.4), dpi=300)
ax = fig.add_subplot(111, projection='polar')

colors = plt.cm.tab20(np.linspace(0, 1, len(all_data)))
for i, data in enumerate(all_data):
    theta = np.arccos(np.clip(data['r'], -1, 1))
    r = data['std_gsim'] / data['std_grdc'] if data['std_grdc'] > 0 else 0
    ax.plot(theta, r, 'o', markersize=7, color=colors[i], alpha=0.8,
            markeredgecolor='white', markeredgewidth=0.5)

ax.set_thetamax(90)
ax.set_ylim(0, 3)
ax.set_ylabel('Normalized Std', fontsize=10, labelpad=30)
ax.set_xlabel('R', fontsize=10, labelpad=10)
ax.set_title('Taylor Diagram', fontsize=11, pad=15, fontweight='bold')
ax.grid(True, linewidth=0.5, alpha=0.3)

plt.tight_layout()
fig.savefig(OUTDIR / "Fig_GRDC_taylor.png", dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTDIR / "Fig_GRDC_taylor.pdf", bbox_inches='tight')
plt.close(fig)
print("Saved Taylor diagram")

# 2. Scatter Plot - 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(240/25.4, 240/25.4), dpi=600)
axes = axes.flatten()

for i in range(16):
    ax = axes[i]
    if i < len(all_data):
        data = all_data[i]

        # Calculate NSE
        gsim_v = data['gsim_v']
        grdc_v = data['grdc_v']
        nse = 1 - np.sum((gsim_v - grdc_v)**2) / np.sum((grdc_v - np.mean(grdc_v))**2) if np.sum((grdc_v - np.mean(grdc_v))**2) > 0 else np.nan

        ax.scatter(grdc_v, gsim_v, s=12, color="#0072B2", alpha=0.75,
                   edgecolors='white', linewidths=0.25)

        max_val = max(grdc_v.max(), gsim_v.max())
        ax.plot([0, max_val], [0, max_val], 'k--', lw=0.8, alpha=0.5)
        ax.set_xlabel(r'GRDC (m$^3$/s)', fontsize=9)
        ax.set_ylabel(r'GSIM-PLUS (m$^3$/s)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Use full station name
        station_parts = data['name'].split('_')
        station_label = '_'.join(station_parts[:2]) if len(station_parts) >= 2 else station_parts[0]
        metrics_text = f"{station_label}\nNSE={nse:.2f}\nR={data['r']:.2f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', lw=0.5, alpha=0.9))
    else:
        ax.axis('off')

fig.suptitle('Scatter Plot - GRDC Reference Comparison', fontsize=12, fontweight='bold', y=0.995)
plt.tight_layout()
fig.savefig(OUTDIR / "Fig_GRDC_scatter.png", dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTDIR / "Fig_GRDC_scatter.pdf", bbox_inches='tight')
plt.close(fig)
print("Saved scatter plot")

# 3. Time Series - 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(240/25.4, 190/25.4), dpi=600)
axes = axes.flatten()

for i in range(16):
    ax = axes[i]
    if i < len(all_data):
        data = all_data[i]
        df = data['df']
        fill_mask = data['fill_mask']

        # Calculate metrics
        gsim_v = data['gsim_v']
        grdc_v = data['grdc_v']
        nse = 1 - np.sum((gsim_v - grdc_v)**2) / np.sum((grdc_v - np.mean(grdc_v))**2) if np.sum((grdc_v - np.mean(grdc_v))**2) > 0 else np.nan
        rmse = np.sqrt(np.mean((gsim_v - grdc_v)**2))
        n = len(gsim_v)

        ax.plot(df['date'], df['grdc_observed'], color='#222222', lw=0.9,
                alpha=0.9, label='GRDC')
        ax.plot(df['date'], df['gsim_filled'], color='#0072B2', lw=0.9,
                ls='--', alpha=0.95, label='GSIM-PLUS')

        for _, r in df[fill_mask].iterrows():
            ax.axvspan(r['date'] - pd.Timedelta(days=15), r['date'] + pd.Timedelta(days=15),
                       facecolor='#E69F00', edgecolor='none', zorder=0, alpha=0.12)

        ax.set_ylabel(r'Q (m$^3$/s)', fontsize=9)
        ax.tick_params(labelsize=8)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Use full station name from comparison file
        station_parts = data['name'].split('_')
        station_label = '_'.join(station_parts[:2]) if len(station_parts) >= 2 else station_parts[0]
        metrics_text = f"{station_label}\nNSE={nse:.2f}, R={data['r']:.2f}\nRMSE={rmse:.1f}, n={n}"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', lw=0.5, alpha=0.9))

    else:
        ax.axis('off')

fig.suptitle('Time Series - GRDC Reference Comparison', fontsize=12, fontweight='bold', y=0.995)
fig.legend(
    handles=[
        plt.Line2D([0], [0], color='#222222', lw=1.2, label='GRDC'),
        plt.Line2D([0], [0], color='#0072B2', lw=1.2, ls='--', label='GSIM-PLUS'),
    ],
    loc='upper center', bbox_to_anchor=(0.5, 0.975), ncol=2, fontsize=8,
    frameon=False,
)
plt.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUTDIR / "Fig_GRDC_timeseries.png", dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTDIR / "Fig_GRDC_timeseries.pdf", bbox_inches='tight')
plt.close(fig)
print("Saved time series")

print(f"\nAll figures saved to {OUTDIR}")
print(f"Total stations: {len(all_data)}")




