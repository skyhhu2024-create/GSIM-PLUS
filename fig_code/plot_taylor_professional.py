"""
Generate professional Taylor diagram for GRDC validation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__import__("os").environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
DATADIR = ROOT / "09 GRDC交叉验证" / "4stations_comparison"
OUTDIR = ROOT / "111-paper" / "GRDC独立验证"

# Load all comparison CSV files
csv_files = sorted(list(DATADIR.glob("*_comparison.csv")))
all_data = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    fill_mask = df['fill_method'] != 'OBSERVED'

    if fill_mask.sum() >= 1:
        gsim_v = df.loc[fill_mask, 'gsim_filled'].values
        grdc_v = df.loc[fill_mask, 'grdc_observed'].values

        r = np.corrcoef(gsim_v, grdc_v)[0, 1] if len(gsim_v) > 1 else 0
        std_gsim = np.std(gsim_v, ddof=1) if len(gsim_v) > 1 else 0
        std_grdc = np.std(grdc_v, ddof=1) if len(grdc_v) > 1 else 1

        station_name = csv_file.stem.replace('_comparison', '')
        station_parts = station_name.split('_')
        station_label = '_'.join(station_parts[:2]) if len(station_parts) >= 2 else station_parts[0]

        all_data.append({
            'name': station_label,
            'r': r,
            'std_model': std_gsim,
            'std_ref': std_grdc
        })

print(f"Loaded {len(all_data)} stations")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10,
    "axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1,
    "mathtext.fontset": "stix",
})

# Create Taylor diagram
fig = plt.figure(figsize=(140/25.4, 120/25.4), dpi=600)
ax = fig.add_subplot(111, projection='polar')

# Plot reference point at (1, 0) - smaller and red
ax.plot(0, 1, marker='*', ls='none', color='#000000', markersize=11,
        label='Reference', zorder=10, markeredgecolor='white', markeredgewidth=0.5)

# Plot data points
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
          "#56B4E9", "#D55E00", "#F0E442", "#000000"]
markers = ["o", "s"]
for i, data in enumerate(all_data):
    theta = np.arccos(np.clip(data['r'], -1, 1))
    r = data['std_model'] / data['std_ref'] if data['std_ref'] > 0 else 0
    ax.plot(theta, r, marker=markers[i // len(colors)], ls='none', markersize=7,
            color=colors[i % len(colors)], alpha=0.9, markeredgecolor='white',
            markeredgewidth=0.6, label=data['name'], zorder=5)

# Set limits and labels
ax.set_thetamax(90)
ax.set_ylim(0, 1.8)
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')

# Correlation labels
corr_ticks = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
theta_ticks = [np.arccos(c) for c in corr_ticks]
ax.set_xticks(theta_ticks)
ax.set_xticklabels([f'{c:.2f}' for c in corr_ticks], fontsize=9)

# Std labels
ax.set_ylabel('Normalized Std', fontsize=10, labelpad=5)
ax.set_yticks([0.5, 1.0, 1.5])
ax.set_yticklabels(['0.5', '1.0', '1.5'], fontsize=9)

ax.set_title('Taylor Diagram', fontsize=11, pad=15, fontweight='bold')
ax.grid(True, linewidth=0.5, alpha=0.4, linestyle='--')

# Legend - move up to avoid blocking numbers
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.12), fontsize=8,
          ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.3)

plt.tight_layout()
fig.savefig(OUTDIR / "Fig_GRDC_taylor_v2.png", dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUTDIR / "Fig_GRDC_taylor_v2.pdf", bbox_inches='tight')
plt.close(fig)
print("Saved professional Taylor diagram")

