"""
Calculate metrics for 16 GRDC validation stations
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__import__("os").environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[2])).resolve()
DATADIR = ROOT / "09 GRDC交叉验证" / "4stations_comparison"
OUTDIR = ROOT / "111-paper" / "GRDC独立验证"

csv_files = sorted(list(DATADIR.glob("*_comparison.csv")))
results = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    fill_mask = df['fill_method'] != 'OBSERVED'

    if fill_mask.sum() >= 1:
        gsim_v = df.loc[fill_mask, 'gsim_filled'].values
        grdc_v = df.loc[fill_mask, 'grdc_observed'].values

        # Metrics
        n_filled = len(gsim_v)
        mean_flow = np.mean(grdc_v)
        r = np.corrcoef(gsim_v, grdc_v)[0, 1] if len(gsim_v) > 1 else 0
        r2 = r ** 2
        nse = 1 - np.sum((gsim_v - grdc_v)**2) / np.sum((grdc_v - np.mean(grdc_v))**2)
        rmse = np.sqrt(np.mean((gsim_v - grdc_v)**2))
        mae = np.mean(np.abs(gsim_v - grdc_v))

        station_name = csv_file.stem.replace('_comparison', '')
        station_parts = station_name.split('_')
        station_label = '_'.join(station_parts[:2])

        results.append({
            'Station': station_label,
            'Gap_length': n_filled,
            'Mean_flow': mean_flow,
            'Fill_NSE': nse,
            'Fill_R': r,
            'Fill_R2': r2,
            'Fill_RMSE': rmse,
            'Fill_MAE': mae
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Station')
results_df.to_csv(OUTDIR / "16stations_metrics.csv", index=False, float_format='%.4f')

print(results_df.to_string(index=False))
print(f"\nSaved to {OUTDIR / '16stations_metrics.csv'}")
