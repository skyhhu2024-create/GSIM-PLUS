"""
Generate comparison CSV for all GRDC-GSIM stations from 6 regions
"""
import pandas as pd
import os
import numpy as np
from pathlib import Path

ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[4])).resolve()
GRDC_BASE = ROOT / "09 GRDC交叉验证" / "GRDC"
GSIM_PRODUCT1 = ROOT / "08_GSIM_PLUS_Product" / "DTRR_Guarded_Anchor" / "GSIM_fill_anchor"
GSIM_PRODUCT2 = ROOT / "08_GSIM_PLUS_Product" / "dtrr_guarded" / "GSIM_fill"
OUTDIR = ROOT / "09 GRDC交叉验证" / "4stations_comparison" / "Global"

def parse_grdc_txt(filepath):
    with open(filepath, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    data_start = next((i+1 for i, l in enumerate(lines) if l.startswith('# YYYY-MM-DD')), 0)
    data = []
    for line in lines[data_start:]:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split(';')
            if len(parts) >= 3:
                try:
                    val = float(parts[2])
                    if val == -999.0 and len(parts) >= 4:
                        val = float(parts[3])
                    if val != -999.0:
                        data.append({'date': parts[0], 'MEAN': val})
                except:
                    continue
    return pd.DataFrame(data)

# Load all stations from 6 regions
all_stations = []
for continent in ['AS', 'AU', 'SA', 'AF']:
    summary_file = GRDC_BASE / f"{continent}-GRDC" / "plots" / f"{continent}_summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df['continent'] = continent
        all_stations.append(df)
        print(f"Loaded {continent}: {len(df)} stations")

for region in ['Europe', 'USA']:
    summary_file = ROOT / "09 GRDC交叉验证" / region / "validation_summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df['continent'] = region
        all_stations.append(df)
        print(f"Loaded {region}: {len(df)} stations")

all_stations_df = pd.concat(all_stations, ignore_index=True)
print(f"\nTotal: {len(all_stations_df)} stations\n")

success_count = 0
for idx, row in all_stations_df.iterrows():
    gsim_id = row['gsim_id']
    grdc_id = str(row['grdc_id'])
    continent = row['continent']

    csv_file = OUTDIR / f"{gsim_id}_{grdc_id}_comparison.csv"
    if csv_file.exists():
        continue

    # Load GRDC
    if continent in ['Europe', 'USA']:
        grdc_dir = GRDC_BASE / continent
    else:
        grdc_dir = GRDC_BASE / f"{continent}-GRDC"

    grdc_files = list(grdc_dir.glob(f"**/{grdc_id}_Q_Month.txt"))
    if len(grdc_files) == 0:
        continue
    grdc = parse_grdc_txt(grdc_files[0])
    if len(grdc) == 0:
        continue

    grdc['date'] = pd.to_datetime(grdc['date'])
    grdc = grdc[(grdc['date'] >= '1995-01-01') & (grdc['date'] <= '2015-12-31')]
    grdc['ym'] = grdc['date'].dt.to_period('M')

    # Load GSIM
    gsim_path = GSIM_PRODUCT1 / f"{gsim_id}.csv"
    if not gsim_path.exists():
        gsim_path = GSIM_PRODUCT2 / f"{gsim_id}.csv"
    if not gsim_path.exists():
        continue

    gsim = pd.read_csv(gsim_path)
    gsim['date'] = pd.to_datetime(gsim['date'])
    gsim['ym'] = gsim['date'].dt.to_period('M')

    merged = gsim.merge(grdc[['ym', 'MEAN']], on='ym', how='inner')
    if len(merged) == 0:
        continue

    output = merged[['date', 'year', 'month', 'MEAN', 'final_streamflow', 'fill_method', 'quality_flag']]
    output = output.rename(columns={'MEAN': 'grdc_observed', 'final_streamflow': 'gsim_filled'})
    output.to_csv(csv_file, index=False)
    success_count += 1

    if success_count % 50 == 0:
        print(f"Processed {success_count} stations...")

print(f"\nTotal: {success_count} comparison files saved")


