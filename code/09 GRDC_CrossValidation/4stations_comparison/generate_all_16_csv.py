"""
Generate comparison CSV for all 16 stations
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[3])).resolve()
GRDC_BASE = ROOT / "09 GRDC交叉验证" / "GRDC"
GSIM_PRODUCT1 = ROOT / "08_GSIM_PLUS_Product" / "DTRR_Guarded_Anchor" / "GSIM_fill_anchor"
GSIM_PRODUCT2 = ROOT / "08_GSIM_PLUS_Product" / "dtrr_guarded" / "GSIM_fill"
OUTDIR = ROOT / "09 GRDC交叉验证" / "4stations_comparison"

# Station mapping (corrected)
stations = {
    'FR_0001112': {'grdc_id': '6135110', 'continent': 'Europe'},
    'US_0005774': {'grdc_id': '4125804', 'continent': 'USA'},
    'US_0002812': {'grdc_id': '4123245', 'continent': 'USA'},
    'US_0004183': {'grdc_id': '4119313', 'continent': 'USA'},
    'AU_0000817': {'grdc_id': '5302301', 'continent': 'AU'},
    'AU_0000756': {'grdc_id': '5302265', 'continent': 'AU'},  # Changed from AU_0000781
    'AU_0000768': {'grdc_id': '5302276', 'continent': 'AU'},
    'BR_0000649': {'grdc_id': '3650640', 'continent': 'SA'},
    'BR_0000276': {'grdc_id': '3637180', 'continent': 'SA'},
    'BR_0000662': {'grdc_id': '3650649', 'continent': 'SA'},
    'JP_0000268': {'grdc_id': '2588702', 'continent': 'AS'},
    'JP_0000805': {'grdc_id': '2589201', 'continent': 'AS'},
    'IN_0000071': {'grdc_id': '2853200', 'continent': 'AS'},  # Corrected from JP
    'ZA_0000030': {'grdc_id': '1159805', 'continent': 'AF'},  # First match
    'ZA_0000048': {'grdc_id': '1160230', 'continent': 'AF'},
    'ZA_0000051': {'grdc_id': '1160245', 'continent': 'AF'},  # First match
}

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

for gsim_id, info in stations.items():
    grdc_id = info['grdc_id']
    continent = info['continent']

    # Check if CSV already exists
    csv_file = OUTDIR / f"{gsim_id}_{grdc_id}_comparison.csv"
    if csv_file.exists():
        print(f"{gsim_id}: Already exists")
        continue

    # Load GRDC
    grdc_dir = GRDC_BASE / f"{continent}-GRDC"
    grdc_files = list(grdc_dir.glob(f"**/{grdc_id}_Q_Month.txt"))
    if len(grdc_files) == 0:
        print(f"{gsim_id}: GRDC file not found")
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
        print(f"{gsim_id}: GSIM file not found")
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

    print(f"{gsim_id}_{grdc_id}_comparison.csv saved")

print(f"\nAll files saved to {OUTDIR}")



