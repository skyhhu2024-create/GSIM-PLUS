"""
Generate GRDC-GSIM matching table based on coordinates
"""
import numpy as np
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.environ.get("GSIM_PLUS_PROJECT_DIR", Path(__file__).resolve().parents[3])).resolve()
GRDC_BASE = ROOT / "09 GRDC交叉验证" / "GRDC"
GSIM_ATTR = ROOT / "999 material" / "GSIM_attribute.csv"
OUTDIR = ROOT / "09 GRDC交叉验证" / "GRDC"

print("Loading GSIM attributes...")
gsim_attr = pd.read_csv(GSIM_ATTR)
gsim_attr = gsim_attr[['gsim.no', 'latitude', 'longitude', 'river', 'station', 'country']].rename(
    columns={'gsim.no': 'gsim_id', 'latitude': 'gsim_lat', 'longitude': 'gsim_lon',
             'river': 'gsim_river', 'station': 'gsim_station', 'country': 'gsim_country'})

def parse_grdc_metadata(filepath):
    with open(filepath, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    metadata = {}
    for line in lines:
        if line.startswith('# GRDC-No.:'):
            metadata['grdc_id'] = line.split(':')[1].strip()
        elif line.startswith('# River:'):
            metadata['grdc_river'] = line.split(':')[1].strip()
        elif line.startswith('# Station:'):
            metadata['grdc_station'] = line.split(':')[1].strip()
        elif line.startswith('# Country:'):
            metadata['grdc_country'] = line.split(':')[1].strip()
        elif line.startswith('# Latitude'):
            metadata['grdc_lat'] = float(line.split(':')[1].strip())
        elif line.startswith('# Longitude'):
            metadata['grdc_lon'] = float(line.split(':')[1].strip())
        elif line.startswith('# Catchment area'):
            try:
                metadata['grdc_area'] = float(line.split(':')[1].strip())
            except:
                metadata['grdc_area'] = None
    return metadata

all_matches = []
continents = ['AS', 'AU', 'SA', 'AF']

for continent in continents:
    print(f"\nProcessing {continent}...")
    continent_dir = GRDC_BASE / f"{continent}-GRDC"
    grdc_dirs = list(continent_dir.glob("**/2026-*"))
    if not grdc_dirs:
        continue

    grdc_dir = grdc_dirs[0]
    grdc_files = list(grdc_dir.glob("*_Q_Month.txt"))
    print(f"  Found {len(grdc_files)} GRDC stations")

    for grdc_file in grdc_files:
        metadata = parse_grdc_metadata(grdc_file)
        if 'grdc_lat' not in metadata or 'grdc_lon' not in metadata:
            continue

        lat, lon = metadata['grdc_lat'], metadata['grdc_lon']
        dist = np.sqrt((gsim_attr['gsim_lat'] - lat)**2 + (gsim_attr['gsim_lon'] - lon)**2)
        nearest_idx = dist.idxmin()
        min_dist = dist[nearest_idx]

        match = {
            'continent': continent,
            'grdc_id': metadata.get('grdc_id', ''),
            'grdc_river': metadata.get('grdc_river', ''),
            'grdc_station': metadata.get('grdc_station', ''),
            'grdc_country': metadata.get('grdc_country', ''),
            'grdc_lat': lat,
            'grdc_lon': lon,
            'grdc_area_km2': metadata.get('grdc_area', None),
            'gsim_id': gsim_attr.loc[nearest_idx, 'gsim_id'],
            'gsim_river': gsim_attr.loc[nearest_idx, 'gsim_river'],
            'gsim_station': gsim_attr.loc[nearest_idx, 'gsim_station'],
            'gsim_country': gsim_attr.loc[nearest_idx, 'gsim_country'],
            'gsim_lat': gsim_attr.loc[nearest_idx, 'gsim_lat'],
            'gsim_lon': gsim_attr.loc[nearest_idx, 'gsim_lon'],
            'distance_deg': min_dist,
            'match_quality': 'Good' if min_dist < 0.1 else 'Fair' if min_dist < 0.5 else 'Poor'
        }
        all_matches.append(match)

match_df = pd.DataFrame(all_matches)
match_df = match_df.sort_values(['continent', 'distance_deg'])
match_df.to_csv(OUTDIR / "GRDC_GSIM_matching_table.csv", index=False)

print(f"\n\nTotal matches: {len(match_df)}")
print(f"Good matches (<0.1°): {(match_df['distance_deg'] < 0.1).sum()}")
print(f"Fair matches (0.1-0.5°): {((match_df['distance_deg'] >= 0.1) & (match_df['distance_deg'] < 0.5)).sum()}")
print(f"Poor matches (>0.5°): {(match_df['distance_deg'] >= 0.5).sum()}")
print(f"\nSaved to: {OUTDIR / 'GRDC_GSIM_matching_table.csv'}")
